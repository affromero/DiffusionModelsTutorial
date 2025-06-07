"""Evaluation script for MNIST diffusion model.

This script evaluates the quality of generated samples:
1. Load trained model and generate samples
2. Compute quantitative metrics (FID, IS)
3. Analyze sample diversity and quality
4. Create evaluation reports

Usage:
    python scripts/evaluate.py --help
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from jaxtyping import Float
from pydantic.dataclasses import dataclass
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.data.dataset import create_data_loaders
from src.diffusion.scheduler import DDIMScheduler, DDPMScheduler, PNDMScheduler
from src.models.unet import UNet
from src.training.sampler import DiffusionSampler


class SimpleClassifier(torch.nn.Module):
    """Simple CNN classifier for evaluating generated samples."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the classifier.

        Args:
            num_classes: Number of classes.

        """
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(
        self, x: Float[torch.Tensor, "batch_size channels h w"]
    ) -> Float[torch.Tensor, "batch_size num_classes"]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, h, w).

        Returns:
            Output tensor of shape (batch_size, num_classes).

        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_trained_model(
    model_path: str, config: Config
) -> tuple[UNet, DDPMScheduler | DDIMScheduler | PNDMScheduler]:
    """Load trained diffusion model."""
    print(f"Loading model from: {model_path}")

    model = UNet(
        img_channels=config.model.img_channels,
        img_size=config.model.img_size,
        time_emb_dim=config.model.time_emb_dim,
        num_classes=config.model.num_classes,
        base_channels=config.model.base_channels,
        channel_mults=config.model.channel_mults,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    ).to(config.device)

    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config.scheduler


def train_classifier_on_real_data(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> torch.nn.Module:
    """Train a classifier on real MNIST data for evaluation."""
    print("Training classifier on real MNIST data...")

    classifier = SimpleClassifier().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Train for a few epochs
    classifier.train()
    for epoch in range(3):
        total_loss = 0
        correct = 0
        total = 0

        for _batch_idx, (data, target) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/3")
        ):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    # Evaluate on validation set
    classifier.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()
            val_total += target.size(0)

    val_accuracy = 100.0 * val_correct / val_total
    print(f"Classifier validation accuracy: {val_accuracy:.2f}%")

    return classifier


def evaluate_sample_quality(
    sampler: DiffusionSampler,
    classifier: SimpleClassifier,
    *,
    num_samples_per_class: int = 100,
    img_size: tuple[int, int],
    guidance_scale: float | None = None,
) -> tuple[float, dict[int, float], list[int], list[int]]:
    """Evaluate the quality of generated samples using a trained classifier."""
    print(
        f"Evaluating sample quality with {num_samples_per_class} samples per class..."
    )

    all_samples = []
    all_true_labels = []
    all_pred_labels = []

    classifier.eval()

    for digit in range(10):
        print(f"  Generating samples for digit {digit}...")

        # Generate samples for this digit
        class_labels = torch.full((num_samples_per_class,), digit, dtype=torch.long)
        samples = sampler.sample(
            num_samples=num_samples_per_class,
            class_labels=class_labels,
            show_progress=False,
            img_size=img_size,
            guidance_scale=guidance_scale,
        )

        # Classify generated samples
        with torch.no_grad():
            outputs = classifier(samples)
            predictions = outputs.argmax(dim=1)

        all_samples.append(samples.cpu())
        all_true_labels.extend([digit] * num_samples_per_class)
        all_pred_labels.extend(predictions.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_true_labels, all_pred_labels)

    # Per-class accuracy
    class_accuracies = {}
    for digit in range(10):
        digit_mask = np.array(all_true_labels) == digit
        digit_preds = np.array(all_pred_labels)[digit_mask]
        digit_accuracy = (digit_preds == digit).mean()
        class_accuracies[digit] = digit_accuracy

    return accuracy, class_accuracies, all_true_labels, all_pred_labels


def compute_inception_score(
    samples: torch.Tensor,
    classifier: SimpleClassifier,
    splits: int = 10,
    device: str = "cuda",
) -> tuple[float, float]:
    """Compute Inception Score for generated samples.

    Mathematical Background:
    IS = exp(E_x[KL(p(y|x) || p(y))])

    Where:
    - p(y|x) is the conditional label distribution
    - p(y) is the marginal label distribution
    - Higher IS indicates better quality and diversity
    """
    print("Computing Inception Score...")

    classifier.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(samples), 100):  # Process in batches
            batch = samples[i : i + 100].to(device)
            outputs = classifier(batch)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Compute IS
    split_scores = []

    for k in range(splits):
        part = all_probs[
            k * (len(all_probs) // splits) : (k + 1) * (len(all_probs) // splits)
        ]

        # p(y|x)
        py_x = part

        # p(y) = E_x[p(y|x)]
        py = np.mean(part, axis=0)

        # KL divergence
        kl_div = py_x * (np.log(py_x + 1e-16) - np.log(py + 1e-16))  # type: ignore[operator]
        kl_div = np.sum(kl_div, axis=1)

        # Inception Score for this split
        split_scores.append(np.exp(np.mean(kl_div)))

    is_mean = np.mean(split_scores)
    is_std = np.std(split_scores)

    return is_mean, is_std


def analyze_sample_diversity(
    samples: torch.Tensor, true_labels: torch.Tensor
) -> tuple[float, dict[int, float]]:
    """Analyze the diversity of generated samples."""
    print("Analyzing sample diversity...")

    # Convert to numpy for analysis
    samples_np = samples.cpu().numpy()

    # Compute pairwise distances within each class
    class_diversities = {}

    for digit in range(10):
        digit_mask = np.array(true_labels) == digit
        digit_samples = samples_np[digit_mask]

        if len(digit_samples) > 1:
            # Flatten samples for distance computation
            flattened = digit_samples.reshape(len(digit_samples), -1)

            # Compute pairwise L2 distances
            distances = []
            for i in range(len(flattened)):
                for j in range(i + 1, len(flattened)):
                    dist = np.linalg.norm(flattened[i] - flattened[j])
                    distances.append(dist)

            # Average intra-class distance (higher = more diverse)
            class_diversities[digit] = np.mean(distances)
        else:
            class_diversities[digit] = 0.0

    overall_diversity = np.mean(list(class_diversities.values()))

    return overall_diversity, class_diversities


def create_evaluation_report(
    *,
    accuracy: float,
    class_accuracies: dict[int, float],
    inception_score: tuple[float, float],
    diversity_metrics: tuple[float, dict[int, float]],
    output_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Create a comprehensive evaluation report."""
    print("Creating evaluation report...")

    # Prepare report data
    report_data = {
        "model_path": str(model_path),
        "overall_metrics": {
            "classification_accuracy": float(accuracy),
            "inception_score_mean": float(inception_score[0]),
            "inception_score_std": float(inception_score[1]),
            "overall_diversity": float(diversity_metrics[0]),
        },
        "per_class_metrics": {
            "accuracies": {str(k): float(v) for k, v in class_accuracies.items()},
            "diversities": {str(k): float(v) for k, v in diversity_metrics[1].items()},
        },
    }

    # Save JSON report
    report_path = output_dir / "evaluation_report.json"
    with Path(report_path).open("w") as f:
        json.dump(report_data, f, indent=4)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Per-class accuracy
    digits = list(range(10))
    accuracies = [class_accuracies[d] for d in digits]

    axes[0, 0].bar(digits, accuracies)
    axes[0, 0].set_title("Per-Class Classification Accuracy")
    axes[0, 0].set_xlabel("Digit Class")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0, 1)

    # Per-class diversity
    diversities = [diversity_metrics[1][d] for d in digits]

    axes[0, 1].bar(digits, diversities)
    axes[0, 1].set_title("Per-Class Sample Diversity")
    axes[0, 1].set_xlabel("Digit Class")
    axes[0, 1].set_ylabel("Average Pairwise Distance")

    # Overall metrics
    metrics_names = ["Accuracy", "IS Mean", "Diversity"]
    metrics_values = [accuracy, inception_score[0], diversity_metrics[0]]

    axes[1, 0].bar(metrics_names, metrics_values)
    axes[1, 0].set_title("Overall Quality Metrics")
    axes[1, 0].set_ylabel("Score")

    # Summary text
    axes[1, 1].axis("off")
    summary_text = f"""
    Evaluation Summary:

    Overall Accuracy: {accuracy:.3f}
    Inception Score: {inception_score[0]:.2f} ± {inception_score[1]:.2f}
    Sample Diversity: {diversity_metrics[0]:.3f}

    Best Performing Classes:
    {
        ", ".join(
            [
                str(i)
                for i in sorted(
                    class_accuracies.keys(),
                    key=lambda x: class_accuracies[x],
                    reverse=True,
                )[:3]
            ]
        )
    }

    Most Diverse Classes:
    {
        ", ".join(
            [
                str(i)
                for i in sorted(
                    diversity_metrics[1].keys(),
                    key=lambda x: diversity_metrics[1][x],
                    reverse=True,
                )[:3]
            ]
        )
    }
    """

    axes[1, 1].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()

    # Save visualization
    viz_path = output_dir / "evaluation_metrics.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Evaluation report saved to: {report_path}")
    print(f"Evaluation visualization saved to: {viz_path}")

    return report_data


@dataclass
class EvaluateArgs:
    """Evaluation arguments."""

    model_path: str = "outputs/models/best_model.pth"
    """Path to trained model checkpoint."""
    num_samples: int = 100
    """Number of samples per class for evaluation."""
    output_dir: str | None = None
    """Output directory (default: config.sample_dir)."""
    seed: int = 42
    """Random seed for sampling."""
    show_progress: bool = True
    """Show sampling progress bar."""


def main(args: EvaluateArgs, config: Config) -> None:
    """Run evaluation pipeline."""
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.output_dir) / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MNIST Diffusion Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {config.device}")
    print(f"Samples per class: {args.num_samples}")
    print(f"Output directory: {output_dir}")

    # Load data for training classifier
    print("\nLoading MNIST dataset...")
    train_loader, val_loader = create_data_loaders(
        data_root=config.data.data_root,
        img_size=config.data.resize,
        batch_size=128,
        num_workers=config.data.num_workers,
    )

    # Train classifier on real data
    classifier = train_classifier_on_real_data(train_loader, val_loader, config.device)

    # Load trained diffusion model
    model, scheduler = load_trained_model(args.model_path, config)
    sampler = DiffusionSampler(model, scheduler, config.device)

    # Evaluate sample quality
    accuracy, class_accuracies, _, _ = evaluate_sample_quality(
        sampler,
        classifier,
        num_samples_per_class=args.num_samples,
        img_size=config.model.img_size,
        guidance_scale=config.sampling.guidance_scale,
    )

    # Generate samples for other metrics
    print("Generating samples for additional metrics...")
    all_samples = []
    all_labels = []

    for digit in range(config.model.num_classes):
        class_labels = torch.full((args.num_samples,), digit, dtype=torch.long)
        samples = sampler.sample(
            num_samples=args.num_samples,
            class_labels=class_labels,
            show_progress=False,
            img_size=(config.data.resize, config.data.resize),
        )
        all_samples.append(samples)
        all_labels.extend([digit] * args.num_samples)

    all_samples = torch.cat(all_samples, dim=0)

    # Compute Inception Score
    inception_score = compute_inception_score(all_samples, classifier)

    # Analyze diversity
    diversity_metrics = analyze_sample_diversity(all_samples, all_labels)

    # Create evaluation report
    report = create_evaluation_report(
        accuracy=accuracy,
        class_accuracies=class_accuracies,
        inception_score=inception_score,
        diversity_metrics=diversity_metrics,
        output_dir=Path(output_dir),
        model_path=Path(args.model_path),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Classification Accuracy: {accuracy:.3f}")
    print(f"Inception Score: {inception_score[0]:.2f} ± {inception_score[1]:.2f}")
    print(f"Sample Diversity: {diversity_metrics[0]:.3f}")
    print("=" * 60)
    print(f"Report: {report}")
    print("=" * 60)


def run_task(model_path: str = "outputs/models/best_model.pth") -> None:
    """Run evaluation pipeline."""
    config = Config.load_pkl()
    main(EvaluateArgs(model_path=model_path), config=config)


if __name__ == "__main__":
    tyro.cli(run_task)
