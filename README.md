# Diffusion Models and Inpainting

This is a toy example using MNIST, so **no** latent diffusion, insted the diffusion is directly on the pixel level.

## Overview

This project implements a diffusion model for image generation, likely on the MNIST dataset, and explores advanced concepts such as Classifier-Free Guidance (CFG) and Score Distillation Sampling (SDS) for inpainting tasks. The codebase allows for training diffusion models, generating samples, evaluating model performance, and performing inpainting.

## Project Structure

Key directories in this project:

-   `src/`: Contains the core implementation of the diffusion models, data handling, training loops, and utility functions.
-   `scripts/`: Holds Python scripts that serve as entry points for various tasks like training, sampling, evaluation, and inpainting.
-   `outputs/`: Default directory for saving trained models, generated images, evaluation results, visualizations (including GIFs and plots).
-   `config.py`: (At the root) Crucial for managing all configurations.

Below is a more detailed view of the project's folder and file structure:

```
.
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── README.md
├── config.py
├── data/
│   ├── MNIST/
│   │   └── raw/
│   │       ├── t10k-images-idx3-ubyte
│   │       ├── t10k-images-idx3-ubyte.gz
│   │       ├── t10k-labels-idx1-ubyte
│   │       ├── t10k-labels-idx1-ubyte.gz
│   │       ├── train-images-idx3-ubyte
│   │       ├── train-images-idx3-ubyte.gz
│   │       ├── train-labels-idx1-ubyte
│   │       └── train-labels-idx1-ubyte.gz
│   ├── README.md
│   ├── anscombe.json
│   ├── california_housing_test.csv
│   ├── california_housing_train.csv
│   ├── mnist_test.csv
│   └── mnist_train_small.csv
├── outputs/
│   ├── config.pkl
│   ├── evaluation/
│   │   ├── evaluation_metrics.png
│   │   └── evaluation_report.json
│   ├── logs/
│   │   ├── denoise_steps_epoch_1.gif
│   │   ├── denoise_steps_epoch_2.gif
│   │   ├── denoise_steps_epoch_3.gif
│   │   ├── denoise_steps_epoch_4.gif
│   │   ├── denoise_steps_epoch_5.gif
│   │   └── training_curves.png
│   ├── models/
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch_1.pth
│   │   ├── checkpoint_epoch_2.pth
│   │   ├── checkpoint_epoch_3.pth
│   │   ├── checkpoint_epoch_4.pth
│   │   ├── checkpoint_epoch_5.pth
│   │   └── final_model.pth
│   ├── samples/
│   │   ├── all_classes_grid.png
│   │   ├── class_1_samples.png
│   │   ├── class_2_samples.png
│   │   ├── digit_0_samples.png
│   │   ├── digit_1_samples.png
│   │   ├── digit_2_samples.png
│   │   ├── digit_3_samples.png
│   │   ├── digit_4_samples.png
│   │   ├── digit_5_samples.png
│   │   ├── digit_6_samples.png
│   │   ├── digit_7_samples.png
│   │   ├── digit_8_samples.png
│   │   ├── digit_9_samples.png
│   │   └── generated_samples.png
│   └── sds_inpaint/
│       └── log.png
├── plots/
│   ├── alphas_cumprod_comparison.png
│   ├── betas_comparison.png
│   └── sqrt_one_minus_alphas_cumprod_comparison.png
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── plot_schedulers.py
│   ├── sampler.py
│   ├── sds_inpaint.py
│   └── train.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── diffusion/
│   │   ├── __init__.py
│   │   └── scheduler.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   └── utils.py
│   └── training/
│       ├── __init__.py
│       ├── sampler.py
│       └── trainer.py
└── uv.lock
```


## Configuration (`config.py`)

Key training and model parameters are defined in `config.py`. One important parameter is `TrainingConfig.diffusion_mode`, which can be set to:
- `"score_matching"`: Standard Denoising Diffusion Probabilistic Models (DDPM) objective.
- `"flow_matching"`: Learns the velocity field of a probability flow ODE, typically `x_1 - x_0` (noise - data).
- `"rectified_flow"`: Aims to learn a simplified, often straight, path between noise and data distributions. The specific implementation in this project targets `x_0 - x_1` (data - noise). The prediction then produces reverse colored images.


All experimental parameters, model configurations, training settings, and paths are managed centrally in `config.py`. This file uses Pydantic dataclasses for structured and type-checked configurations. Before running any script, you might want to review and adjust settings in `config.py` to suit your needs (e.g., batch size, learning rate, number of epochs, model architecture details, paths for saving outputs).

## Running the Code

The primary way to interact with this project is through the scripts located in the `scripts/` directory. All scripts are designed to be run from the root of the project directory.

### 1. Training the Diffusion Model

-   **Script:** `scripts/train.py`
-   **Purpose:** Trains the diffusion model based on the settings in `config.py`.
-   **Usage:**
    ```bash
    python scripts/train.py
    ```
-   **Details:** This script will initialize the model, data loaders, and trainer. It saves model checkpoints (e.g., `best_model.pth`, `latest_model.pth`) to the directory specified in `config.model_dir` (usually within `outputs/models/`). It might also generate some sample images at the end of training.

### 2. Generating Samples

-   **Script:** `scripts/sampler.py`
-   **Purpose:** Generates image samples from a trained diffusion model. It supports several generation modes: unconditional (generating random images from noise), conditional (generating images of specific classes if the model is class-conditional), and Classifier-Free Guidance (CFG) for enhanced class adherence and sample quality. Additionally, it can perform class-specific sampling, interpolation between classes, and **a demonstration of inpainting on newly sampled images.**
-   **Key Command-Line Arguments:**
    -   `--model_path <path>`: Path to the trained model checkpoint. **Defaults to `outputs/models/best_model.pth`. Examples below assume this default.**
    -   `--num_samples <int>`: Number of samples per class (default: 6).
    -   `--specific_class <int>`: Generate samples for a specific class only.
    -   `--output_dir <path>`: Directory to save samples (default: from `config.py`).
    -   `--seed <int>`: Random seed for sampling (default: 42).
    -   `--guidance_scale <float>`: Classifier-Free Guidance scale (default: 1.0). 0.0 for unconditional, 1.0 for standard conditional, >1.0 for stronger guidance.
    -   `--interpolate <int> <int>`: Create an interpolation between two specified classes (e.g., `--interpolate 3 8`).
    -   `--inpaint`: Enable a demo inpainting mode. When used with `--specific_class`, the script will first sample an image of that class, create a mask, and then attempt to inpaint the masked region using the model's generative capabilities.
    -   `--show_progress <bool>`: Show a progress bar during sampling (default: True).
-   **Usage Examples:** (Assuming `outputs/models/best_model.pth` exists or `--model_path` is specified)
    -   Generate default samples for all classes:
        ```bash
        python scripts/sampler.py
        ```
    -   Generate 10 samples for digit '7' with strong guidance:
        ```bash
        python scripts/sampler.py --specific_class 7 --num_samples 10 --guidance_scale 7.5
        ```
    -   Create an interpolation between digit '2' and '9':
        ```bash
        python scripts/sampler.py --interpolate 2 9
        ```
    -   Demonstrate inpainting on a newly sampled image of digit '5':
        ```bash
        python scripts/sampler.py --specific_class 5 --inpaint --num_samples 1
        ```
-   **Details:** Saves generated images (grids, individual samples, interpolations, inpainting demos) to the specified output directory (usually within `outputs/samples/`). For more advanced Score Distillation Sampling (SDS) based inpainting on existing images, see `scripts/sds_inpaint.py`.

### 3. Evaluating Model Performance

-   **Script:** `scripts/evaluate.py`
-   **Purpose:** Evaluates a trained model, often by generating a large number of samples and calculating metrics like Frechet Inception Distance (FID) if applicable, or by providing qualitative sample sets.
-   **Usage:** (Assumes `outputs/models/best_model.pth` exists or `--checkpoint_path` is specified)
    ```bash
    python scripts/evaluate.py
    ```
-   **Details:** Uses the configuration associated with the loaded checkpoint. Results and generated images for evaluation are typically saved in `outputs/eval/`.

### 4. Score Distillation Sampling (SDS) for Inpainting

-   **Script:** `scripts/sds_inpaint.py`
-   **Purpose:** Demonstrates inpainting using Score Distillation Sampling. It takes an image and a mask, and optimizes the masked region to match a target class or description using a pre-trained diffusion model as a prior.
-   **Usage:** (Assumes `outputs/models/best_model.pth` exists or `--model_path` is specified)
    ```bash
    python scripts/sds_inpaint.py
    ```
-   **Details:** The script will load an image, create a mask, perform the SDS optimization, and save a visualization (e.g., `outputs/sds_inpaint/log.png`) showing the original, mask, masked input, and inpainted result.

### 5. Plotting Scheduler Noise Levels

-   **Script:** `scripts/plot_schedulers.py`
-   **Purpose:** Visualizes the noise levels (alpha and sigma schedules) for different diffusion schedulers (DDPM, DDIM, PNDM).
-   **Usage:**
    ```bash
    python scripts/plot_schedulers.py
    ```
-   **Details:** Saves the generated plots to `outputs/scheduler_plots/`. This is useful for understanding and comparing the noise schedules used in the diffusion process.

## Outputs

Generated content, including trained models, sample images, evaluation data, and visualizations, are stored in the `outputs/` directory. Key subdirectories include:

-   `outputs/models/`: Trained model checkpoints.
-   `outputs/samples/`: General image samples, including those from `sampler.py` and potentially end-of-training samples.
-   `outputs/eval/`: Images and metrics from the evaluation script.
-   `outputs/sds_inpaint/`: Visualizations from the SDS inpainting script (e.g., `log.png`).
-   `outputs/scheduler_plots/`: Plots of noise schedules.
-   Look for `.png` images, `.gif` animations (if any are generated by specific scripts), and model files like `.pth`.

## Dependencies and Setup

This project is developed with Python 3.10.16.

It is recommended to use a virtual environment. If you are using `uv` (from Astral), dependency management should be very smooth. You will likely find a `pyproject.toml` file that `uv` can use, or a `requirements.txt` file.

**General Setup Steps:**

1.  **Clone the repository.**
2.  **Create and activate a virtual environment:**
    *   Using `venv` (standard Python):
        ```bash
        python3.10 -m venv .venv
        source .venv/bin/activate
        ```
    *   If using `conda` or other environment managers, follow their respective procedures.
3.  **Install dependencies:**
    *   If using `uv` with a `pyproject.toml`:
        ```bash
        uv sync
        ```
    *   Or, with `requirements.txt`:
        ```bash
        uv pip install -r requirements.txt
        # or
        # pip install -r requirements.txt
        ```

(Ensure you have the necessary CUDA12.1. If you don't have it, you can install it from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads). If you have a different one, please change it in the `pyproject.toml` file.)

---
