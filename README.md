# Machine Learning Take-Home Exercise: Diffusion Models and Inpainting

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
-   **Purpose:** Generates image samples from a trained diffusion model. Supports various modes including Classifier-Free Guidance (CFG), class-specific sampling, interpolation, and **a demonstration of inpainting on newly sampled images.**
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

This README provides a guide to navigating and using the project. Refer to individual script help (`python scripts/<script_name>.py --help`) for more detailed options if available.

## Project Retrospective: Journey, Challenges, and Future Directions

This section provides a brief reflection on the development process, highlighting key learnings, debugging experiences, and potential avenues for future exploration.

### What Was Tried and What Worked

*   **Core Diffusion Model (DDPM-like):** A foundational diffusion model was successfully implemented, capable of generating MNIST digits. This involved setting up the UNet architecture, noise schedulers (DDPM, DDIM, PNDM), and the forward/reverse diffusion processes. The training loop, data loading, and basic sampling worked as expected, producing recognizable digits.
*   **Classifier-Free Guidance (CFG):** CFG was a key feature successfully integrated into the pipeline. This involved:
    *   Modifying the UNet to optionally accept class labels, using zero embeddings for unconditional passes.
    *   Updating the `DiffusionSampler` to perform both conditional and unconditional model predictions and combine them using the `guidance_scale`.
    *   Propagating the `guidance_scale` parameter through CLI arguments, configuration, and relevant sampling/evaluation functions.
    This significantly improved the visual quality and class adherence of generated samples.
*   **Classifier Dropout for CFG Training:** To enable effective CFG, a classifier dropout mechanism (`p_unconditional`) was introduced during training. This randomly drops class labels for a portion of training steps, forcing the model to learn unconditional generation alongside conditional generation.
*   **Score Distillation Sampling (SDS) for Inpainting:** An SDS-based inpainting module (`scripts/sds_inpaint.py`) was developed. This involved:
    *   Defining an SDS loss function that uses a pre-trained diffusion model as a prior to guide the optimization of masked image regions.
    *   Setting up an optimization loop that iteratively refines the inpainted area.
    *   Visualizing the original image, mask, masked input, and the final inpainted result. This demonstrated the potential of using diffusion models as powerful priors for image editing tasks.
*   **Configuration Management:** Using Pydantic dataclasses in `config.py` provided a structured and type-safe way to manage all experiment parameters, which was beneficial for reproducibility and clarity.
*   **Scripting and Visualization:** Dedicated scripts for training, sampling, evaluation, SDS inpainting, and scheduler plotting made the project modular and easy to use. Matplotlib was used effectively for various visualizations, including sample grids, training logs, and inpainting results.

### Debugging Journey

*   **Different Scheduler for Inpainting:** The inpainting script used a different scheduler (DDIM) than the training script (DDPM). This led to different noise schedules and could affect the quality of the inpainted results. To resolve this, the inpainting script was modified to use the same scheduler as the training script. I spent here a lot of time debugging the inpainting script thinking there was a bug due to the images looking saturated like with a clamping issue. After reading in the internet I found that DDIM is more suitable for inpainting tasks.
*   **Linter and Mypy Errors:** Throughout development, `ruff` (for linting) and `mypy` (for type checking) were used. Addressing their feedback (e.g., unused variables, incorrect type hints, missing arguments in function calls) helped maintain code quality and catch potential bugs early. For instance, a Mypy error in `scripts/train.py` regarding `Config` instantiation was resolved by ensuring all required arguments were provided. See `.pre-commit-config.yaml` for more details.

### If I Had More Time

*   **Advanced Inpainting Techniques:**
    *   Explore more sophisticated masking strategies for SDS or other inpainting methods.
    *   Implement RePaint or other diffusion-based inpainting algorithms that might offer better coherence for large missing regions.
*   **Broader Dataset Support:** Adapt the model and pipeline to work with more complex datasets beyond MNIST (e.g., CelebA, CIFAR-10) to tackle more challenging generation tasks. This would likely require architectural adjustments to the UNet and more extensive training.
*   **Hyperparameter Optimization:** Conduct a more systematic hyperparameter search for the diffusion model, CFG guidance scale, SDS optimization (learning rate, number of steps), and training parameters using tools like Weights & Biases Sweeps.
*   **Quantitative Evaluation of Inpainting:** Develop quantitative metrics to evaluate the quality of inpainting results beyond visual inspection.
*   **Interactive Demo/UI:** Build a simple Gradio or Streamlit interface to allow users to interactively upload images, draw masks, and perform inpainting or sampling with different parameters.
*   **Further Model Exploration:**
    *   Experiment with different UNet architectures or attention mechanisms.
    *   Try latent diffusion models (LDMs) for more efficient training and sampling on higher-resolution images.
*   **Test Coverage:** Increase unit and integration test coverage to ensure robustness and facilitate easier refactoring.
*   **Documentation Enhancement:** Add even more detailed docstrings, comments, and potentially Sphinx-generated API documentation.

This project provided a valuable hands-on experience with diffusion models, from fundamental implementation to advanced applications like CFG and SDS inpainting, along with practical insights into debugging and managing a machine learning codebase.
