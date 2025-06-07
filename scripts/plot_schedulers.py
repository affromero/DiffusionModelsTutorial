"""Script to visualize and compare different diffusion schedulers.

This script generates plots for key properties of various diffusion schedulers
to help understand their behavior. The following properties are typically plotted:

| Property                        | Symbol      | Description                                                                                                | Interpretation on Plot                                                                                                |
| :------------------------------ | :---------- | :--------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| Betas                           | β<sub>t</sub>   | Variance of noise added at each specific timestep `t`.                                                     | Higher Y-values mean more noise is added at that timestep. Shows the "aggressiveness" of noise injection over time. |
| Alphas Cumulative Product       | α̅<sub>t</sub>   | Cumulative product of (1 - β<sub>i</sub>) up to timestep `t`. Represents how much original signal is retained. | Higher Y-values mean more original signal is preserved. Shows how quickly the original image information degrades.    |
| Sqrt(1 - Alphas Cumprod)      | √(1-α̅<sub>t</sub>) | Standard deviation of total noise added to `x_0` to get `x_t`. Quantifies overall "noisiness".             | Higher Y-values mean the image is more noisy. Shows the rate of overall noise accumulation.                         |


**Mathematical Context of Plotted Properties:**

The forward diffusion process gradually adds noise to an image `x_0` over `T` timesteps.
The state `x_t` at any timestep `t` can be expressed directly from the original image `x_0`
and a standard Gaussian noise sample `ε ~ N(0, I)` using the following key formula:

`x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε`

Here's how the plotted properties relate to this formula:

1.  **`alphas_cumprod` (α̅<sub>t</sub>)**:
    *   **In the formula:** `x_t = √( **α̅<sub>t</sub>** ) * x_0 + √(1 - **α̅<sub>t</sub>**) * ε`
    *   **Definition:** `α̅_t = Π_{i=1 to t} (1 - β_i)`, where `β_i` is the noise variance at step `i`.
    *   **Effect:** The term `√(α̅_t)` scales the original image `x_0`. As `t` increases, `α̅_t` (and thus `√(α̅_t)`) decreases from ~1 to ~0.
    *   **Plot Interpretation:** The plot of `alphas_cumprod` shows this decay. Higher values mean more of the original signal `x_0` is preserved in `x_t`.

2.  **`sqrt_one_minus_alphas_cumprod` (√(1 - α̅<sub>t</sub>))**:
    *   **In the formula:** `x_t = √(α̅_t) * x_0 + **√(1 - α̅<sub>t</sub>)** * ε`
    *   **Effect:** This term scales the noise `ε`. It represents the standard deviation of the total effective noise added to `x_0` to get `x_t`. As `t` increases, `√(1 - α̅<sub>t</sub>)` increases from ~0 to ~1.
    *   **Plot Interpretation:** The plot shows this growth in noise magnitude. Higher values mean `x_t` is more dominated by noise.

3.  **`betas` (β<sub>t</sub>)**:
    *   **Relationship:** `β_t` is not directly in the closed-form `x_t` equation above but is fundamental as it defines `α̅_t` (since `α_i = 1 - β_i`).
    *   **Step-wise formula:** `x_t = √(1-β_t) * x_{t-1} + √β_t * ε_t` (where `ε_t` is new noise at step `t`).
    *   **Effect:** The schedule of `β_t` values dictates the per-step noise addition and thus shapes the overall trajectory of `α̅_t`.
    *   **Plot Interpretation:** The plot of `betas` shows the per-step noise variance. It drives the changes seen in the other two plots.

Understanding these relationships helps interpret how different schedulers control the noising process by varying `β_t`, which in turn affects `α̅_t` and the balance between signal and noise in `x_t` over time.


**Scheduler Comparisons and Plot Expectations:**

*   **DDPMScheduler (Linear):**
    *   **Plot Expectation:**
        *   `betas`: Increases linearly from `beta_start` to `beta_end`.
        *   `alphas_cumprod`: Decreases steadily, forming a somewhat convex curve.
        *   `sqrt_one_minus_alphas_cumprod`: Increases steadily, forming a somewhat concave curve.
    *   **Pros:** Simple, foundational diffusion schedule.
    *   **Cons:** Sampling can be slow (requires many steps, e.g., 1000). Quality might be surpassed by newer schedulers or variants.

*   **DDPMScheduler (Cosine):**
    *   **Plot Expectation:**
        *   `betas`: Starts very small, increases slowly, then more rapidly towards the end of timesteps.
        *   `alphas_cumprod`: Stays close to 1 for longer initially, then drops more sharply compared to linear. This means signal is preserved better in early stages.
        *   `sqrt_one_minus_alphas_cumprod`: Rises slowly at first, then more quickly.
    *   **Pros:** Often yields better sample quality than linear DDPM. Slower initial noise addition can be beneficial.
    *   **Cons:** Still typically requires a large number of inference steps.

*   **DDIMScheduler (Linear & Scaled Linear):**
    *   *(Note: The underlying `betas`, `alphas_cumprod` schedules are generated similarly to DDPM's linear or a scaled variant. The main difference in DDIM is the sampling step, which is deterministic and allows for fewer steps.)*
    *   **Plot Expectation (for `betas`, `alphas_cumprod`):**
        *   Linear: Will look very similar to "DDPMScheduler (Linear)".
        *   Scaled Linear: The `betas` will also increase linearly but might have a different range or slope, affecting the `alphas_cumprod` curve accordingly. This is common in models like Stable Diffusion.
    *   **Pros:**
        *   **Deterministic Sampling:** Given the same initial noise and parameters, DDIM produces the same output.
        *   **Fast Sampling:** Can generate high-quality samples in significantly fewer steps (e.g., 20-100) compared to DDPM.
        *   The `eta` parameter allows interpolation between deterministic DDIM (`eta=0`) and stochastic DDPM-like behavior (`eta=1`).
    *   **Cons:** If too few steps are used, quality can degrade or artifacts might appear. The "ideal" schedule might still be data-dependent.

*   **PNDMScheduler (Linear):**
    *   *(Note: PNDM also typically uses a standard beta schedule like linear DDPM for its `__post_init__`. Its strength lies in its more sophisticated solver during the sampling (reverse) process.)*
    *   **Plot Expectation (for `betas`, `alphas_cumprod`):** Will generally look similar to "DDPMScheduler (Linear)" as it initializes `betas` linearly by default.
    *   **Pros:**
        *   **Fast Sampling:** A pseudo-numerical method that can achieve good results in few steps (e.g., 50).
        *   Often performs well for tasks like inpainting.
        *   Uses a higher-order solver which can lead to better accuracy per step.
    *   **Cons:** The solver logic is more complex than DDPM or DDIM.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Adjust import path if necessary, assuming 'src' is in PYTHONPATH or script is run from root
from src.diffusion.scheduler import DDIMScheduler, DDPMScheduler, PNDMScheduler

# Ensure the plots directory exists
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def plot_scheduler_properties(
    schedulers: dict[str, DDPMScheduler | DDIMScheduler | PNDMScheduler],
    properties_to_plot: dict[str, str],
    num_timesteps: int,
) -> None:
    """Generate and saves plots for specified scheduler properties.

    Args:
        schedulers (dict): A dictionary of scheduler_name: scheduler_instance.
        properties_to_plot (dict): A dictionary of property_name: y_axis_label.
        num_timesteps (int): The number of timesteps used for the schedulers.

    """
    timesteps_range = torch.arange(num_timesteps)

    for prop_name, y_label in properties_to_plot.items():
        plt.figure(figsize=(12, 8))
        for name, scheduler in schedulers.items():
            if hasattr(scheduler, prop_name):
                values = getattr(scheduler, prop_name).cpu().numpy()
                # Ensure the property has the correct length
                if len(values) == num_timesteps:
                    plt.plot(timesteps_range, values, label=name)
                else:
                    print(
                        f"Warning: Property '{prop_name}' for scheduler '{name}' has length {len(values)}, expected {num_timesteps}. Skipping plot."
                    )
            else:
                print(
                    f"Warning: Scheduler '{name}' does not have property '{prop_name}'. Skipping plot."
                )

        plt.xlabel("Timestep")
        plt.ylabel(y_label)
        plt.title(f"{y_label} vs. Timestep for Different Schedulers")
        plt.legend()
        plt.grid(visible=True)
        plt.savefig(PLOTS_DIR / f"{prop_name}_comparison.png")
        plt.close()
        print(f"Saved plot: {prop_name}_comparison.png")


def main() -> None:
    """Run the main function to create and plot schedulers."""
    num_timesteps = 1000
    device = "cpu"  # Use CPU for plotting, no CUDA needed

    schedulers: dict[str, DDPMScheduler | DDIMScheduler | PNDMScheduler] = {
        "DDPM (Linear)": DDPMScheduler(
            num_timesteps=num_timesteps, beta_schedule="linear", device=device
        ),
        "DDPM (Cosine)": DDPMScheduler(
            num_timesteps=num_timesteps, beta_schedule="cosine", device=device
        ),
        "DDIM (Linear)": DDIMScheduler(
            num_timesteps=num_timesteps, beta_schedule="linear", device=device
        ),
        "DDIM (Scaled Linear)": DDIMScheduler(
            num_timesteps=num_timesteps, beta_schedule="scaled_linear", device=device
        ),
        "PNDM (Linear)": PNDMScheduler(
            num_timesteps=num_timesteps,
            device=device,  # PNDM defaults to linear
        ),
    }

    properties_to_plot = {
        "betas": "Beta Values",
        "alphas_cumprod": "Alpha Cumulative Product (α̅_t)",
        "sqrt_one_minus_alphas_cumprod": "Noise Level (√(1-α̅_t))",
    }

    plot_scheduler_properties(schedulers, properties_to_plot, num_timesteps)

    print(f"All plots saved in '{PLOTS_DIR}' directory.")


if __name__ == "__main__":
    main()
