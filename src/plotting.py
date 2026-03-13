import matplotlib.pyplot as plt


def plot_results(y_test, y_pred, name, save_path=None):
    """
    Generates plots for model evaluation:
    1. Predicted vs. Actual scatter plot.
    2. Residual plot (Error vs. Actual).

    Args:
        y_test: True fatigue strength values.
        y_pred: Model predicted fatigue strength values.
        name: Name of the model/dataset for titles.
        save_path: Path to save the resulting figure.
    """
    residuals = y_pred - y_test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot predicted vs. actual
    ax1.scatter(y_test, y_pred, alpha=0.6, color='royalblue', edgecolors='k', s=40)

    # Draw ideal line
    min_val, max_val = y_test.min(), y_test.max()
    ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label="Ideal")

    ax1.set_xlabel("Actual Fatigue Strength (MPa)")
    ax1.set_ylabel("Predicted Fatigue Strength (MPa)")
    ax1.set_title(f"{name}: Predicted vs. Actual")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.scatter(y_test, residuals, alpha=0.6, color='indianred', edgecolors='k', s=40)
    ax2.axhline(0, color='black', linestyle='-', lw=1.5)

    ax2.set_xlabel("Actual Fatigue Strength (MPa)")
    ax2.set_ylabel("Prediction Error (MPa)")
    ax2.set_title(f"{name}: Residual Analysis")
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


    plt.show()
