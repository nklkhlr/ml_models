from .logging import Logger
import numpy as np
import matplotlib.pyplot as plt


def plot_progression(logs: Logger, metric_type: str = "Loss",
                     ax: plt.axis = None, evaluation_metric: str = None,
                     **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    epochs = np.arange(logs.test_metric.size) + 1
    if metric_type == "Loss":
        error = [std_loss for _, std_loss in logs.epoch_loss]
        train_y = [mean_loss for mean_loss, _ in logs.epoch_loss]
        test_y = logs.test_loss
    elif metric_type == "Evaluation":
        if evaluation_metric is None:
            raise ValueError("'evaluation_metric must be set, when plotting Evaluation")
        error = [std_eval for _, std_eval in logs.train_metric]
        train_y = [mean_eval for mean_eval, _ in logs.train_metric]
        test_y = logs.test_metric
    else:
        raise ValueError(f"{metric_type} is invalid for 'metric_type', please use 'Loss' or 'Evaluation'")
    ax.errorbar(
        epochs, train_y, yerr=error, c="tab:blue",
        label=f"Training {metric_type}", capsize=kwargs.pop('capsize', .01),
        **kwargs
    )
    ax.scatter(
        epochs, train_y, c="tab:blue", label=f"Training {metric_type}"
    )
    ax.plot(
        epochs, test_y, c="tab:orange", label=f"Test {metric_type}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_type)
    ax.set_xticks(epochs)
    ax.legend()


def plot_curves(logs: Logger, evaluation_metric: str, **kwargs):
    fig, ax = plt.subplots(figsize=(16, 9), ncols=2)
    plot_progression(logs, ax=ax[0], **kwargs)
    ax[0].set_title("Loss progression")
    plot_progression(
        logs, evaluation_metric=evaluation_metric,
        ax=ax[1], **kwargs
    )
    ax[1].set_title(f"{evaluation_metric} progression")
    plt.show()
