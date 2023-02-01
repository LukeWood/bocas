import ml_experiments
import luketils
import pandas as pd

results = ml_experiments.Result.load_collection("artifacts/")

metrics_to_plot = {}

for experiment in results:
    metrics = experiment.get_artifact("fit_history").metrics

    metrics_to_plot[f"{experiment.name} Train"] = metrics["accuracy"]
    metrics_to_plot[f"{experiment.name} Validation"] = metrics["val_accuracy"]

luketils.visualization.line_plot(
    metrics_to_plot,
    path=f"results/combined-accuracy.png",
    title="Model Accuracy",
)
