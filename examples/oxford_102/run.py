"""
Name: Quickstart
... other metadata


With `ml-experiments` it is easy to write parameterized experiments, collect metrics
from various parameterizations, and aggregate those metrics to produce plots.
The goal of `ml-experiments` is to allow you to fully automate the process of running
an experiment, producing an artifact such as a figure or chart, and including that 
artifact in your research paper.

This tutorial will show you how to perform a sweep over two model architectures and
two data augmentation schemes.  The models are trained on the Oxford 102 flowers
classification dataset.  By the end of this guide, you will be able to modify this
script, re-run the pipeline, and automatically re-render the `paper/` directory to
include the updated tables and figures.
"""
import ml_experiments
import tensorflow_datasets as tfds
from tensorflow import keras
from absl import flags
import keras_cv
import sys
import luketils


"""
First, we define some functions to parse the config that `ml-experiments` will pass to
our experiment run.  This is all just python code, but its important to use some sort
of unique identifier to distinguish your models.  This allows TensorBoard to function
properly.

I like to use information about the model in the name, so in this case the model name 
is:

```python
fmodel={config.model_type}-augmenter={config.augmenter_type}"
```
"""


def get_name(config):
    return f"model={config.model_type}-augmenter={config.augmenter_type}"


"""
Next, we define a function to fetch a model constructor based on a config avlue:
"""


def get_model(model_type):
    if model_type == "resnet50":
        return keras_cv.models.ResNet50V2
    if model_type == "efficientnet":
        return keras_cv.models.EfficientNetV2Large
    raise ValueError(f"Unsupported model_type={model_type}")


"""
and define some data augmentation pipelines to sweep over:
"""


def get_augmenter(augmenter_type):
    if augmenter_type == "eval":
        return [
            keras_cv.layers.Resizing(width=224, height=224, crop_to_aspect_ratio=True)
        ]
    if augmenter_type == "basic":
        return [
            keras_cv.layers.RandomFlip(),
        ]
    if augmenter_type == "optimized":
        return [
            keras_cv.layers.RandomCropAndResize(
                area_factor=(0.8, 1.0),
                aspect_ratio_factor=(3 / 4, 4 / 3),
                target_size=(224, 224),
            ),
            keras_cv.layers.RandomFlip(),
            keras_cv.layers.RandAugment(value_range=(0, 255)),
            kereas_cv.layers.CutMix(),
            keras_cv.layers.MixUp(),
        ]

    raise ValueError(f"Unsupported augmenter_type={augmenter_type}")


"""
After a run, you may wish to plot out a plot for that specific model.  This can be done
by passing an `artifacts_dir` from your root config:
"""


def plot(metrics, name, artifacts_dir):
    metrics = {"Train": metrics["accuracy"], "Validation": metrics["val_accuracy"]}
    luketils.visualization.line_plot(
        data=metrics, path=f"{artifacts_dir}/{name}-accuracy.png"
    )


"""
Finally, we assemble our data preprocessing pipeline:
"""


def prepare_dataset(ds, augmentation):
    augmentor = get_augmentor(augmentation)

    def preproc(x, y):
        inputs = {"images": x, "labels": y}
        for layer in augmentor:
            inputs = layer(inputs)
        return inputs["images"], inputs["labels"]

    ds = ds.batch(64)
    ds = ds.map(preproc, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


"""
And define our entrypoint.  In `ml-experiments`, the entrypoint to a run is called 
`run()`.
"""


def run(config):
    config = ml_experiments.get_config()
    name = get_name(config)

    model = get_model(config.model_type)(
        include_rescaling=True, include_top=True, classes=102, weights=None, name=name
    )

    model.compile(loss="mse", optimizer="adam")

    callbacks = [keras.callbacks.TensorBoard(logdir=config.log_dir)]

    train_ds, test_ds = tfds.load(
        "oxford102", as_supervised=True, split=["train", "test"]
    )
    train_ds = prepare_dataset(train_ds, augmentation=config.augmenter_type)
    test_ds = prepare_dataset(test_ds, augmentation="eval")

    history = model.fit(train_ds, epochs=10, callbacks=callbacks)
    metrics = model.evaluate(test_ds, return_dict=True)

    train_metrics = history.history
    plot(train_metrics, name, config.artifacts_dir)

    return ml_experiments.Result(
        name=name,
        artifacts=[
            ml_experiments.artifacts.KerasHistory(history, name="fit_history"),
            ml_experiments.artifacts.Metrics(metrics, name="eval_metrics"),
        ],
    )


"""
`ml-experiments` supports standalone runs as well.  Simply write the following code:

```python
if __name__ == "__main__":
    ml_experiments.run(main)
```

and whatever path is passed to `--config` from the command line will be parsed by the
`ml-collections` library as the config in `run()`.
"""
