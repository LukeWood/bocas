# Bocas

![Downloads](https://img.shields.io/pypi/dm/bocas.svg)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/lukewood/bocas/issues)


`bocas` is an opinionated open source framework for organizing,
orchestrating, and ultimately publishing research experiments.

Some design highlights of `bocas` include:

-  the ability to cache artifacts between experiment runs
- the de-coupling of plot generation and training jobs
-  `bocas` augments the `ml-collections` library to allow you to describes an
array of experiments in a single config
- run all of the experiments with a single command
- gather artifacts from the experiments
- aggregate the results into plots, tables, and figures for use in your final report
- easily combine results from multiple experiments
- and more!

## Quick-Start

- [Basic: Oxford 102 flowers classification example](examples/oxford_102/)
- [Intermediate: Object detection benchmarks with KerasCV](https://github.com/LukeWood/OD-Benchmarks)
- [Overview](#Overview)

## Overview

Using `bocas` is easy!  
To get started, you need to be familiar with a few concepts.
This overview covers everything you need to know.

[To quickly jump right into things, check out the Oxford 102 flowers classification example.](examples/oxford_102/)

### Tasks & Tactics

In the mental model of `bocas` there exists *Tasks* and *Tactics*.  A *Task* is
something like: "classify images from MNIST", or "cluster samples into N classes", or
"perform generative learning in X style".  

A *Tactic* refers to the combination of all the details used to produce a
solution to a Task.  For example, one such Tactic for solving MNIST classification might
be to train a ResNet50V2 on data augmented with AugMix.  
Typically, to get a publishable result your paper will require you to have numerous
tactics to benchmark your novel tactic against.

Typically, a research work will have many Tasks: where
the overall goal of the paper is to benchmark a new Tactic's ability at solving a variety
of tasks.

`bocas` is structured around this idea: you will have at least one Task, and
each Task may be solved by numerous tactics.
As such, I recommend breaking your codebase down at the `Task` level, structuring your
paper's artifact with splits made on the `Task` level.  For example, a classification
paper might have the structure:

```
- tasks/
      - mnist/
            - ...
      - imagenet/
            - ...
```

### Code Structure

`bocas` provides an opinionated framework for generating

Keeping these concepts in mind, `bocas` recommends that you structure your code
into three levels:

- `library/` holds anything unique to your report/paper/publication.  This might include
  a new augmentation, a new `keras.Layer`, a new loss function, or a new metric.
- `tasks/` holds all of the tasks to benchmark your new technique on.
- `paper/` holds the `Latex` or `Markdown` code required to render your paper
- `paper/artifacts` subdirectory of `paper` that holds all of the artifacts produced by
  the `tasks`.  Typically when running a Task sweep you'll want to provide this directory
  to your scripts.

Your tasks should be structured as follows:

All code for a task should reside in `tasks/{task}/`, i.e. `tasks/oxford_102`.
You should create a `run.py` script.  This script must have a `run()` method that
accepts an `ml_collections.ConfigDict` as its first positional argument.  If you follow
the example in the [Oxford Flowers 102 example](examples/oxford_102/run.py), your
`run.py` file will support both independent run and mass-scale sweeps:

```python
def run(config):
    name = f'{config.optimizer}'
    train_ds, test_ds = tfds.load(
        "oxford102", as_supervised=True, split=["train", "test"]
    )
    model = keras_cv.models.ResNet50V2(
      include_rescaling=True,
      include_top=True,
      classes=102
    )
    model.compile(loss="mse", optimizer=config.optimizer)
    history = model.fit(train_ds, epochs=10)

    return bocas.Result(
        name=name,
        artifacts=[
            bocas.artifacts.KerasHistory(history, name="fit_history"),
        ],
    )
```

Once you are happy with the results from a single `run.py` run, create a `sweep.py`
config file.  In `sweep.py`, specify a `ml_collections.ConfigDict` containing
`bocas.Sweep` objects for any value you'd like to sweep oer.

```python
config = ml_collections.ConfigDict()

config.static_value = 'any-string-or-int-or-float-or-python-object'
config.optimizer = bocas.Sweep(['sgd', 'adam'])
```

Anytime a value of type `bocas.Sweep()` is encountered, the product of all
other defined `bocas.Sweep()` parameters is run with the addition of the new
values in that sweep.  

Be careful with this!  It is easy to create a lot of experiments:

```python
config = ml_collections.ConfigDict()
config.learning_rate = bocas.Sweep([x/100 for x in range(5, 21)])
config.optimizer = bocas.Sweep(['sgd', 'adam'])
config.model = bocas.Sweep(
  ['resnet50', 'resnet50v2', 'densenet101', 'efficientnet']
)
```

This configuration already contains `15 * 2 * 4` or `120` runs!  That is probably
way more than you'd like.  Try to define a few experiments that are all encompassing.
To accomplish this, run hyper parameter sweeps separately, and hardcode the values into
the final runs that are used to produce the charts.

After all of your runs are complete, create some charts and plots.  Save them to your
designated directory in your `paper/` directory so that they are rendered
into your updated paper.

I recommend writing a script to produce desired plots based on the artifacts that can
be run entirely separately from your experiments themselves.  Any example of this can
be found in the `oxford_102` example:

```python
# scripts/create_plots.py
results = bocas.Result.load_collection("artifacts/")

metrics_to_plot = {}

for experiment in results:
    metrics = experiment.get_artifact("fit_history").metrics

    metrics_to_plot[f"{experiment.name} Train"] = metrics["accuracy"]
    metrics_to_plot[f"{experiment.name} Validation"] = metrics["val_accuracy"]

luketils.visualization.line_plot(
    metrics_to_plot,
    path=f"{paper_dir}/results/combined-accuracy.png",
    title="Model Accuracy",
)
```

[Check out the full code in oxford_102.](examples/oxford_102/)

### Conclusions & Further Reading

Thats all it takes to get running with `bocas`.  Please check out the
[`examples/`](examples/) directory for more reading.  It contains a few more patterns
that might be useful in structuring your experiments.

## Limitations

:warning: right now `bocas` is under active development :warning:

While the API is relatively straightforward and simple, `bocas`
lacks support for multi-worker experiment runs.  This means that you will need to run
all of your experiments concurrently on a single machine.  If you are running 10-20
`fit()` loops to convergence, this will likely be an extremely expensive process.

Personally, I'd rather just wait for my experiments to run then fiddle with a ton of
infrastructure.  That being said, I mainly run small scale research.

If someone wants to contribute distributed runs, feel free!

## License

[Apache v2 License](LICENSE)

## Contributing

Contributions are more than welcome to `bocas`.  
Please see the GitHub issue tracker, and feel free to pick up any issue annotated
with [Contribution Welcome](https://github.com/lukewood/bocas/issues).

Additionally, bug reports are not only welcome but encouraged.  
Help me improve `bocas`!  
I made this project because I needed the tool.
I'm sure many others do as well.

## Thanks!

If you find this tool helpful, please toss a GitHub star on the repo and follow me on Twitter.

Thank you to all of our GitHub contributors:

<a href="https://github.com/lukewood/bocas/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=lukewood/bocas" />
</a>
