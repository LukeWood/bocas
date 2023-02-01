from bocas.artifacts.artifact import Artifact


class Metrics(Artifact):
    """Metrics is used to report scalar metrics, typically from `model.evaluate()`."""

    def __init__(self, metrics, **kwargs):
        super().__init__(**kwargs)
        # TODO(lukewood): override get item to return metrics item
        self.metrics = metrics
