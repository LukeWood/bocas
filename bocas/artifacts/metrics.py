from bocas.artifacts.artifact import Artifact
from .yaml_utils import parse_yaml_node


class Metrics(Artifact):
    """Metrics is used to report scalar metrics, typically from `model.evaluate()`."""

    yaml_tag = "!Metrics"

    def __init__(self, metrics, **kwargs):
        super().__init__(**kwargs)
        # TODO(lukewood): override get item to return metrics item
        self.metrics = metrics

    def to_yaml(self):
        config = super().to_yaml()
        config.update({"metrics": self.metrics})
        return config

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(**parse_yaml_node(loader, node))
