from bocas.artifacts.artifact import Artifact
from .yaml_utils import parse_yaml_node


class KerasHistory(Artifact):
    """KerasHistory is used to return the keras `model.fit()` history."""

    yaml_tag = "!KerasHistory"

    def __init__(self, history, **kwargs):
        super().__init__(**kwargs)
        try:
            self.history = history.history
        except:
            self.history = history

    @property
    def metrics(self):
        """Returns a dictionary holding the metrics from `fit()`."""
        return self.history

    def to_yaml(self):
        config = super().to_yaml()
        config.update({"history": self.history})
        return config

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(**parse_yaml_node(loader, node))
