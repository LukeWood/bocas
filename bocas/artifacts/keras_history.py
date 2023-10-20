from bocas.artifacts.artifact import Artifact
from bocas.yaml_utils import parse_yaml_node
from tensorflow.keras.callbacks import History


class KerasHistory(Artifact):
    """KerasHistory is used to return the keras `model.fit()` history."""

    yaml_tag = "!KerasHistory"

    def __init__(self, history, **kwargs):
        super().__init__(**kwargs)
        if isinstance(history, History):
            self.history = history.history
        elif isinstance(history, dict):
            self.history = history
        else:
            raise ValueError(
                f"Expected `history` to be a `tensorflow.keras.callbacks.History` or a "
                f"dictionary, instead got type {type(history)}."
            )

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
