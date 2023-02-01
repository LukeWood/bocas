from ml_experiments.artifacts.artifact import Artifact
import pickle


class KerasHistory(Artifact):
    """KerasHistory is used to return the keras `model.fit()` history."""

    def __init__(self, history, **kwargs):
        super().__init__(**kwargs)
        self.history = history.history

    @property
    def metrics(self):
        """Returns a dictionary holding the metrics from `fit()`."""
        return self.history
