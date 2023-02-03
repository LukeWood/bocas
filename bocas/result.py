import glob
from termcolor import colored, cprint
import os
import pickle

from bocas.artifacts import Artifact


class Result:
    """Result contains the result of an experiment.

    Usage:
    ```python
    return bocas.Result(
        name='testing-123',
        artifacts=[
            bocas.artifacts.KerasHistory(history, name="fit_history"),
            bocas.artifacts.Metrics(metrics, name="eval_metrics"),
        ],
    )
    ```

    Args:
        name: string identifier for the experiment
        artifacts: (Optional) list of `bocas.artifacts.Artifact` to be included in the
            result.
        optional: (Optional) `ml_collections.ConfigDict` to be included in the
            result.
    """

    def __init__(self, name, artifacts=None, config=None):
        if not _all_artifacts(artifacts):
            raise ValueError(
                "Expected all of `artifacts` to be subclasses of "
                "`bocas.artifacts.Artifact`.  Instead, got "
                f"artifacts={artifacts}."
            )
        if not isinstance(name, str):
            raise ValueError(f"Expected `name` to be a string, instead got name={name}")
        self.name = name
        self.artifacts = artifacts or []
        self.config = config

    def get(self, name):
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        raise ValueError(
            f"Didn't find an artifact with name `name={name}`. "
            "Instead, found artifacts with the following names: "
            f"[{', '.join([a.name for a in self.artifacts])}]"
        )

    @staticmethod
    def load(path):
        with open(f"{path}/results.p", "rb") as f:
            result = pickle.load(f)
        return result

    @staticmethod
    def load_collection(path):
        results = []
        for path in glob.glob(f"{path}/*"):
            try:
                r = Result.load(path)
                results.append(r)
            except Exception as e:
                cprint(colored("Error loading result:", "red", attrs=["bold"]) + ' ' + path)
                print(e)
                pass
        return results

    def serialize_to(self, artifacts_dir):
        subdir = f"{artifacts_dir}/{self.name}"
        os.makedirs(subdir, exist_ok=True)

        # TODO(lukewood): pickle.load/dump/etc!
        for artifact in self.artifacts:
            artifact.serialize_to(subdir)


def _all_artifacts(artifacts):
    return all([isinstance(x, Artifact) for x in artifacts])
