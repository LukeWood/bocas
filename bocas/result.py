import glob
from termcolor import colored, cprint
import os
import pickle
import yaml
from bocas.artifacts import Artifact

from .yamlify import (
    configure_custom_yaml,
)

configure_custom_yaml()


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
        # Maintain backwards compatibility with pickled results
        if os.path.exists(f"{path}results.p"):
            with open(f"{path}results.p", "rb") as f:
                result = pickle.load(f)
        elif os.path.exists(f"{path}results.yaml"):
            with open(f"{path}results.yaml", "r") as f:
                result = yaml.load(f, Loader=yaml.FullLoader)

        return result

    @staticmethod
    def load_collection(path):
        results = []
        for path in glob.glob(f"{path}/*/"):
            try:
                r = Result.load(path)
                results.append(r)
            except Exception as e:
                cprint(
                    colored("Error loading result:", "red", attrs=["bold"]) + " " + path
                )
                print(e)
                pass
        return results

    def to_yaml(self):
        yaml_dict = {
            "name": self.name,
            "artifacts": self.artifacts,
            "config": self.config,
        }
        return yaml_dict

    @classmethod
    def from_yaml(cls, loader, node):
        for key_node, value_node in node.value:
            if key_node.value == "config":
                config = loader.construct_object(value_node)
            elif key_node.value == "name":
                name = loader.construct_scalar(value_node)
            elif key_node.value == "artifacts":
                artifacts = [
                    loader.construct_object(art_node) for art_node in value_node.value
                ]
            else:
                raise ValueError(f"Key {key_node.value} not recognized")

        return cls(name, artifacts, config)


# Configure yaml for Result
def result_representer(dumper, data):
    return dumper.represent_mapping("!Result", data.to_yaml())


def result_constructor(loader, node):
    return Result.from_yaml(loader, node)


yaml.add_representer(Result, result_representer)
yaml.add_constructor("!Result", result_constructor)


def _all_artifacts(artifacts):
    return all([isinstance(x, Artifact) for x in artifacts])
