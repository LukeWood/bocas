import yaml
import numpy as np
import ml_collections
from bocas.artifacts import Artifact, KerasHistory, Metrics


def contains_only_registered_tags(yaml_str):
    return not "!!python/object" in yaml_str


def numpy_scalar_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:float", str(data.item()))


def numpy_array_representer(dumper, data):
    if isinstance(data, np.ndarray):
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data.tolist(), flow_style=False
        )
    elif isinstance(data, np.core.multiarray.scalar):
        return numpy_scalar_representer(dumper, data)


def numpy_float32_representer(dumper, data):
    return dumper.represent_scalar("!np.float32", str(data))


def config_dict_representer(dumper, data):
    return dumper.represent_mapping("!ConfigDict", data.to_dict())


def config_dict_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return ml_collections.ConfigDict(fields)


def numpy_float32_constructor(loader, node):
    value = loader.construct_scalar(node)
    return np.array([value]).astype(np.float32)[0]


def base_representer(tag):
    def representer(dumper, data):
        return dumper.represent_mapping(tag, data.to_yaml())

    return representer


def base_constructor(cls):
    def constructor(loader, node):
        return cls.from_yaml(loader, node)

    return constructor


def configure_dumper():
    yaml.add_representer(np.float32, numpy_float32_representer)
    yaml.add_representer(np.ndarray, numpy_array_representer)
    yaml.add_representer(np.core.multiarray.scalar, numpy_scalar_representer)
    yaml.add_representer(ml_collections.ConfigDict, config_dict_representer)
    yaml.add_representer(Artifact, base_representer(Artifact.yaml_tag))
    yaml.add_representer(KerasHistory, base_representer(KerasHistory.yaml_tag))
    yaml.add_representer(Metrics, base_representer(Metrics.yaml_tag))


def configure_loader():
    yaml.add_constructor("!np.float32", numpy_float32_constructor)
    yaml.add_constructor("!ConfigDict", config_dict_constructor)
    yaml.add_constructor(Artifact.yaml_tag, base_constructor(Artifact))
    yaml.add_constructor(KerasHistory.yaml_tag, base_constructor(KerasHistory))
    yaml.add_constructor(Metrics.yaml_tag, base_constructor(Metrics))


def configure_custom_yaml():
    configure_dumper()
    configure_loader()
