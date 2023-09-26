from bocas.artifacts.yaml_utils import parse_yaml_node


class Artifact:
    yaml_tag = "!Artifact"

    def __init__(self, name=None):
        self.name = name

    def to_yaml(self):
        return {"name": self.name}

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(**parse_yaml_node(loader, node))
