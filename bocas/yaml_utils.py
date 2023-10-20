import yaml


def parse_yaml_node(loader, node):
    if isinstance(node, yaml.MappingNode):
        # Recursively process key-value pairs
        data = {}
        for key_node, value_node in node.value:
            key = parse_yaml_node(loader, key_node)
            value = parse_yaml_node(loader, value_node)
            data[key] = value
        return data
    elif isinstance(node, yaml.SequenceNode):
        # Recursively process elements
        return [parse_yaml_node(loader, item) for item in node.value]
    elif isinstance(node, yaml.ScalarNode):
        return loader.construct_object(node, deep=True)
    else:
        # Handle other node types as needed
        raise ValueError(f"Unexpected node type {type(node)}")
