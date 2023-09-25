import importlib
import itertools
import os
import pickle
import yaml
from termcolor import cprint

import ml_collections

from bocas.sweep import Sweep


def _import_run_lib(path):
    loader = importlib.machinery.SourceFileLoader("_run_task", path)
    module = loader.load_module()

    if not hasattr(module, "run"):
        raise ValueError(
            "Expected the module specified in `path` to contain a "
            "`run()` method that accepts a `ml-collections.ConfigDict()` object "
            "as its first positional argument"
        )
    return module


def _iter_configs(config):
    static_keys = {}
    dynamic_keys = {}

    for key in config:
        if not isinstance(config[key], Sweep):
            static_keys[key] = config[key]
            continue

        dynamic_keys[key] = config[key]

    selections = []
    for key in dynamic_keys:
        selections.append([(key, val) for val in dynamic_keys[key].items])

    for selection in itertools.product(*selections):
        result = static_keys.copy()
        for key, val in selection:
            result[key] = val
        yield ml_collections.ConfigDict(initial_dictionary=result)


def run(path, config, artifact_dir="artifacts"):
    config_values = config.to_dict()
    # handle dynamic import
    if isinstance(path, str):
        module = _import_run_lib(path)
        run = module.run
    else:
        run = path

    os.makedirs(artifact_dir, exist_ok=True)

    results = []

    for config in _iter_configs(config_values):
        # TODO(lukewood): Graceful error handling, allow specification of strategies
        # for error handling.
        result = run(config)
        if result is None:
            raise ValueError(
                "`result` returned from `run()` was `None`. "
                "Did you forget a return statement?"
            )
        if result.config is None:
            result.config = config

        result_dir = f"{artifact_dir}/{result.name}"
        os.makedirs(result_dir, exist_ok=True)

        try:
            # Dump results to yaml
            serialized_result = yaml.dump(result, default_flow_style=False)
        except Exception as e:
            cprint(f"YAML serialization failed with error: {e}.", "red")
            serialized_result = None

        if serialized_result:
            with open(f"{result_dir}/results.yaml", "w") as f:
                f.write(serialized_result)
        # If dumping to yaml fails, default to pickling
        else:
            cprint("Defaulting to saving result as pickle.", "red")
            with open(f"{result_dir}/results.p", "wb") as f:
                pickle.dump(result, f)

        results.append(result)

    return results
