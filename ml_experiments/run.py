from ml_experiments import Sweep
from ml_experiments import Result
import ml_collections
import itertools
import importlib

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


def run(path, config):
    config_values = config.to_dict()
    # handle dynamic import
    module = _import_run_lib(path)

    results = []
    for config in _iter_configs(config_values):
        print(config)
        # TODO(lukewood): Graceful error handling, allow specification of strategies
        # for error handling.
        result = module.run(config)
        results.append(result)

    return results
