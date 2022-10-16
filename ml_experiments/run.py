from ml_experiments import Sweep
from ml_experiments import Result
import ml_collections
import itertools


class DummyModule:
    def run(self, config):
        return Result(name="testing-123", artifacts=[])


def _import_run_lib(path):
    # TODO(lukewood): some magic to import 'run' from `path`
    module = DummyModule()

    # TODO(lukewood): also inspect the arity of the function and raise value error
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
        if not isinstance(key, Sweep):
            static_keys[key] = config[key]
            continue

        dynamic_keys[key] = config[key]

    selections = []
    for key in dynamic_keys:
        selections.append([(key, val) for val in dynamic_keys[key]])

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
        # TODO(lukewood): Graceful error handling, allow specification of strategies
        # for error handling.
        result = module.run(config)
        results.append(result)

    return results
