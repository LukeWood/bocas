from ml_experiments import Sweep
from ml_experiments import Result


def _import_run_lib(path):
    # TODO(lukewood): some magic to import 'run' from `path`
    module = {"run": lambda config: config}

    # TODO(lukewood): also inspect the arity of the function and raise value error
    if not hasattr(module):
        raise ValueError(
            "Expected the module specified in `path` to contain a "
            "`run()` method that accepts a `ml-collections.ConfigDict()` object "
            "as its first positional argument"
        )


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
