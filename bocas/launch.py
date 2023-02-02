import sys

import ml_collections
from absl import flags
from ml_collections import config_flags

import bocas


def launch():
    FLAGS = flags.FLAGS

    flags.DEFINE_string("task", None, "the path to `run.py`.")
    config_flags.DEFINE_config_file("config")

    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("config")

    FLAGS(sys.argv)
    bocas.run(FLAGS.task, FLAGS.config)


if __name__ == "__main__":
    launch()
