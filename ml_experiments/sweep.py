class Sweep:
    """Sweep allows you to define a sweep over a specific configuration value.

    By default, the product of all sweeps will be run at experiment time.
    """

    def __init__(self, items):
        self.items = items
