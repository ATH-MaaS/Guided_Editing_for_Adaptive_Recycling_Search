"""Minimal local EasyDict fallback used when the package is unavailable."""


class EasyDict(dict):
    """Dictionary with attribute access.

    This covers the config usage in Wan scripts without requiring the external
    ``easydict`` package in isolated Python environments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def update(self, *args, **kwargs):
        other = dict(*args, **kwargs)
        for key, value in other.items():
            if isinstance(value, dict) and not isinstance(value, EasyDict):
                value = EasyDict(value)
            super().__setitem__(key, value)
        return None
