from typing import Any


MAX = 'max'
MIN = 'min'


class RunningExtrema:
    """Keeps a running extremum (max/min) of a set of named values.

    Attributes:
        extremum: The extremum (max/min) to use.
        extrema_dict: A dictionary with the running extremum per key. The key
            is prepended with 'Max' or 'Min', depending on the choses extremum.
    """
    def __init__(self, extremum: str):
        """
        Args:
            extremum: The extremum (max/min) to use.
        """
        if extremum not in [MAX, MIN]:
            raise ValueError(
                f'Unknown extremum "{extremum}". '
                f'Possible  values: "{MAX}" and "{MIN}".'
            )
        self.extrema_dict = {}
        self.extremum = extremum

    def is_new_extremum(self, key: str, val: Any):
        """Returns if the value is a new extremum for the key.

        Args:
            key: The key.
            val: The value.
        """
        if key not in self.extrema_dict:
            try:
                # Check if the value can be compared and returns a (single)
                # boolean value
                if self._comp_fn(val, 0) or True:
                    return True
            except (TypeError, RuntimeError):
                return False

        return self._comp_fn(val, self.extrema_dict[key])

    def _comp_fn(self, new, curr):
        return (
            new >= curr if self.extremum == MAX
            else new <= curr
        )

    def update(self, key: str, val: Any):
        """
        Replaces the running value of the given key if the value is a new
        extremum.
        """
        if self.is_new_extremum(key, val):
            self.extrema_dict[key] = val

    def update_dict(self, d: dict):
        """
        Replaces the running value of those keys in the given dict that have
        a new extremum as value.
        """
        for k, v in d.items():
            self.update(k, v)

    def clear(self):
        """Clears all running values."""
        self.extrema_dict = {}
