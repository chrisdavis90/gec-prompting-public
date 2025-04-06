from dataclasses import dataclass
from itertools import product
from typing import List

"""
Containers for model text generation parameters
"""


class BaseParams:
    def __init__(self, *args, **kwargs):
        # parse args and create lists from comma-separated strings
        # for key, value in kwargs.items():
        #     if isinstance(value, str):
        #         kwargs[key] = value.split(",")
        #         # check if string list should be an int list
        #         if all([v.isnumeric() for v in kwargs[key]]):
        #             kwargs[key] = [int(v) for v in kwargs[key]]

        # save kwargs as attributes
        for key, value in kwargs["gen_kwargs"].items():
            # HACK to skip name
            # TODO: fix this
            if key == "name":
                continue
            # check if key exists as attribute
            # if hasattr(self, key):
            # if it does, set the attribute
            setattr(self, key, value)

    def update_max_len(self, len: int):
        raise NotImplementedError


class TestParams(BaseParams):
    def __init__(self, *args, **kwargs):
        # parse args and create lists from comma-separated strings
        for key, value in kwargs.items():
            if isinstance(value, str):
                kwargs[key] = value.split(",")

                # check if string list should be an int list
                if all([v.isnumeric() for v in kwargs[key]]):
                    kwargs[key] = [int(v) for v in kwargs[key]]

        # save kwargs as attributes
        for key, value in kwargs.items():
            # check if key exists as attribute
            # if hasattr(self, key):
            # if it does, set the attribute
            setattr(self, key, value)

        self.reset_cross_product()

    one: List[str] = None
    two: List[str] = None
    three: List[int] = None

    def reset_cross_product(self):
        # save the cross product of all attributes as a list of dicts
        kwargs = self.__dict__
        kwargs.pop("_cross_product", None)

        keys = kwargs.keys()
        _cross_product = []

        for values in product(*kwargs.values()):
            _cross_product.append(dict(zip(keys, values)))

        self._cross_product = _cross_product

        # self._cross_product = list(product(*[v for v in kwargs.values()]))

    # self._cross_product = list(product(*[v for v in kwargs.values()]))

    def update_max_len(self, len: int):
        pass

    def to_name(self):
        args = ["test"]
        args = sorted(args)
        return "_".join(args)

    def __iter__(self):
        return self

    def __next__(self):
        # return the next cross product until there are no more
        if len(self._cross_product) == 0:
            raise StopIteration

        return self._cross_product.pop(0)


class HFParams(BaseParams):
    def update_max_len(self, len: int):
        self.max_length = len


class OpenAIParams(BaseParams):
    def update_max_len(self, len: int):
        self.max_tokens = len


class CohereParams(BaseParams):
    def update_max_len(self, len: int):
        self.max_tokens = len
