from ..squeeze_option import SqueezeOption
from typing import List
import numpy as np


class Cluster:
    """
    one dim cluster, give a 1d-array, return each clusters indices
    """

    def __init__(self, option: SqueezeOption):
        self.option = option

    def __call__(self, array) -> List[np.ndarray]:
        raise NotImplementedError()

