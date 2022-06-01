from .cluster import *
from .density_cluster import *


def cluster_factory(option: SqueezeOption):
    method_map = {
        "density": DensityBased1dCluster,
    }
    return method_map[option.cluster_method](option)
