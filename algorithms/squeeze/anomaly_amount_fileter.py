import numpy as np
from kneed import KneeLocator
from loguru import logger
from scipy.stats import gaussian_kde


class KPIFilter:
    def __init__(self, real_array, predict_array):
        # self.select_metrics = np.log(np.abs(real_array - predict_array) + 1) / 10
        self.select_metrics = np.abs(real_array - predict_array)
        # self.select_metrics = np.abs(predict_array - real_array) / np.abs(real_array + predict_array)
        kernel = gaussian_kde(self.select_metrics)
        _x = sorted(np.linspace(np.min(self.select_metrics), np.max(self.select_metrics), 1000))
        _y = np.cumsum(kernel(_x))
        knee = KneeLocator(_x, _y, curve='concave', direction='increasing').knee
        logger.info(f"kneed: {knee}")
        if knee is None:
            logger.warning("no knee point found")
            knee = np.min(self.select_metrics)
        self.filtered_indices = np.where(self.select_metrics > knee)

        self.original_indices = np.arange(len(real_array))[self.filtered_indices]

    def inverse_map(self, indices):
        return self.original_indices[indices]
