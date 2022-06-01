class SqueezeOption:
    def __init__(self, **kwargs):
        self.debug = False

        # Filter
        self.enable_filter = True

        # Density Estimation
        self.cluster_method = "density"
        self.density_estimation_method = 'histogram'

        # KDE
        self.density_smooth_conv_kernel = [1.]
        self.kde_bw_method = None
        self.kde_weights = None

        # Histogram
        self.histogram_bar_width = "auto"

        # relative max
        self.max_allowed_deviation_bias = 0.10
        self.max_allowed_deviation_std = 0.01

        # Cluster
        self.cluster_smooth_window_size = "auto"
        self.max_normal_deviation = 0.20

        # Group
        # self.least_score = 2.0
        self.least_descent_score = 0.6
        self.normal_deviation_std = 0.1
        self.score_weight = "auto"
        self.max_num_elements_single_cluster = 12
        self.ps_upper_bound = 0.90

        self.__dict__.update(kwargs)
