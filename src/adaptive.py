import numpy as np
from scipy.ndimage import generic_filter

def get_adaptive_indices(image):

    # measure local variance for each pixel
    def local_variance(window):
        return np.var(window)

    kernel_size = 5
    variance_map = generic_filter(image, local_variance, size=kernel_size)

    # now select the pixels to sample by weighted roll
    # prefix sum the variance map
    prefix_sum = np.cumsum(variance_map.flatten())
    variance_sum = variance_map.sum()

    rolls = np.random.rand(variance_map.size) * variance_sum
    adaptive_indices = np.searchsorted(prefix_sum, rolls).astype(np.uint32)

    return adaptive_indices
