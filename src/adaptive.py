import numpy as np
from scipy.ndimage import generic_filter
from numba import njit

def get_adaptive_indices(image):

    # measure local variance for each pixel
    @njit
    def local_variance(window):
        return np.std(window - window[window.size // 2])

    # get image brightness
    brightness = np.linalg.norm(image, axis=2)

    kernel_size = (5, 5)
    variance_map = generic_filter(brightness, local_variance, size=kernel_size)

    # now select the pixels to sample by weighted roll
    # prefix sum the variance map
    prefix_sum = np.cumsum(variance_map.flatten())
    variance_sum = variance_map.sum()

    rolls = np.random.rand(variance_map.size) * variance_sum
    adaptive_indices = np.searchsorted(prefix_sum, rolls).astype(np.uint32)

    bins = np.bincount(adaptive_indices, minlength=image.shape[0] * image.shape[1])
    summed_bins = np.insert(np.cumsum(bins), 0, 0).astype(np.uint32)
    try:
        assert summed_bins.size == image.shape[0] * image.shape[1] + 1
    except AssertionError:
        print(summed_bins.size, image.shape[0] * image.shape[1] + 1)
        print(summed_bins)
        print(image.shape)
        raise
    sorted_indices = adaptive_indices[np.argsort(adaptive_indices)].astype(np.uint32)

    return bins, summed_bins, sorted_indices
