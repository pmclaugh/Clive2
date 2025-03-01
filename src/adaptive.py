import numpy as np
from scipy.ndimage import generic_filter
from numba import njit
import cv2

random_generator = np.random.default_rng()

def get_adaptive_indices(image):

    # measure local variance for each pixel
    @njit
    def local_variance(window):
        return np.var(window)

    # convert to grayscale
    brightness = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    kernel_size = (5, 5)
    variance_map = generic_filter(brightness, local_variance, size=kernel_size)
    cv2.imshow("variance", (variance_map / np.max(variance_map)))
    cv2.waitKey(1)

    # cap the variance so edges don't completely dominate
    variance_map = np.clip(variance_map, 0, np.median(variance_map) * 2)

    variance_map = variance_map.flatten()
    variance_sum = np.sum(variance_map)
    adaptive_indices = random_generator.choice(np.arange(variance_map.size), size=variance_map.size, p=variance_map / variance_sum)

    bins = np.bincount(adaptive_indices, minlength=image.shape[0] * image.shape[1])
    summed_bins = np.insert(np.cumsum(bins), 0, 0).astype(np.uint32)
    sorted_indices = adaptive_indices[np.argsort(adaptive_indices)].astype(np.uint32)

    return bins, summed_bins, sorted_indices
