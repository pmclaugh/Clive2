import numpy as np
import numba
from constants import *
from primitives import unit, point, vec
import cv2
from collections import defaultdict
from struct_types import Ray
from struct_types import Camera as camera_struct

max_pixel_key = None
max_val = None
max_var = None
max_mean = None


class Camera:
    def __init__(self, center=point(0, 0, 0), direction=vec(1, 0, 0), phys_width=1.0, phys_height=1.0,
                 pixel_width=1280, pixel_height=720):
        self.center = center
        self.direction = direction
        self.phys_width = phys_width
        self.phys_height = phys_height
        self.focal_dist = (phys_width / 2.) / np.tan(H_FOV / 2.0)
        self.focal_point = self.center + self.focal_dist * direction
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.samples = 0
        self.sample_counts = np.ones((pixel_height, pixel_width), dtype=np.int64)
        self.variances = np.zeros_like(self.sample_counts, dtype=np.float64)
        self.var_means = np.ones_like(self.sample_counts, dtype=np.float64)
        self.var_m2 = np.zeros_like(self.sample_counts, dtype=np.float64)

        if abs(self.direction[0]) < FLOAT_TOLERANCE:
            self.dx = UNIT_X if direction[2] > 0 else UNIT_X * -1
        else:
            self.dx = unit(np.cross(direction * (UNIT_X + UNIT_Z), UNIT_Y * -1))

        if abs(self.direction[1]) < FLOAT_TOLERANCE:
            self.dy = UNIT_Y
        else:
            self.dy = unit(np.cross(direction, self.dx))

        self.dx_dp = self.dx * self.phys_width / self.pixel_width
        self.dy_dp = self.dy * self.phys_height / self.pixel_height
        self.pixel_phys_size = np.linalg.norm(self.dx_dp) * np.linalg.norm(self.dy_dp)
        self.origin = center - self.dx * phys_width / 2 - self.dy * phys_height / 2
        self.image = np.zeros((pixel_height, pixel_width, 3), dtype=np.float64)

    def ray_batch(self, pixels):
        indices = pixels[0] * self.pixel_width + pixels[1]
        offsets = np.random.rand(2, self.pixel_height, self.pixel_width)
        x_vectors = np.expand_dims(pixels[0] + offsets[0], axis=2) * self.dx_dp
        y_vectors = np.expand_dims(pixels[1] + offsets[1], axis=2) * self.dy_dp
        origins = self.origin + x_vectors + y_vectors
        directions = self.focal_point - origins
        directions = directions / np.linalg.norm(directions, axis=2)[:, :, np.newaxis]
        return origins, directions, indices, pixels

    def ray_batch_numpy(self, adaptive=False):
        batch = np.zeros((self.pixel_height, self.pixel_width), dtype=Ray)
        if adaptive:
            grid, weights = self.adaptive_grid
            origins, directions, indices, pixels = self.ray_batch(grid)
        else:
            grid = self.grid
            weights = np.ones(batch.shape) / (self.phys_width * self.phys_height)
            origins, directions, indices, pixels = self.ray_batch(grid)

        # weights = np.ones(batch.shape) / (self.phys_width * self.phys_height)

        batch['origin'][:, :, :3] = origins + 0.0001 * directions
        batch['direction'][:, :, :3] = directions
        batch['inv_direction'][:, :, :3] = 1.0 / directions
        batch['color'] = np.ones(4)
        batch['c_importance'] = weights
        batch['l_importance'] = 1.0  # set in kernel
        batch['tot_importance'] = weights
        batch['hit_light'] = -1
        batch['hit_camera'] = -1
        batch['material'] = -1
        batch['normal'][:, :, :3] = directions
        batch['from_camera'] = 1
        batch['triangle'] = -1
        batch['i'] = self.grid[0]
        batch['j'] = self.grid[1]
        return batch, grid

    def to_struct(self):
        c = np.zeros(1, dtype=camera_struct)
        c[0]['origin'][:3] = self.origin
        c[0]['focal_point'][:3] = self.focal_point
        c[0]['direction'][:3] = self.direction
        c[0]['dx'][:3] = self.dx
        c[0]['dy'][:3] = self.dy
        c[0]['pixel_width'] = self.pixel_width
        c[0]['pixel_height'] = self.pixel_height
        c[0]['phys_width'] = self.phys_width
        c[0]['phys_height'] = self.phys_height
        return c

    @property
    def adaptive_grid(self):
        variance_roller = np.cumsum(self.variances).flatten()
        if np.any(np.isnan(variance_roller)):
            print(np.sum(np.isnan(variance_roller)), "nans in variance roller")
        rolls = np.random.rand(self.pixel_height * self.pixel_width) * np.max(variance_roller)
        picks = np.searchsorted(variance_roller, rolls)
        pixel_map = np.zeros((2, self.pixel_height, self.pixel_width), dtype=np.int32)
        pixel_map[0] = np.reshape(picks % self.pixel_width, (self.pixel_height, self.pixel_width))
        pixel_map[1] = np.reshape(picks // self.pixel_width, (self.pixel_height, self.pixel_width))

        pick_counts = defaultdict(int)
        for pick in picks:
            pick_counts[pick] += 1

        global max_pixel_key, max_val, max_var, max_mean
        max_pixel_key = max(pick_counts, key=pick_counts.get)
        max_val = pick_counts[max_pixel_key]
        max_var = self.variances.flatten()[max_pixel_key]
        max_mean = self.var_means.flatten()[max_pixel_key]
        print(f"the most picked pixel was {max_pixel_key} with {max_val} samples. it has mean {max_mean} and variance {max_var}")

        return pixel_map, self.variances.reshape(self.pixel_height, self.pixel_width) / np.sum(self.variances)

    def process_samples(self, samples, pixel_map, increment=True):
        sample_intensities = np.linalg.norm(samples, axis=2)

        delta = sample_intensities - self.var_means[pixel_map[1], pixel_map[0]]
        np.add.at(self.var_means, (pixel_map[1], pixel_map[0]), delta)
        delta2 = sample_intensities - self.var_means[pixel_map[1], pixel_map[0]]
        np.add.at(self.var_m2, (pixel_map[1], pixel_map[0]), delta * delta2)
        self.variances = self.var_m2 / (self.sample_counts - 1)

        this_image = np.zeros_like(self.image)
        np.add.at(this_image, (pixel_map[1], pixel_map[0]), samples)
        self.image += this_image
        if increment:
            np.add.at(self.sample_counts, (pixel_map[1], pixel_map[0]), 1)
        return this_image

    def get_image(self):
        return tone_map(self.image / (self.sample_counts.reshape(self.pixel_height, self.pixel_width, 1).astype(float)))

    @property
    def grid(self):
        return np.array(np.meshgrid(np.arange(self.pixel_width), np.arange(self.pixel_height)))


def tone_map(image):
    print(f"min {np.min(image)} max {np.max(image)}")
    tone_vector = np.array([0.0722, 0.7152, 0.2126])
    tone_sums = np.sum(image * tone_vector, axis=2)
    log_tone_sums = np.log(0.1 + tone_sums)
    per_pixel_lts = np.sum(log_tone_sums) / np.prod(image.shape[:2])
    Lw = np.exp(per_pixel_lts)
    result = image * 2. / Lw
    return (255 * result / (result + 1)).astype(np.uint8)


if __name__ == '__main__':
    c = Camera()
