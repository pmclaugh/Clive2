import numpy as np
import numba
from constants import *
from primitives import unit, point, vec
import cv2
from collections import defaultdict
from struct_types import Ray
from struct_types import Camera as camera_struct


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
        self.sample_counts = np.zeros((pixel_height, pixel_width), dtype=np.int64)
        self.variances = np.zeros_like(self.sample_counts, dtype=np.float64)
        self.means = np.zeros_like(self.sample_counts, dtype=np.float64)
        self.sums = np.zeros_like(self.sample_counts, dtype=np.float64)
        self.m2 = np.zeros_like(self.sample_counts, dtype=np.float64)

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

        weights = np.ones(batch.shape) / (self.phys_width * self.phys_height)

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
        variance_roller = np.cumsum(self.variances.flatten())

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

        max_pick = 0
        max_count = 0
        for pick, count in pick_counts.items():
            if count > max_count:
                max_pick = pick
                max_count = count

        print(f"max pick count {max_count} at {max_pick}, min pick count {np.min(list(pick_counts.values()))}")

        print(f"sum of picks {np.sum(list(pick_counts.values()))}")

        return pixel_map, self.variances.reshape(self.pixel_height, self.pixel_width) / np.sum(self.variances)

    def process_samples(self, samples, pixel_map, increment=True, adaptive=False):
        if not adaptive:
            self.sample_counts += 1
            self.image += samples
            return

        sample_intensities = np.linalg.norm(samples, axis=2)
        first = np.all(self.sample_counts == 0)
        this_sample_counts = np.zeros_like(self.sample_counts)
        for n, (i, j) in enumerate(zip(pixel_map[1].flatten(), pixel_map[0].flatten())):
            # i, j are the pixel
            # a, b are the sample

            a, b = n // self.pixel_width, n % self.pixel_width

            if increment:
                self.sample_counts[i, j] += 1
                this_sample_counts[i, j] += 1

            delta = sample_intensities[a, b] - self.means[i, j]
            self.means[i, j] += delta / self.sample_counts[i, j]
            delta2 = sample_intensities[a, b] - self.means[i, j]
            self.m2[i, j] += delta * delta2

            self.image[i, j] += samples[a, b]

        if not first:
            self.variances = self.m2 / (self.sample_counts - 1)
        print(f"max variance {np.max(self.variances)}, min variance {np.min(self.variances)}")
        print("sample count", np.sum(this_sample_counts))
        print("most sampled", np.max(this_sample_counts), "at", np.argmax(this_sample_counts))

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
