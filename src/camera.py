import numpy as np
import numba
from constants import *
from primitives import unit, point, vec
import cv2
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
        self.var_means = np.zeros_like(self.sample_counts, dtype=np.float64)
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

    def make_ray(self, i, j):
        n1 = np.random.random()
        n2 = np.random.random()
        origin = self.origin + self.dx_dp * (j + n1) + self.dy_dp * (i + n2)
        ray = Ray(origin, unit(self.focal_point - origin))
        ray.i = i
        ray.j = j
        return ray

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
            origins, directions, indices, pixels = self.ray_batch(self.adaptive_grid)
        else:
            origins, directions, indices, pixels = self.ray_batch(self.grid)
        batch['origin'] = 0
        batch['origin'][:, :, :3] = origins + 0.0001 * directions
        batch['direction'] = 0
        batch['direction'][:, :, :3] = directions
        batch['inv_direction'] = 0
        batch['inv_direction'][:, :, :3] = 1.0 / directions
        batch['color'] = np.ones(4)
        batch['c_importance'] = 1.0 / (self.phys_width * self.phys_height)
        batch['l_importance'] = 1.0  # set in kernel
        batch['tot_importance'] = 1.0 / (self.phys_width * self.phys_height)
        batch['hit_light'] = -1
        batch['hit_camera'] = -1
        batch['material'] = -1
        batch['normal'] = 0
        batch['normal'][:, :, :3] = directions
        batch['from_camera'] = 1
        batch['triangle'] = -1
        batch['i'] = pixels[0]
        batch['j'] = pixels[1]
        return batch, pixels

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
        rolls = np.random.rand(self.pixel_height * self.pixel_width) * np.max(variance_roller)
        picks = np.searchsorted(variance_roller, rolls)
        map = np.zeros((2, self.pixel_height, self.pixel_width), dtype=np.int32)
        map[0] = np.reshape(picks // self.pixel_height, (self.pixel_height, self.pixel_width))
        map[1] = np.reshape(picks % self.pixel_height, (self.pixel_height, self.pixel_width))
        return map

    def process_samples(self, samples, map):
        sample_intensities = np.sum(samples, axis=2)
        for (i, j) in zip(map[0].flatten(), map[1].flatten()):
            self.image[j, i] += samples[j, i]
            self.sample_counts[j, i] += 1
            delta = sample_intensities[j, i] - self.var_means[j, i]
            self.var_means[j, i] += delta / self.sample_counts[j, i]
            delta2 = sample_intensities[j, i] - self.var_means[j, i]
            self.var_m2[j, i] += delta * delta2
        self.variances = self.var_m2 / self.sample_counts

    def get_image(self):
        return tone_map(self.image / (self.sample_counts.reshape(self.pixel_height, self.pixel_width, 1)))

    @property
    def grid(self):
        return np.array(np.meshgrid(np.arange(self.pixel_width), np.arange(self.pixel_height)))


def composite_image(camera):
    total_image = camera.image * 0
    for s, row in enumerate(camera.images):
        for t, sub_image in enumerate(row):
            sample_counts = np.sum(camera.sample_counts)
            if sample_counts > 0:
                weighted_sub_image = np.nan_to_num(sub_image) * camera.sample_counts[s][t] / sample_counts
                total_image += weighted_sub_image
                cv2.imwrite('../renders/components/%ds_%dt.jpg' % (s, t), tone_map(weighted_sub_image))
    return tone_map(total_image)


def tone_map(image):
    tone_vector = point(0.0722, 0.7152, 0.2126)
    # tone_vector = ONES
    tone_sums = np.sum(image * tone_vector, axis=2)
    log_tone_sums = np.log(0.1 + tone_sums)
    per_pixel_lts = np.sum(log_tone_sums) / np.prod(image.shape[:2])
    Lw = np.exp(per_pixel_lts)
    result = image * 2. / Lw
    return (255 * result / (result + 1)).astype(np.uint8)


if __name__ == '__main__':
    c = Camera()
