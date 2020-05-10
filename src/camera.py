import numpy as np
import numba
from constants import *
from primitives import unit, point, vec, Ray
import cv2


@numba.experimental.jitclass([
    ('center', numba.float64[3]),
    ('direction', numba.float64[3]),
    ('phys_width', numba.float64),
    ('phys_height', numba.float64),
    ('focal_dist', numba.float64),
    ('focal_point', numba.float64[3]),
    ('pixel_width', numba.int32),
    ('pixel_height', numba.int32),
    ('dx', numba.float64[3]),
    ('dy', numba.float64[3]),
    ('dx_dp', numba.float64[3]),
    ('dy_dp', numba.float64[3]),
    ('origin', numba.float64[3]),
    ('image', numba.float64[:, :, :]),
    ('samples', numba.int64),
    ('sample_counts', numba.int64[:, :]),
    ('images', numba.float64[:, :, :, :, :]),
])
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
        self.sample_counts = np.zeros((MAX_BOUNCES + 2, MAX_BOUNCES + 2), dtype=np.int64)

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

        self.origin = center - self.dx * phys_width / 2 - self.dy * phys_height / 2

        self.image = np.zeros((pixel_height, pixel_width, 3), dtype=np.float64)
        self.images = np.zeros((MAX_BOUNCES + 2, MAX_BOUNCES + 2, pixel_height, pixel_width, 3), dtype=np.float64)

    def make_ray(self, i, j):
        # was having difficulty making a good mass-ray-generation routine, settled on on-demand
        # speed is fine and it'll be good for future adaptive sampling stuff
        n1 = np.random.random()
        n2 = np.random.random()
        origin = self.origin + self.dx_dp * (j + n1) + self.dy_dp * (i + n2)
        ray = Ray(origin, unit(self.focal_point - origin))
        ray.i = i
        ray.j = j
        return ray


def composite_image(camera):
    total_image = camera.image * 0
    for s, row in enumerate(camera.images):
        for t, sub_image in enumerate(row):
            weighted_sub_image = np.nan_to_num(sub_image) * camera.sample_counts[s][t] / np.sum(camera.sample_counts)
            total_image += weighted_sub_image
            cv2.imwrite('../renders/components/%ds_%dt.jpg' % (s, t), tone_map(weighted_sub_image))
    return tone_map(total_image)


def tone_map(image):
    # tone_vector = point(0.0722, 0.7152, 0.2126)
    tone_vector = ONES
    tone_sums = np.sum(image * tone_vector, axis=2)
    log_tone_sums = np.log(0.1 + tone_sums)
    per_pixel_lts = np.sum(log_tone_sums) / np.product(image.shape[:2])
    Lw = np.exp(per_pixel_lts)
    result = image * 2. / Lw
    return (255 * result / (result + 1)).astype(np.uint8)


if __name__ == '__main__':
    c = Camera()
