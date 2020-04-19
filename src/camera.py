import numpy as np
import numba
from constants import *
from primitives import unit, point, vec, Ray


@numba.jitclass([
    ('center', numba.float32[3]),
    ('direction', numba.float32[3]),
    ('phys_width', numba.float32),
    ('phys_height', numba.float32),
    ('focal_dist', numba.float32),
    ('focal_point', numba.float32[3]),
    ('pixel_width', numba.int32),
    ('pixel_height', numba.int32),
    ('dx', numba.float32[3]),
    ('dy', numba.float32[3]),
    ('dx_dp', numba.float32[3]),
    ('dy_dp', numba.float32[3]),
    ('origin', numba.float32[3]),
    ('image', numba.float32[:, :, :]),
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

        self.origin = (center - self.dx * phys_width / 2 - self.dy * phys_height / 2).astype(np.float32)

        self.image = np.zeros((pixel_height, pixel_width, 3), dtype=np.float32)

    def make_ray(self, i, j):
        # was having difficulty making a good mass-ray-generation routine, settled on on-demand
        # speed is fine and it'll be good for future adaptive sampling stuff
        n1 = np.random.random()
        n2 = np.random.random()
        # todo: uniform sampling here is a little iffy, does it result in oversampling the pixel edges?
        #  is this where those moire patterns come from?
        origin = self.origin + self.dx_dp * (j + n1) + self.dy_dp * (i + n2)
        ray = Ray(origin.astype(np.float32), unit(self.focal_point - origin).astype(np.float32))
        ray.i = i
        ray.j = j
        return ray


def tone_map(camera):
    tone_vector = point(0.0722, 0.7152, 0.2126)
    Lw = np.exp(np.sum(np.log(0.1 + np.sum(camera.image * tone_vector, axis=2))) / np.product(camera.image.shape))
    result = camera.image * 0.64 / Lw
    result = result / (result + 1)
    return (result * 255).astype(np.uint8)


if __name__ == '__main__':
    c = Camera()
