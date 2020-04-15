import numpy as np
import numba
from scene import SimpleScene
from constants import *
from primitives import unit, point, vec, Ray
from multiprocessing.dummy import Pool
from datetime import datetime

# @numba.jitclass([
#     ('center', numba.float32[3]),
#     ('direction', numba.float32[3]),
#     ('phys_width', numba.float32),
#     ('phys_height', numba.float32),
#     ('focal_dist', numba.float32),
#     ('focal_point', numba.float32[3]),
#     ('pixel_width', numba.int32),
#     ('pixel_height', numba.int32),
#     ('dx', numba.float32[3]),
#     ('dy', numba.float32[3]),
#     ('origin', numba.float32[3]),
#     ('image', numba.float32[:, :, :]),
# ])
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

        self.origin = (center - self.dx * phys_width / 2 - self.dy * phys_height / 2).astype(np.float32)

        self.image = np.zeros((pixel_height, pixel_width, 3), dtype=np.float32)

        self.pool = Pool()

    def pixel_grid(self):
        x_range = np.arange(0, 1, 1 / self.pixel_width) * self.phys_width
        # camera is inherently upside down (like a pinhole camera), reverse that quietly here
        y_range = np.arange(1, 0, -1 / self.pixel_height) * self.phys_height
        pixel_origins = np.ones_like(self.image) * self.origin
        # todo this is bad and slow
        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                pixel_origins[i][j] += self.dy * y + self.dx * x
        return pixel_origins

    def capture(self, scene: SimpleScene):
        s = datetime.now()
        origins = self.pixel_grid()
        directions = self.focal_point - origins
        # todo this is bad and slow
        for i in range(origins.shape[0]):
            for j in range(origins.shape[1]):
                ray = Ray(origins[i][j], directions[i][j])
                self.pool.apply_async(self.hit_for_pixel, [self.image, ray, scene, i, j])
        print('setup took', datetime.now() - s)
        self.pool.close()
        self.pool.join()
        return self.image

    @staticmethod
    def hit_for_pixel(image, ray, scene, i, j):
        hit = scene.hit(ray)
        if hit is not None:
            image[i][j] = hit.color


if __name__ == '__main__':
    Camera()
