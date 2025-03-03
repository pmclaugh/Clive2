import numpy as np
from constants import *
from struct_types import Ray
from struct_types import Camera as camera_struct


class Camera:
    def __init__(self, center=np.zeros(3), direction=np.array([1, 0, 0]), phys_width=1.0, phys_height=1.0,
                 pixel_width=1280, pixel_height=720):
        self.center = center
        self.direction = direction
        self.phys_width = phys_width
        self.phys_height = phys_height
        self.aspect_ratio = phys_width / phys_height
        self.h_fov = H_FOV
        self.v_fov = 2.0 * np.arctan(np.tan(H_FOV / 2.0) / self.aspect_ratio)
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

        self.dx_dp = self.dx * self.phys_width / self.pixel_width
        self.dy_dp = self.dy * self.phys_height / self.pixel_height
        self.pixel_phys_size = np.linalg.norm(self.dx_dp) * np.linalg.norm(self.dy_dp)

        self.origin = center - self.dx * phys_width / 2 - self.dy * phys_height / 2

    @property
    def focal_dist(self):
        return self.phys_width / (2 * np.tan(self.h_fov / 2))

    @property
    def focal_point(self):
        return self.center + self.focal_dist * self.direction

    @property
    def dx(self):
        if abs(self.direction[0]) < 0.0001:
            return UNIT_X if self.direction[2] > 0 else UNIT_X * -1
        else:
            dx = np.cross(self.direction * (UNIT_X + UNIT_Z), UNIT_Y * -1)
            return dx / np.linalg.norm(dx)

    @property
    def dy(self):
        if abs(self.direction[1]) < 0.0001:
            return UNIT_Y
        else:
            dy = np.cross(self.direction, self.dx)
            return dy / np.linalg.norm(dy)

    def to_struct(self):
        c = np.zeros(1, dtype=camera_struct)
        c[0]['center'][:3] = self.center
        c[0]['focal_point'][:3] = self.focal_point
        c[0]['direction'][:3] = self.direction
        c[0]['dx'][:3] = self.dx
        c[0]['dy'][:3] = self.dy
        c[0]['pixel_width'] = self.pixel_width
        c[0]['pixel_height'] = self.pixel_height
        c[0]['phys_width'] = self.phys_width
        c[0]['phys_height'] = self.phys_height
        c[0]['h_fov'] = self.h_fov
        c[0]['v_fov'] = self.v_fov
        return c


def tone_map(image, exposure=2.0, white_point=1.0):
    # print(f"IN min: {np.min(image)}, mean: {np.mean(image)}, max: {np.max(image)}")
    tone_vector = np.array([0.0722, 0.7152, 0.2126])
    tone_sums = np.sum(image * tone_vector, axis=2)
    log_tone_sums = np.log(0.1 + tone_sums)
    per_pixel_lts = np.sum(log_tone_sums) / np.prod(image.shape[:2])
    Lw = np.exp(per_pixel_lts)
    result = image * exposure / Lw
    # print(f"OUT min: {np.min(result)}, mean: {np.mean(result)}, max: {np.max(result)}")
    return (255 * result / (result + white_point ** 2)).astype(np.uint8)


def basic_tone_map(image):
    return (255 * np.sqrt(image) / image).astype(np.uint8)
