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
        self.focal_dist = (phys_width / 2.) / np.tan(H_FOV / 2.0)
        self.focal_point = self.center + self.focal_dist * direction
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

        if abs(self.direction[0]) < 0.0001:
            self.dx = UNIT_X if direction[2] > 0 else UNIT_X * -1
        else:
            dx = np.cross(direction * (UNIT_X + UNIT_Z), UNIT_Y * -1)
            self.dx = dx / np.linalg.norm(dx)

        if abs(self.direction[1]) < 0.0001:
            self.dy = UNIT_Y
        else:
            dy = np.cross(direction, self.dx)
            self.dy = dy / np.linalg.norm(dy)

        self.dx_dp = self.dx * self.phys_width / self.pixel_width
        self.dy_dp = self.dy * self.phys_height / self.pixel_height
        self.pixel_phys_size = np.linalg.norm(self.dx_dp) * np.linalg.norm(self.dy_dp)

        self.origin = center - self.dx * phys_width / 2 - self.dy * phys_height / 2

    def ray_batch(self):
        pixels = np.meshgrid(np.arange(self.pixel_width), np.arange(self.pixel_height))
        offsets = np.random.rand(2, self.pixel_height, self.pixel_width)
        x_vectors = np.expand_dims(pixels[0] + offsets[0], axis=2) * self.dx_dp
        y_vectors = np.expand_dims(pixels[1] + offsets[1], axis=2) * self.dy_dp
        origins = self.origin + x_vectors + y_vectors
        directions = self.focal_point - origins
        directions = directions / np.linalg.norm(directions, axis=2)[:, :, np.newaxis]
        return origins, directions

    def ray_batch_numpy(self):
        batch = np.zeros((self.pixel_height, self.pixel_width), dtype=Ray)
        origins, directions = self.ray_batch()
        batch['origin'][:, :, :3] = origins + 0.0001 * directions
        batch['direction'][:, :, :3] = directions
        batch['inv_direction'][:, :, :3] = 1.0 / directions
        batch['color'] = np.ones(4)
        batch['c_importance'] = 1.0 / (self.phys_width * self.phys_height)
        batch['l_importance'] = 1.0  # set in kernel
        batch['tot_importance'] = 1.0 / (self.phys_width * self.phys_height)
        batch['hit_light'] = -1
        batch['hit_camera'] = -1
        batch['normal'][:, :, :3] = self.direction
        batch['from_camera'] = 1
        batch['triangle'] = -1
        batch['material'] = 7
        return batch

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


def tone_map(image):
    tone_vector = np.array([0.0722, 0.7152, 0.2126])
    # tone_vector = ONES
    tone_sums = np.sum(image * tone_vector, axis=2)
    log_tone_sums = np.log(0.1 + tone_sums)
    per_pixel_lts = np.sum(log_tone_sums) / np.prod(image.shape[:2])
    Lw = np.exp(per_pixel_lts)
    result = image * 2. / Lw
    return (255 * result / (result + 1)).astype(np.uint8)

