import numpy as np
from camera import Camera
import numba
import objloader
from constants import *
from bvh import construct_BVH, np_flatten_bvh, BoxGroup, TriangleGroup
import cv2
from utils import timed
import metalcompute as mc
import time


class Triangle:
    v0 = np.zeros(3, dtype=np.float64)
    v1 = np.zeros(3, dtype=np.float64)
    v2 = np.zeros(3, dtype=np.float64)
    n0 = np.zeros(3, dtype=np.float64)
    n1 = np.zeros(3, dtype=np.float64)
    n2 = np.zeros(3, dtype=np.float64)
    t0 = np.zeros(3, dtype=np.float64)
    t1 = np.zeros(3, dtype=np.float64)
    t2 = np.zeros(3, dtype=np.float64)

    @property
    def min(self):
        return np.minimum(self.v0, np.minimum(self.v1, self.v2))

    @property
    def max(self):
        return np.maximum(self.v0, np.maximum(self.v1, self.v2))

    @property
    def n(self):
        a = cross(self.v1 - self.v0, self.v2 - self.v0)
        return a / np.linalg.norm(a)


def load_obj(obj_path):
    obj = objloader.Obj.open(obj_path)
    triangles = []
    for i, ((v0, n0, t0), (v1, n1, t1), (v2, n2, t2)) in enumerate(zip(*[iter(obj.face)] * 3)):
        triangle = Triangle()
        # vertices
        triangle.v0 = np.array(obj.vert[v0 - 1])
        triangle.v1 = np.array(obj.vert[v1 - 1])
        triangle.v2 = np.array(obj.vert[v2 - 1])

        # normals
        triangle.n0 = np.array(obj.norm[n0 - 1]) if n0 is not None else INVALID
        triangle.n1 = np.array(obj.norm[n1 - 1]) if n1 is not None else INVALID
        triangle.n2 = np.array(obj.norm[n2 - 1]) if n2 is not None else INVALID

        # texture UVs
        triangle.t0 = np.array(obj.text[t0 - 1]) if t0 is not None else INVALID
        triangle.t1 = np.array(obj.text[t1 - 1]) if t1 is not None else INVALID
        triangle.t2 = np.array(obj.text[t2 - 1]) if t2 is not None else INVALID

        triangles.append(triangle)
    return triangles


if __name__ == '__main__':
    tris = load_obj('../resources/teapot.obj')
    bvh = construct_BVH(tris)
    c = Camera(
        center=np.array([0, 0, -5]),
        direction=np.array([0, 0, 1]),
    )
    rays = c.ray_batch_numpy()

    boxes, triangles = np_flatten_bvh(bvh)

    dev = mc.Device()
    with open("trace.metal", "r") as f:
        kernel = f.read()

    kernel_fn = dev.kernel(kernel).function("bounce")
    buf_0 = rays.flatten()
    buf_1 = boxes.flatten()
    buf_2 = triangles.flatten()
    buf_3 = dev.buffer(np.size(rays) * 16)
    buf_4 = dev.buffer(16)

    start_time = time.time()
    kernel_fn(rays.size, buf_0, buf_1, buf_2, buf_3, buf_4)
    print(f"Render time: {time.time() - start_time}")

    retrieved_image = np.frombuffer(buf_3, dtype=np.float32).reshape(rays.shape[0], rays.shape[1], 4)
    retrieved_values = np.frombuffer(buf_4, dtype=np.int32)

    print("ok")
    print(retrieved_image.shape)
    print(retrieved_values)

    # open a window to display the image
    cv2.imshow('image', retrieved_image[:, :, :3])
    cv2.waitKey(0)
