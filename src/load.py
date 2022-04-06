import objloader
import logging
import numpy as np
from primitives import point, Triangle
from utils import timed
from constants import Material
import numba

logger = logging.getLogger('rtv3-loader')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


@timed
def load_obj(path, material=Material.DIFFUSE.value):
    obj = objloader.Obj.open(path)
    logger.info('model %s has %d vertices and %d faces', path, len(obj.vert), len(obj.face)/3)
    if obj.norm:
        logger.info('model %s specifies normal vectors', path)
    if obj.text:
        logger.info('model %s is texture-mapped', path)

    packed_model = obj.pack()
    # build the vertices and triangles
    vertices = numba.typed.List()
    for vertex in obj.vert:
        vertices.append(point(*vertex))
    triangles = numba.typed.List()
    for (v0, _, _), (v1, _, _), (v2, _, _) in zip(*[iter(obj.face)]*3):
        triangles.append(Triangle(vertices[v0 - 1], vertices[v1 - 1], vertices[v2 - 1], material=material))
    return triangles


if __name__ == '__main__':
    teapot = load_obj('resources/teapot.obj')
    print(len(teapot))