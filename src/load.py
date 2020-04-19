import objloader
import logging
import numpy as np
from primitives import point, Triangle
from utils import timed
from constants import Material

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
    # build the vertices and triangles
    verts = [point(*vert) for vert in obj.vert]
    triangles = [Triangle(verts[v0 - 1], verts[v1 - 1], verts[v2 - 1], material=material)
                 for (v0, _, _), (v1, _, _), (v2, _, _) in zip(*[iter(obj.face)]*3)]
    return triangles


if __name__ == '__main__':
    teapot = load_obj('resources/teapot.obj')
    print(len(teapot))