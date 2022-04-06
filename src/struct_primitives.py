import numpy as np
import objloader
from constants import INVALID

# todo: the goal here is to make much, much more compact data types that use numpy structured arrays.
#  jitclass is really cool but it creates a lot of overhead and headaches. Most of the code is just loops and static math,
#  turns out there isn't actually a huge need/motive for OO like I thought there would be.
#  instead of keeping material, color, etc in Triangle or Ray, there should just be other arrays for those, with matching indexes.

Triangle = np.dtype([
    ('v0', (float, 3)),
    ('v1', (float, 3)),
    ('v2', (float, 3)),
    ('n0', (float, 3)),
    ('n1', (float, 3)),
    ('n2', (float, 3)),
    ('t0', (float, 3)),
    ('t1', (float, 3)),
    ('t2', (float, 3))])

Ray = np.dtype([
    ('origin', (float, 3)),
    ('direction', (float, 3))])

Box = np.dtype([
    ('min', (float, 3)),
    ('max', (float, 3)),
    ('left', int),
    ('right', int)])


def load_obj(obj_path):
    obj = objloader.Obj.open(obj_path)
    triangles = np.zeros(len(obj.face), dtype=Triangle)
    for i, ((v0, n0, t0), (v1, n1, t1), (v2, n2, t2)) in enumerate(zip(*[iter(obj.face)] * 3)):
        # vertices
        triangles[i]['v0'] = obj.vert[v0 - 1]
        triangles[i]['v1'] = obj.vert[v1 - 1]
        triangles[i]['v2'] = obj.vert[v2 - 1]

        # normals
        triangles[i]['n0'] = obj.norm[n0 - 1] if n0 is not None else INVALID
        triangles[i]['n1'] = obj.norm[n1 - 1] if n1 is not None else INVALID
        triangles[i]['n2'] = obj.norm[n2 - 1] if n2 is not None else INVALID

        # texture UVs
        triangles[i]['t0'] = obj.text[t0 - 1] if t0 is not None else INVALID
        triangles[i]['t1'] = obj.text[t1 - 1] if t1 is not None else INVALID
        triangles[i]['t2'] = obj.text[t2 - 1] if t2 is not None else INVALID

    return triangles


if __name__ == '__main__':
    load_obj('resources/teapot.obj')