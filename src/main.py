import numpy as np
from struct_primitives import load_obj
from camera import setup_camera

if __name__ == '__main__':
    camera = setup_camera()
    print("ok")

# todo, returning to this after a year:
#  eliminate jitclass entirely, use numpy structured arrays
#  revisit multithreading
#  I hate this chainmap-config pattern, main should be clearer, maybe bring it back much later on
#  finish bvh implementation
#  integration tests