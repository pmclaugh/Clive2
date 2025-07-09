import time

import metalcompute
import numpy as np
from camera import Camera
from bvh import construct_BVH, np_flatten_bvh, FastTreeBox
from load import (
    triangles_for_box,
    fast_load_ply,
    fast_load_obj,
    get_materials,
    camera_geometry,
    surface_area,
)
from constants import (
    UNIT_Z,
    ZERO_VECTOR,
)


def create_scene(
    pixel_width=1280,
    pixel_height=720,
    cam_center=ZERO_VECTOR,
    cam_direction=UNIT_Z,
    file_specs=None,
):

    dev = metalcompute.Device()

    camera = Camera(
        center=cam_center,
        direction=cam_direction,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        phys_width=pixel_width / pixel_height,
        phys_height=1,
    )

    triangles = []
    camera_tris = camera_geometry(camera)
    triangles.extend(camera_tris)

    box_triangles = triangles_for_box()
    triangles.extend(box_triangles)

    box = FastTreeBox.from_triangle_objects(triangles)

    if file_specs:
        for file_spec in file_specs:
            if file_spec["file_path"].endswith(".ply"):
                box = box + fast_load_ply(
                    ply_path=file_spec["file_path"],
                    material=file_spec.get("material", 0),
                    scale=file_spec.get("scale", 1.0),
                    offset=file_spec.get("offset", ZERO_VECTOR),
                )
            elif file_spec["file_path"].endswith(".obj"):
                box = box + fast_load_obj(
                    obj_path=file_spec["file_path"],
                    material=file_spec.get("material", 0),
                    scale=file_spec.get("scale", 1.0),
                    offset=file_spec.get("offset", ZERO_VECTOR),
                )
            else:
                raise NotImplementedError

    bvh_start_time = time.time()
    bvh = construct_BVH(box)
    print(f"BVH construction took {time.time() - bvh_start_time:.4f} seconds")
    np_boxes, np_triangles = np_flatten_bvh(bvh)
    light_triangles = [t for t in np_triangles if t["is_light"]]

    camera_buffer = dev.buffer(np.array([camera.to_struct()]))
    box_buffer = dev.buffer(np_boxes)
    tri_buffer = dev.buffer(np_triangles)
    mat_buffer = dev.buffer(get_materials())

    light_triangles_buffer = dev.buffer(np.array(light_triangles))
    light_counts_buffer = dev.buffer(np.array(len(light_triangles), dtype=np.int32))
    light_surface_areas_buffer = dev.buffer(
        np.array([surface_area(t) for t in light_triangles], dtype=np.float32)
    )
    light_triangle_indices_buffer = np.array(
        [i for i, t in enumerate(np_triangles) if t["is_light"]], dtype=np.int32
    )
    camera_triangle_indices_buffer = np.array(
        [i for i, t in enumerate(np_triangles) if t["is_camera"]], dtype=np.int32
    )

    return Scene(
        device=dev,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        camera=camera_buffer,
        triangles=tri_buffer,
        boxes=box_buffer,
        materials=mat_buffer,
        light_triangles=light_triangles_buffer,
        light_counts=light_counts_buffer,
        light_surface_areas=light_surface_areas_buffer,
        light_triangle_indices=light_triangle_indices_buffer,
        camera_triangle_indices=camera_triangle_indices_buffer,
    )


class Scene:
    def __init__(
        self,
        device: metalcompute.Device,
        pixel_width: int,
        pixel_height: int,
        camera,
        triangles,
        boxes,
        materials,
        light_triangles,
        light_counts,
        light_surface_areas,
        light_triangle_indices,
        camera_triangle_indices,
    ):
        self.device = device
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.camera = camera
        self.triangles = triangles
        self.boxes = boxes
        self.materials = materials
        self.light_triangles = light_triangles
        self.light_counts = light_counts
        self.light_surface_areas = light_surface_areas
        self.light_triangle_indices = light_triangle_indices
        self.camera_triangle_indices = camera_triangle_indices

    def __del__(self):
        metalcompute.release(self.camera)
        metalcompute.release(self.triangles)
        metalcompute.release(self.boxes)
        metalcompute.release(self.materials)
        metalcompute.release(self.light_triangles)
        metalcompute.release(self.light_counts)
        metalcompute.release(self.light_surface_areas)
        metalcompute.release(self.light_triangle_indices)
        metalcompute.release(self.camera_triangle_indices)
        metalcompute.release(self.device)


scene_presets = {
    "empty": {
        "cam_center": np.array([0, 1.5, 6]),
        "cam_direction": np.array([0, 0, -1]),
    },
    "teapots": {
        "cam_center": np.array([7, 0, 8]),
        "cam_direction": np.array([-1, 0, -1]),
        "file_specs": [
            {
                "file_path": "../resources/teapot.obj",
                "offset": np.array([0, 0, 2.5]),
                "material": 5,
            },
            {
                "file_path": "../resources/teapot.obj",
                "offset": np.array([0, 0, -2.5]),
                "material": 0,
            },
        ],
    },
    "dragon": {
        "cam_center": np.array([0, 1.5, 7.5]),
        "cam_direction": np.array([0, 0, -1]),
        "file_specs": [
            {
                "file_path": "../resources/dragon_vrip_res3.ply",
                "offset": np.array([0, -4, 0]),
                "material": 5,
                "scale": 50,
            }
        ],
    },
    "medium-dragon": {
        "cam_center": np.array([0, 1.5, 7.5]),
        "cam_direction": np.array([0, 0, -1]),
        "file_specs": [
            {
                "file_path": "../resources/dragon_vrip_res2.ply",
                "offset": np.array([0, -4, 0]),
                "material": 5,
                "scale": 50,
            }
        ],
    },
    "big-dragon": {
        "cam_center": np.array([0, 1.5, 7.5]),
        "cam_direction": np.array([0, 0, -1]),
        "file_specs": [
            {
                "file_path": "../resources/dragon_vrip.ply",
                "offset": np.array([0, -4, 0]),
                "material": 5,
                "scale": 50,
            }
        ],
    },
}


def create_scene_from_preset(
    preset_name, pixel_width=1280, pixel_height=720
):
    preset = scene_presets.get(preset_name)
    if not preset:
        raise ValueError(f"Preset '{preset_name}' not found.")

    return create_scene(
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        cam_center=preset["cam_center"],
        cam_direction=preset["cam_direction"],
        file_specs=preset.get("file_specs"),
    )


def create_scene_from_preset_with_params(
    preset_name,
    pixel_width=1280,
    pixel_height=720,
    frame_idx=0,
    total_frames=1,
):
    preset = scene_presets.get(preset_name)
    if not preset:
        raise ValueError(f"Preset '{preset_name}' not found.")

    theta = 2 * np.pi * frame_idx / total_frames

    cam_center = np.array([np.sin(theta) * 7.5, 1.5, np.cos(theta) * 7.5])
    cam_direction = np.array([-np.sin(theta), 0, -np.cos(theta)])

    return create_scene(
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        cam_center=cam_center,
        cam_direction=cam_direction,
        file_specs=preset.get("file_specs"),
    )
