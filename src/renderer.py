import metalcompute as mc
import numpy as np
from struct_types import Path, Ray
from scene import Scene
from adaptive import get_adaptive_indices
from camera import tone_map


class Renderer:
    def __init__(
        self,
        scene: Scene,
        kernel_path="trace.metal",
        adaptive=False,
    ):
        # device and kernels
        dev = scene.device
        self.device = dev
        self.scene = scene

        with open(kernel_path, "r") as f:
            kernel = f.read()
        self.trace_fn = dev.kernel(kernel).function("generate_paths")
        self.join_fn = dev.kernel(kernel).function("connect_paths")
        self.camera_ray_fn = dev.kernel(kernel).function("generate_camera_rays")
        self.light_ray_fn = dev.kernel(kernel).function("generate_light_rays")
        self.finalize_fn = dev.kernel(kernel).function("adaptive_finalize_samples")
        self.light_image_gather_fn = dev.kernel(kernel).function("light_image_gather")

        # numpy image buffers
        self.summed_image = np.zeros(
            (scene.pixel_height, scene.pixel_width, 3), dtype=np.float32
        )
        self.summed_sample_counts = np.zeros(
            (scene.pixel_height, scene.pixel_width, 1), dtype=np.int32
        )
        self.summed_sample_weights = np.zeros(
            (scene.pixel_height, scene.pixel_width, 1), dtype=np.float32
        )
        self.light_image = np.zeros(self.summed_image.shape, dtype=np.float32)

        # buffers - camera and light rays
        self.pixel_width = scene.pixel_width
        self.pixel_height = scene.pixel_height
        self.batch_size = scene.pixel_width * scene.pixel_height
        self.camera_ray_buffer = dev.buffer(self.batch_size * Ray.itemsize)
        self.light_ray_buffer = dev.buffer(self.batch_size * Ray.itemsize)
        self.indices_buffer = dev.buffer(self.batch_size * 4)
        self.rand_buffer = dev.buffer(
            np.random.randint(0, 2**32, size=(self.batch_size, 2), dtype=np.uint32)
        )

        # buffers - trace
        self.out_camera_image = dev.buffer(self.batch_size * 16)
        self.out_camera_paths = dev.buffer(self.batch_size * Path.itemsize)
        self.out_camera_debug_image = dev.buffer(self.batch_size * 16)

        # buffers - join
        self.out_samples = dev.buffer(self.batch_size * 16)
        self.out_light_indices = dev.buffer(self.batch_size * 8 * 4)
        self.out_light_path_indices = dev.buffer(self.batch_size * 8 * 4)
        self.out_light_ray_indices = dev.buffer(self.batch_size * 8 * 4)
        self.out_light_weights = dev.buffer(self.batch_size * 8 * 4)
        self.out_light_shade = dev.buffer(self.batch_size * 8 * 4 * 4)

        # buffers - finalize
        self.weight_aggregators = dev.buffer(self.batch_size * 128)
        self.finalized_samples = dev.buffer(self.batch_size * 16)
        self.sample_counts = dev.buffer(self.batch_size * 4)
        self.summed_bins_buffer = dev.buffer((self.batch_size + 1) * 4)
        self.sample_weights = dev.buffer(self.batch_size * 4)

        # buffers - light image
        self.out_light_image = dev.buffer(self.batch_size * 16)
        self.out_light_paths = dev.buffer(self.batch_size * Path.itemsize)
        self.out_light_debug_image = dev.buffer(self.batch_size * 16)

        self.adaptive = adaptive
        self.samples = 0

    def assign_indices(self):
        mc.release(self.indices_buffer)
        mc.release(self.summed_bins_buffer)
        if self.adaptive and self.samples > 0:
            bins, summed_bins, indices = get_adaptive_indices(
                tone_map(self.summed_image / self.summed_sample_weights)
            )
        else:
            indices = np.arange(self.batch_size, dtype=np.uint32)
            summed_bins = np.arange(self.batch_size + 1, dtype=np.uint32)
        self.indices_buffer = self.device.buffer(indices)
        self.summed_bins_buffer = self.device.buffer(summed_bins)

    def process_light_step(self):
        light_image_indices = np.frombuffer(self.out_light_indices, dtype=np.int32)
        light_path_indices = np.frombuffer(self.out_light_path_indices, dtype=np.int32)
        light_ray_indices = np.frombuffer(self.out_light_ray_indices, dtype=np.int32)
        light_weights = np.frombuffer(self.out_light_weights, dtype=np.float32)
        light_shade = np.frombuffer(self.out_light_shade, dtype=np.float32)

        bins = np.bincount(
            light_image_indices[light_image_indices >= 0],
            minlength=self.pixel_height * self.pixel_width,
        )
        summed_bins = self.device.buffer(
            np.insert(np.cumsum(bins), 0, 0).astype(np.uint32)
        )
        missed_count = np.sum(light_image_indices < 0)

        sorting_indices = np.argsort(light_image_indices)
        sorted_path_indices = self.device.buffer(
            light_path_indices[sorting_indices][missed_count:]
        )
        sorted_ray_indices = self.device.buffer(
            light_ray_indices[sorting_indices][missed_count:]
        )
        sorted_light_weights = self.device.buffer(
            light_weights[sorting_indices][missed_count:]
        )
        sorted_light_shade = self.device.buffer(
            light_shade[sorting_indices][missed_count:]
        )

        return (
            summed_bins,
            sorted_path_indices,
            sorted_ray_indices,
            sorted_light_weights,
            sorted_light_shade,
        )

    def run_sample(self):

        self.assign_indices()

        self.camera_ray_fn(
            self.batch_size,
            self.scene.camera,
            self.rand_buffer,
            self.indices_buffer,
            self.camera_ray_buffer,
        )

        self.trace_fn(
            self.batch_size,
            self.camera_ray_buffer,
            self.scene.boxes,
            self.scene.triangles,
            self.scene.materials,
            self.rand_buffer,
            self.out_camera_image,
            self.out_camera_paths,
            self.out_camera_debug_image,
        )

        self.light_ray_fn(
            self.batch_size,
            self.scene.light_triangles,
            self.scene.light_surface_areas,
            self.scene.light_triangle_indices,
            self.scene.materials,
            self.rand_buffer,
            self.light_ray_buffer,
            self.scene.light_counts,
        )

        self.trace_fn(
            self.batch_size,
            self.light_ray_buffer,
            self.scene.boxes,
            self.scene.triangles,
            self.scene.materials,
            self.rand_buffer,
            self.out_light_image,
            self.out_light_paths,
            self.out_light_debug_image,
        )

        self.join_fn(
            self.batch_size,
            self.out_camera_paths,
            self.out_light_paths,
            self.scene.triangles,
            self.scene.materials,
            self.scene.boxes,
            self.scene.camera,
            self.weight_aggregators,
            self.out_samples,
            self.out_light_indices,
            self.out_light_path_indices,
            self.out_light_ray_indices,
            self.out_light_weights,
            self.out_light_shade,
        )

        self.finalize_fn(
            self.batch_size,
            self.weight_aggregators,
            self.scene.camera,
            self.finalized_samples,
            self.sample_counts,
            self.summed_bins_buffer,
            self.sample_weights,
        )

        finalized_image = np.frombuffer(
            self.finalized_samples, dtype=np.float32
        ).reshape(self.pixel_height, self.pixel_width, 4)[:, :, :3]

        (
            summed_bins,
            sorted_path_indices,
            sorted_ray_indices,
            sorted_light_weights,
            sorted_light_shade,
        ) = self.process_light_step()

        self.light_image_gather_fn(
            self.batch_size,
            self.out_light_paths,
            self.scene.materials,
            sorted_path_indices,
            sorted_ray_indices,
            summed_bins,
            sorted_light_weights,
            sorted_light_shade,
            self.out_light_image,
            self.sample_weights,
        )

        light_image = np.frombuffer(self.out_light_image, dtype=np.float32).reshape(
            self.pixel_height, self.pixel_width, 4
        )[:, :, :3]

        finalized_sample_counts = np.frombuffer(
            self.sample_counts, dtype=np.int32
        ).reshape(self.pixel_height, self.pixel_width, 1)
        finalized_sample_weights = np.frombuffer(
            self.sample_weights, dtype=np.float32
        ).reshape(self.pixel_height, self.pixel_width, 1)

        image = light_image + finalized_image

        self.summed_image += np.nan_to_num(image, posinf=0, neginf=0)
        self.summed_sample_counts += finalized_sample_counts
        self.summed_sample_weights += finalized_sample_weights

    @property
    def current_image(self):
        return tone_map(self.summed_image / self.summed_sample_weights, exposure=4.0)
