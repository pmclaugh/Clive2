import metalcompute as mc
import numpy as np
from struct_types import Path, Ray
from scene import Scene
from camera import tone_map
from constants import timed

MAX_PATH_LENGTH = 8


def next_power_of_two(n):
    """Returns the next power of two greater than or equal to n."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


class Renderer:
    def __init__(
        self,
        scene: Scene,
        kernel_path="trace.metal",
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
        self.light_sort_fn = dev.kernel(kernel).function("light_sort")
        self.light_image_gather_fn = dev.kernel(kernel).function("light_image_gather")
        self.light_reset_fn = dev.kernel(kernel).function("reset_light_indices")

        # numpy image buffers
        resolution = (scene.pixel_height, scene.pixel_width)
        self.summed_image = np.zeros((*resolution, 3), dtype=np.float32)
        self.summed_sample_counts = np.zeros((*resolution, 1), dtype=np.int32)
        self.summed_sample_weights = np.zeros((*resolution, 1), dtype=np.float32)
        self.light_image = np.zeros((*resolution, 1), dtype=np.float32)

        # buffers - camera and light rays
        self.pixel_width = scene.pixel_width
        self.pixel_height = scene.pixel_height
        self.batch_size = scene.pixel_width * scene.pixel_height
        self.camera_ray_buffer = dev.buffer(self.batch_size * Ray.itemsize)
        self.light_ray_buffer = dev.buffer(self.batch_size * Ray.itemsize)
        self.indices_buffer = dev.buffer(self.batch_size * 4)
        self.rand_buffer = dev.buffer(self.get_random_buffer())

        # buffers - trace
        self.out_camera_image = dev.buffer(self.batch_size * 16)
        self.out_camera_paths = dev.buffer(self.batch_size * Path.itemsize)
        self.out_camera_debug_image = dev.buffer(self.batch_size * 16)

        # buffers - join
        self.out_samples = dev.buffer(self.batch_size * 16)
        sz = next_power_of_two(self.batch_size * MAX_PATH_LENGTH) * 4
        self.out_light_indices = dev.buffer(sz)
        self.out_light_path_indices = dev.buffer(sz)
        self.out_light_ray_indices = dev.buffer(sz)
        self.out_light_weights = dev.buffer(sz)
        self.out_light_shade = dev.buffer(sz)

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

        self.samples = 0

        self.assign_indices()

    def get_random_buffer(self):
        return np.random.randint(0, 2**32, size=(self.batch_size, 2), dtype=np.uint32)

    @timed
    def assign_indices(self):
        indices = np.arange(self.batch_size, dtype=np.uint32)
        summed_bins = np.arange(self.batch_size + 1, dtype=np.uint32)
        self.indices_buffer = self.device.buffer(indices)
        self.summed_bins_buffer = self.device.buffer(summed_bins)

    @timed
    def light_bins(self):
        light_image_indices = np.frombuffer(self.out_light_indices, dtype=np.int32)

        bins = np.bincount(
            light_image_indices[light_image_indices >= 0],
            minlength=self.pixel_height * self.pixel_width,
        )

        summed_bins = self.device.buffer(
            np.insert(np.cumsum(bins), 0, 0).astype(np.uint32)
        )

        offset = np.sum(light_image_indices < 0).astype(np.uint32)

        return summed_bins, offset

    @timed
    def make_light_rays(self):
        _ = self.light_ray_fn(
            self.batch_size,
            self.scene.light_triangles,
            self.scene.light_surface_areas,
            self.scene.light_triangle_indices,
            self.scene.materials,
            self.rand_buffer,
            self.light_ray_buffer,
            self.scene.light_counts,
        )
        del _

    @timed
    def make_camera_rays(self):
        _ = self.camera_ray_fn(
            self.batch_size,
            self.scene.camera,
            self.rand_buffer,
            self.indices_buffer,
            self.camera_ray_buffer,
        )
        del _

    @timed
    def trace_camera_rays(self):
        _ = self.trace_fn(
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
        del _

    @timed
    def trace_light_rays(self):
        _ = self.trace_fn(
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
        del _

    @timed
    def join_paths(self):
        n = next_power_of_two(self.batch_size * MAX_PATH_LENGTH)
        _ = self.light_reset_fn(
            n,
            self.out_light_indices,
            self.out_light_path_indices,
            self.out_light_ray_indices,
            self.out_light_weights,
            self.out_light_shade,
        )
        del _

        _ = self.join_fn(
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
        del _

    @timed
    def finalize_samples(self):
        _ = self.finalize_fn(
            self.batch_size,
            self.weight_aggregators,
            self.scene.camera,
            self.finalized_samples,
            self.sample_counts,
            self.summed_bins_buffer,
            self.sample_weights,
        )
        del _

    @timed
    def gather_light_image(self):
        n = next_power_of_two(self.batch_size * MAX_PATH_LENGTH)
        log_n = int(np.log2(n))
        pairs_per_thread = 4
        for stage in range(1, log_n + 1):
            for passOfStage in range(stage, 0, -1):
                _ = self.light_sort_fn(
                    n // 8,
                    self.out_light_indices,
                    self.out_light_path_indices,
                    self.out_light_ray_indices,
                    self.out_light_weights,
                    self.out_light_shade,
                    np.uint32(stage),
                    np.uint32(passOfStage),
                    np.uint32(n),
                    np.uint32(pairs_per_thread),
                )
                del _

        bins, offset = self.light_bins()

        _ = self.light_image_gather_fn(
            self.batch_size,
            self.out_light_paths,
            self.scene.materials,
            self.out_light_path_indices,
            self.out_light_ray_indices,
            bins,
            offset,
            self.out_light_weights,
            self.out_light_shade,
            self.out_light_image,
            self.sample_weights,
        )
        del _

        mc.release(bins)

    @timed
    def process_images(self):
        finalized_image = np.frombuffer(
            self.finalized_samples, dtype=np.float32
        ).reshape(self.pixel_height, self.pixel_width, 4)[:, :, :3]

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

    @timed
    def run_sample(self):
        self.make_light_rays()
        self.make_camera_rays()
        self.trace_light_rays()
        self.trace_camera_rays()
        self.join_paths()
        self.finalize_samples()
        self.gather_light_image()
        self.process_images()

        self.samples += 1

    @property
    def image(self):
        return tone_map(
            np.nan_to_num(
                self.summed_image / self.summed_sample_weights, neginf=0, posinf=0
            ),
            exposure=4.0,
        )

    @property
    def unweighted_image(self):
        return tone_map(
            np.nan_to_num(self.summed_image, neginf=0, posinf=0),
            exposure=4.0,
        )

    def __del__(self):
        mc.release(self.camera_ray_buffer)
        mc.release(self.light_ray_buffer)
        mc.release(self.indices_buffer)
        mc.release(self.rand_buffer)

        mc.release(self.out_camera_image)
        mc.release(self.out_camera_paths)
        mc.release(self.out_camera_debug_image)

        mc.release(self.out_samples)
        mc.release(self.out_light_indices)
        mc.release(self.out_light_path_indices)
        mc.release(self.out_light_ray_indices)
        mc.release(self.out_light_weights)
        mc.release(self.out_light_shade)

        mc.release(self.weight_aggregators)
        mc.release(self.finalized_samples)
        mc.release(self.sample_counts)
        mc.release(self.summed_bins_buffer)
        mc.release(self.sample_weights)

        mc.release(self.out_light_image)
        mc.release(self.out_light_paths)
        mc.release(self.out_light_debug_image)

        mc.release(self.camera_ray_fn)
        mc.release(self.light_ray_fn)
        mc.release(self.trace_fn)
        mc.release(self.join_fn)
        mc.release(self.finalize_fn)
        mc.release(self.light_sort_fn)
        mc.release(self.light_image_gather_fn)
        mc.release(self.light_reset_fn)
