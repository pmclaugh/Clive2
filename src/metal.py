import metalcompute
import numpy as np
from camera import Camera, tone_map, basic_tone_map
from bvh import construct_BVH, np_flatten_bvh
import cv2
import metalcompute as mc
from collections import Counter
import time
from struct_types import Path
from struct_types import Camera as camera_struct
from datetime import datetime
import argparse
import os
from load import *
from adaptive import get_adaptive_indices


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--total-frames', type=int, default=1)
    parser.add_argument('--movie-name', type=str, default='default')
    parser.add_argument('--save-on-quit', action='store_true')
    parser.add_argument("--scene", type=str, default="teapots")
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()

    os.makedirs(f'../output/{args.movie_name}', exist_ok=True)

    # Metal stuff. get device, load and compile kernels
    dev = mc.Device()
    with open("trace.metal", "r") as f:
        kernel = f.read()
    trace_fn = dev.kernel(kernel).function("generate_paths")
    join_fn = dev.kernel(kernel).function("connect_paths")
    camera_ray_fn = dev.kernel(kernel).function("generate_camera_rays")
    light_ray_fn = dev.kernel(kernel).function("generate_light_rays")
    adaptive_finalize_fn = dev.kernel(kernel).function("adaptive_finalize_samples")
    light_image_gather_fn = dev.kernel(kernel).function("light_image_gather")

    tris = []
    if args.scene == "empty":
        cam_center = np.array([0, 1.5, 6])
        cam_dir = unit(np.array([0, 0, -1]))
    elif args.scene == "teapots":
        # load the teapots
        tris += load_obj('../resources/teapot.obj', offset=np.array([0, 0, 2.5]), material=5)
        tris += load_obj('../resources/teapot.obj', offset=np.array([0, 0, -2.5]), material=0)
        cam_center = np.array([7, 0, 8])
        cam_dir = unit(np.array([-1, 0, -1]))
    elif args.scene == "dragon":
        # load a reasonable dragon
        load_time = time.time()
        tris += load_ply('../resources/dragon_vrip_res3.ply', offset=np.array([0, -4, 0]), material=5, scale=50)
        print(f"done loading dragon in {time.time() - load_time}")
        cam_center = np.array([0, 1.5, 7.5])
        cam_dir = unit(np.array([0, 0, -1]))
    elif args.scene == "medium-dragon":
        # load a reasonable dragon
        load_time = time.time()
        tris += load_ply('../resources/dragon_vrip_res2.ply', offset=np.array([0, -4, 0]), material=5, scale=50)
        print(f"done loading dragon in {time.time() - load_time}")
        cam_center = np.array([0, 1.5, 7.5])
        cam_dir = unit(np.array([0, 0, -1]))
    elif args.scene == "big-dragon":
        # load the big dragon
        load_time = time.time()
        tris += load_ply('../resources/dragon_vrip.ply', offset=np.array([0, -4, 0]), material=5, scale=50)
        print(f"done loading dragon in {time.time() - load_time}")
        cam_center = np.array([0, 1.5, 7.5])
        cam_dir = unit(np.array([0, 0, -1]))
    elif args.scene == "bunny":
        # load the bunny
        load_time = time.time()
        tris += load_obj('../resources/stanford-bunny.obj', offset=np.array([0, -4, 0]), material=5, scale=50)
        print(f"done loading bunny in {time.time() - load_time}")
        cam_center = np.array([0, 1.5, 6])
        cam_dir = unit(np.array([0, 0, -1]))
    elif args.scene == "double-dragon":
        # load the dragon
        load_time = time.time()
        tris += load_ply('../resources/dragon_vrip_res2.ply', offset=np.array([-3, -4, 0]), material=5, scale=40)
        tris += load_ply('../resources/dragon_vrip_res2.ply', offset=np.array([3, -4, -2]), material=0, scale=40)
        print(f"done loading dragon in {time.time() - load_time}")
        cam_center = np.array([0, 2.5, 6])
        cam_dir = unit(np.array([0, 0, -1]))
    elif args.scene == "happy":
        load_time = time.time()
        tris += load_ply('../resources/happy_vrip_res2.ply', offset=np.array([0, -4, 0]), material=5, scale=35)
        cam_center = np.array([0, 1, 6])
        cam_dir = unit(np.array([0, 0, -1]))
        print(f"done loading happy in {time.time() - load_time}")
    elif args.scene == "happy-small":
        load_time = time.time()
        tris += load_ply('../resources/happy_vrip_res4.ply', offset=np.array([0, -4, 0]), material=5, scale=35)
        cam_center = np.array([0, 1, 6])
        cam_dir = unit(np.array([0, 0, -1]))
        print(f"done loading happy in {time.time() - load_time}")
    elif args.scene == "big-happy":
        load_time = time.time()
        tris += load_ply('../resources/happy_vrip.ply', offset=np.array([0, -4, 0]), material=5, scale=35)
        cam_center = np.array([0, 1, 6])
        cam_dir = unit(np.array([0, 0, -1]))
        print(f"done loading happy in {time.time() - load_time}")
    else:
        raise ValueError(f"Unknown scene {args.scene}")

    smooth_time = time.time()
    smooth_normals(tris)
    print("done smoothing normals in", time.time() - smooth_time)

    # manually define a box around the teapots, don't smooth it
    box_tris = triangles_for_box(np.array([-10, -2, -10]), np.array([10, 10, 10]), light_height=0.95, light_scale=.25)
    dummy_smooth_normals(box_tris)
    tris += box_tris

    # camera setup
    c = Camera(
        center=cam_center,
        direction=cam_dir,
        pixel_width=args.width,
        pixel_height=args.height,
        phys_width=args.width / args.height,
        phys_height=1,
    )
    camera_arr = c.to_struct()
    camera_tris = camera_geometry(c)
    dummy_smooth_normals(camera_tris)
    tris += camera_tris

    # build and marshall BVH
    start_time = time.time()
    bvh = construct_BVH(tris)
    print("done building bvh", time.time() - start_time)
    boxes, triangles = np_flatten_bvh(bvh)
    print("done flattening bvh")

    # make a bunch of buffers
    box_buffer = dev.buffer(boxes)
    tri_buffer = dev.buffer(triangles)
    mat_buffer = dev.buffer(get_materials())

    summed_image = np.zeros((c.pixel_height, c.pixel_width, 3), dtype=np.float32)
    summed_light_image = np.zeros((c.pixel_height, c.pixel_width, 3), dtype=np.float32)
    summed_sample_counts = np.zeros((c.pixel_height, c.pixel_width, 1), dtype=np.int32)
    summed_sample_weights = np.zeros((c.pixel_height, c.pixel_width, 1), dtype=np.float32)

    to_display = np.zeros(summed_image.shape, dtype=np.uint8)
    light_image = np.zeros(summed_image.shape, dtype=np.float32)
    batch_size = c.pixel_width * c.pixel_height

    out_camera_image = dev.buffer(batch_size * 16)
    out_camera_paths = dev.buffer(batch_size * Path.itemsize)
    out_camera_debug_image = dev.buffer(batch_size * 16)

    out_light_image = dev.buffer(batch_size * 16)
    out_light_paths = dev.buffer(batch_size * Path.itemsize)
    out_light_debug_image = dev.buffer(batch_size * 16)
    out_light_indices = dev.buffer(batch_size * 8 * 4)
    out_light_path_indices = dev.buffer(batch_size * 8 * 4)
    out_light_ray_indices = dev.buffer(batch_size * 8 * 4)
    out_light_weights = dev.buffer(batch_size * 8 * 4)
    out_light_shade = dev.buffer(batch_size * 8 * 4 * 4)

    final_out_samples = dev.buffer(batch_size * 16)
    final_out_light_image = dev.buffer(batch_size * 16)
    weight_aggregators = dev.buffer(batch_size * 128)
    finalized_samples = dev.buffer(batch_size * 16)
    sample_counts = dev.buffer(batch_size * 4)

    camera_buffer = dev.buffer(np.array([c.to_struct()]))
    camera_ray_buffer = dev.buffer(batch_size * Ray.itemsize)
    indices_buffer = dev.buffer(batch_size * 4)
    summed_bins_buffer = dev.buffer((batch_size + 1) * 4)
    sample_weights = dev.buffer(batch_size * 4)

    light_ray_buffer = dev.buffer(batch_size * Ray.itemsize)
    light_triangles = np.array([t for t in triangles if t['is_light'] == 1])
    light_triangle_buffer = dev.buffer(light_triangles)
    light_counts = dev.buffer(np.array(len(light_triangles), dtype=np.int32))
    light_surface_areas = dev.buffer(np.array([surface_area(t) for t in light_triangles], dtype=np.float32))
    light_triangle_indices = np.array([i for i, t in enumerate(triangles) if t['is_light'] == 1], dtype=np.int32)
    camera_triangle_indices = np.array([i for i, t in enumerate(triangles) if t['is_camera'] == 1], dtype=np.int32)

    # populate initial rand buffer
    randoms = np.random.randint(0, 2 ** 32, size=(batch_size, 2), dtype=np.uint32)
    rand_buffer = dev.buffer(randoms)

    f = args.start_frame
    while f < args.total_frames:

        if args.total_frames > 1:
            # temporary to make a movie
            time_parameter = f / args.total_frames
            tris = []
            materials = get_materials()
            alpha = (np.sin(time_parameter * 2 * np.pi) + 1) / 2 * 0.9
            print(f"frame {f}, alpha {alpha}")
            materials[5]['alpha'] = alpha
            mat_buffer = dev.buffer(materials)

            # zero image buffers
            summed_image = np.zeros((c.pixel_height, c.pixel_width, 3), dtype=np.float32)
            summed_light_image = np.zeros((c.pixel_height, c.pixel_width, 3), dtype=np.float32)
            summed_sample_counts = np.zeros((c.pixel_height, c.pixel_width, 1), dtype=np.int32)
            summed_sample_weights = np.zeros((c.pixel_height, c.pixel_width, 1), dtype=np.float32)

        try:
            # render loop
            for i in range(args.samples):
                start_sample_time = time.time()

                # make camera rays and rands
                mc.release(indices_buffer)
                mc.release(summed_bins_buffer)
                if args.adaptive and i > 0:
                    bins, summed_bins, indices = get_adaptive_indices(tone_map(summed_image / summed_sample_weights))
                else:
                    indices = np.arange(batch_size, dtype=np.uint32)
                    bins = np.ones(batch_size, dtype=np.uint32)
                    summed_bins = np.arange(batch_size + 1, dtype=np.uint32)
                indices_buffer = dev.buffer(indices)
                summed_bins_buffer = dev.buffer(summed_bins)

                # make camera rays
                camera_ray_fn(batch_size, camera_buffer, rand_buffer, indices_buffer, camera_ray_buffer)

                # trace
                trace_fn(batch_size, camera_ray_buffer, box_buffer, tri_buffer, mat_buffer, rand_buffer, out_camera_image, out_camera_paths, out_camera_debug_image)

                # unidirectional outputs
                unidirectional_image = np.frombuffer(out_camera_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]
                image = unidirectional_image
                finalized_sample_counts = np.ones((c.pixel_height, c.pixel_width, 1), dtype=np.int32)
                finalized_sample_weights = np.ones((c.pixel_height, c.pixel_width, 1), dtype=np.float32)

                if not args.unidirectional:
                    # light paths
                    light_ray_fn(batch_size, light_triangle_buffer, light_surface_areas, light_triangle_indices, mat_buffer, rand_buffer, light_ray_buffer, light_counts)
                    trace_fn(batch_size, light_ray_buffer, box_buffer, tri_buffer, mat_buffer, rand_buffer, out_light_image, out_light_paths, out_light_debug_image)

                    # join
                    join_fn(batch_size, out_camera_paths, out_light_paths, tri_buffer, mat_buffer, box_buffer, camera_buffer,
                            weight_aggregators, final_out_samples, out_light_indices, out_light_path_indices, out_light_ray_indices, out_light_weights, out_light_shade)

                    # retrieve outputs
                    bidirectional_image = np.frombuffer(final_out_samples, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]

                    # finalize
                    adaptive_finalize_fn(batch_size, weight_aggregators, camera_buffer, finalized_samples, sample_counts, summed_bins_buffer, sample_weights)
                    finalized_image = np.frombuffer(finalized_samples, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]

                    # test: light indices
                    light_image_indices = np.frombuffer(out_light_indices, dtype=np.int32)
                    light_path_indices = np.frombuffer(out_light_path_indices, dtype=np.int32)
                    light_ray_indices = np.frombuffer(out_light_ray_indices, dtype=np.int32)
                    light_weights = np.frombuffer(out_light_weights, dtype=np.float32)
                    light_shade = np.frombuffer(out_light_shade, dtype=np.float32)

                    bins = np.bincount(light_image_indices[light_image_indices >= 0], minlength=image.shape[0] * image.shape[1])
                    summed_bins = dev.buffer(np.insert(np.cumsum(bins), 0, 0).astype(np.uint32))
                    sorting_indices = np.argsort(light_image_indices)
                    missed_count = np.sum(light_image_indices < 0)

                    if missed_count < batch_size * 8:
                        sorted_path_indices = dev.buffer(light_path_indices[sorting_indices][missed_count:])
                        sorted_ray_indices = dev.buffer(light_ray_indices[sorting_indices][missed_count:])
                        sorted_light_weights = dev.buffer(light_weights[sorting_indices][missed_count:])
                        sorted_light_shade = dev.buffer(light_shade[sorting_indices][missed_count:])

                        light_image_gather_fn(batch_size, out_light_paths, mat_buffer, sorted_path_indices, sorted_ray_indices, summed_bins, sorted_light_weights, sorted_light_shade, out_light_image, sample_weights)
                        light_image = np.frombuffer(out_light_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]

                    finalized_sample_counts = np.frombuffer(sample_counts, dtype=np.int32).reshape(c.pixel_height, c.pixel_width, 1)
                    finalized_sample_weights = np.frombuffer(sample_weights, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 1)

                    # image = light_image
                    # image = bidirectional_image
                    # image = finalized_image
                    image = light_image + finalized_image

                summed_image += np.nan_to_num(image, posinf=0, neginf=0)
                summed_light_image += np.nan_to_num(light_image, posinf=0, neginf=0)
                summed_sample_counts += finalized_sample_counts
                summed_sample_weights += finalized_sample_weights

                # to_display = tone_map(summed_image)
                # to_display = tone_map(summed_image / (i + 1))
                to_display = tone_map(summed_image / summed_sample_weights, exposure=4.0)
                # to_display = tone_map(image / np.maximum(1, finalized_sample_counts))
                # to_display = summed_sample_counts / np.max(summed_sample_counts)
                # to_display = finalized_sample_counts / np.max(finalized_sample_counts)

                if args.filter:
                    to_display = cv2.bilateralFilter(to_display, 9, 50, 50)

                if np.any(np.isnan(to_display)):
                    print("NaNs in to_display!!!")
                    break

                # display the image
                cv2.imshow('image', to_display)
                cv2.waitKey(1)

                print(f"Whole sample {i} time: {time.time() - start_sample_time}")

        except (KeyboardInterrupt, metalcompute.error):
            if args.save_on_quit:
                pass
            else:
                raise
        # save the image
        if args.total_frames == 1:
            cv2.imwrite(f'../output/{args.movie_name}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', to_display)
        else:
            cv2.imwrite(f'../output/{args.movie_name}/frame_{f:04d}.png', to_display)

        f += 1
        print("most sampled pixel:", np.argmax(summed_sample_counts), "with", np.max(summed_sample_counts), "samples")
