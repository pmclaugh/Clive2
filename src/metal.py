import metalcompute
import numpy as np
from camera import Camera, tone_map
from bvh import construct_BVH, np_flatten_bvh
import cv2
import metalcompute as mc
import time
from struct_types import Path
from datetime import datetime
import argparse
import os
from load import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--frame-number', type=int, default=0)
    parser.add_argument('--total-frames', type=int, default=1)
    parser.add_argument('--movie-name', type=str, default='default')
    parser.add_argument('--save-on-quit', action='store_true')
    parser.add_argument("--scene", type=str, default="teapots")
    parser.add_argument("--unidirectional", action="store_true")
    args = parser.parse_args()

    os.makedirs(f'../output/{args.movie_name}', exist_ok=True)

    # Metal stuff. get device, load and compile kernels
    dev = mc.Device()
    with open("trace.metal", "r") as f:
        kernel = f.read()
    trace_fn = dev.kernel(kernel).function("generate_paths")
    join_fn = dev.kernel(kernel).function("connect_paths")

    tris = []
    if args.scene == "empty":
        cam_center = np.array([0, 1.5, 6])
        cam_dir = unit(np.array([0, 0, -1]))
    elif args.scene == "teapots":
        # load the teapots
        tris += load_obj('../resources/teapot.obj', offset=np.array([0, 0, 2.5]), material=5)
        tris += load_obj('../resources/teapot.obj', offset=np.array([0, 0, -2.5]), material=0)
        cam_center = np.array([7, 1.5, 8])
        cam_dir = unit(np.array([-1, 0, -1]))
    elif args.scene == "dragon":
        # load a reasonable dragon
        load_time = time.time()
        tris += load_ply('../resources/dragon_vrip_res3.ply', offset=np.array([0, -4, 0]), material=5, scale=50)
        print(f"done loading dragon in {time.time() - load_time}")
        cam_center = np.array([0, 1.5, 6])
        cam_dir = unit(np.array([0, 0, -1]))
    elif args.scene == "big-dragon":
        # load the big dragon
        load_time = time.time()
        tris += load_ply('../resources/dragon_vrip.ply', offset=np.array([0, -4, 0]), material=5, scale=50)
        print(f"done loading dragon in {time.time() - load_time}")
        cam_center = np.array([0, 1.5, 6])
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
        tris += load_ply('../resources/dragon_vrip_res3.ply', offset=np.array([-2, -4, 0]), material=5, scale=50)
        tris += load_ply('../resources/dragon_vrip_res3.ply', offset=np.array([2, -4, -2]), material=0, scale=50)
        print(f"done loading dragon in {time.time() - load_time}")
        cam_center = np.array([0, 2.5, 6])
        cam_dir = unit(np.array([0, 0, -1]))
    else:
        raise ValueError(f"Unknown scene {args.scene}")

    smooth_time = time.time()
    smooth_normals(tris)
    print("done smoothing normals in", time.time() - smooth_time)

    # manually define a box around the teapots, don't smooth it
    box_tris = triangles_for_box(np.array([-10, -2, -10]), np.array([10, 10, 10]))
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

    box_buffer = dev.buffer(boxes.size * boxes.itemsize)
    tri_buffer = dev.buffer(triangles.size * triangles.itemsize)

    # load materials (very basic for now)
    mats = get_materials()

    # make a bunch of buffers
    summed_image = np.zeros((c.pixel_height, c.pixel_width, 3), dtype=np.float32)
    to_display = np.zeros(summed_image.shape, dtype=np.uint8)
    batch_size = c.pixel_width * c.pixel_height

    out_camera_image = dev.buffer(batch_size * 16)
    out_camera_paths = dev.buffer(batch_size * Path.itemsize)
    out_camera_debug_image = dev.buffer(batch_size * 16)

    out_light_image = dev.buffer(batch_size * 16)
    out_light_paths = dev.buffer(batch_size * Path.itemsize)
    out_light_debug_image = dev.buffer(batch_size * 16)

    final_out_samples = dev.buffer(batch_size * 16)

    try:
        # render loop
        for i in range(args.samples):
            trace_fn = dev.kernel(kernel).function("generate_paths")
            join_fn = dev.kernel(kernel).function("connect_paths")

            # make camera rays and rands
            camera_ray_start_time = time.time()
            camera_rays = c.ray_batch_numpy().flatten()
            print(f"Create camera rays in {time.time() - camera_ray_start_time}")

            rand_start_time = time.time()
            rands = np.random.rand(camera_rays.size * 32).astype(np.float32)
            print(f"Create camera rands in {time.time() - rand_start_time}")

            # trace camera paths
            start_time = time.time()
            trace_fn(batch_size, camera_rays, boxes, triangles, mats, rands, out_camera_image, out_camera_paths, out_camera_debug_image)
            print(f"Sample {i} camera trace time: {time.time() - start_time}")

            # retrieve camera trace outputs
            unidirectional_image = np.frombuffer(out_camera_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]
            camera_paths = np.frombuffer(out_camera_paths, dtype=Path)
            retrieved_camera_debug_image = np.frombuffer(out_camera_debug_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]

            image = unidirectional_image

            if not args.unidirectional:
                # make light rays and rands
                light_ray_start_time = time.time()
                light_rays = fast_generate_light_rays(triangles, camera_rays.size)
                print(f"Create light rays in {time.time() - light_ray_start_time}")

                rand_start_time = time.time()
                rands = np.random.rand(light_rays.size * 32).astype(np.float32)
                print(f"Create light rands in {time.time() - rand_start_time}")

                # trace light paths
                start_time = time.time()
                trace_fn(batch_size, light_rays, boxes, triangles, mats, rands, out_light_image, out_light_paths, out_light_debug_image)
                print(f"Sample {i} light trace time: {time.time() - start_time}")

                # retrieve light trace outputs
                light_paths = np.frombuffer(out_light_paths, dtype=Path)
                retrieved_light_debug_image = np.frombuffer(out_light_debug_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]

                # join paths
                start_time = time.time()
                join_fn(batch_size, out_camera_paths, out_light_paths, triangles, mats, boxes, camera_arr[0], final_out_samples)
                print(f"Sample {i} join time: {time.time() - start_time}")

                post_start_time = time.time()

                # retrieve joined path outputs
                bidirectional_image = np.frombuffer(final_out_samples, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]

                image = bidirectional_image

            # post processing. tone map, sum, division
            print(np.sum(np.isnan(image)), "nans in image")
            print(np.sum(np.any(np.isnan(image), axis=2)), "pixels with nans")
            print(np.sum(np.isinf(image)), "infs in image")
            summed_image += np.nan_to_num(image, posinf=0, neginf=0)
            if np.any(np.isnan(summed_image)):
                print("NaNs in summed image!!!")
                break
            to_display = tone_map(summed_image / (i + 1))
            if np.any(np.isnan(to_display)):
                print("NaNs in to_display!!!")
                break

            # display the image
            cv2.imshow('image', to_display)
            print(f"Post processing time: {time.time() - post_start_time}")
            cv2.waitKey(1)
    except (KeyboardInterrupt, metalcompute.error):
        if args.save_on_quit:
            pass
        else:
            raise
    # save the image
    if args.total_frames == 1:
        cv2.imwrite(f'../output/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', to_display)
    else:
        cv2.imwrite(f'../output/{args.movie_name}/frame_{args.frame_number:04d}.png', to_display)
