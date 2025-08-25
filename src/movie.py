import argparse
import cv2
import os
import shutil
import numpy as np
import time
from renderer import Renderer
from scene import Scene, create_scene_from_preset_with_params

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--scene", type=str, default="teapots")
    parser.add_argument("--movie-name", type=str, default="test-movie")
    parser.add_argument("--movie-frames", type=int, default=120)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    if args.start_frame == 0:
        if os.path.exists(f"../output/{args.movie_name}"):
            shutil.rmtree(f"../output/{args.movie_name}")
        os.makedirs(f"../output/{args.movie_name}")

    to_display = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    for f in range(args.start_frame, args.movie_frames):
        frame_start_time = time.time()
        scene: Scene = create_scene_from_preset_with_params(
            args.scene,
            pixel_width=args.width,
            pixel_height=args.height,
            frame_idx=f,
            total_frames=args.movie_frames,
        )
        renderer: Renderer = Renderer(scene)
        for i in range(args.samples):
            start_sample_time = time.time()
            renderer.run_sample()
            to_display = renderer.image.copy()
            cv2.imshow("image", to_display)
            cv2.waitKey(1)
            print(f"Sample {i} time: {time.time() - start_sample_time}")

        cv2.imwrite(
            f"../output/{args.movie_name}/frame_{f:04d}.png",
            to_display,
        )

        del renderer
        del scene

        print(f"Frame {f} time: {time.time() - frame_start_time}")
