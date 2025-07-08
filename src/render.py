import metalcompute
import cv2
import numpy as np
import time
from renderer import Renderer
from datetime import datetime
import argparse
from scene import Scene, create_scene_from_preset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--save-on-quit", action="store_true")
    parser.add_argument("--scene", type=str, default="teapots")
    args = parser.parse_args()

    scene: Scene = create_scene_from_preset(
        args.scene,
        pixel_width=args.width,
        pixel_height=args.height,
    )
    renderer: Renderer = Renderer(scene)

    to_display = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    start_time = time.time()
    for i in range(args.samples):
        try:
            renderer.run_sample()
            print(f"Sample {i}/{args.samples} completed")
            to_display = renderer.image.copy()
            cv2.imshow("image", to_display)
            cv2.waitKey(1)

        except (KeyboardInterrupt, metalcompute.error):
            if args.save_on_quit:
                pass
            else:
                raise

    print(f"Rendering took {time.time() - start_time:.2f} seconds")

    cv2.imwrite(
        f'../output/default/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
        to_display,
    )
