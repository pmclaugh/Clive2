import metalcompute
import cv2

from renderer import Renderer
from datetime import datetime
import argparse
from load import *
from scene import Scene, create_scene_from_preset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--save-on-quit", action="store_true")
    parser.add_argument("--scene", type=str, default="teapots")
    args = parser.parse_args()

    device = metalcompute.Device()

    scene: Scene = create_scene_from_preset(
        args.scene,
        pixel_width=args.width,
        pixel_height=args.height,
        metal_device=device,
    )
    renderer: Renderer = Renderer(scene)

    to_display = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    for i in range(args.samples):
        try:
            start_sample_time = time.time()
            renderer.run_sample()
            to_display = renderer.current_image.copy()
            cv2.imshow("image", to_display)
            cv2.waitKey(1)
            print(f"Whole sample {i} time: {time.time() - start_sample_time}")

        except (KeyboardInterrupt, metalcompute.error):
            if args.save_on_quit:
                pass
            else:
                raise

    cv2.imwrite(
        f'../output/default/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
        to_display,
    )
