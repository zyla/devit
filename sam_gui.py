# requirements:
# pip install opencv-python
# pip install git+https://github.com/facebookresearch/segment-anything.git
# pip install 'numpy<2'
# 
# download model checkpoint from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints


import os
import cv2
import torch
import numpy as np
import fire
from segment_anything import sam_model_registry, SamPredictor


def main(
    image_dir="images",
    model_type=None,
    checkpoint=None,
    device="cuda",
):
    """
    Interactive mask generator with SAM + OpenCV.

    Args:
        image_dir: Root directory with images (can have nested dirs).
        model_type: Model type: vit_b, vit_l, or vit_h. (required if checkpoint is given)
        checkpoint: Path to SAM checkpoint (required if model_type is given)
        device: Device to run on: cuda, cpu, or mps (Apple Silicon).
        output_suffix: Suffix for saved masks.
    """
    if checkpoint is None:
        raise ValueError("--checkpoint is required")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
        )

    if model_type is None:
        model_type = guess_model_type(checkpoint)
        print(f"[auto] Detected model_type={model_type} from checkpoint filename")

    # --- load SAM ---
    print(f"Loading SAM model {model_type} from {checkpoint} on {device}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device) # type: ignore
    sam.eval()
    predictor = SamPredictor(sam)

    # --- recursively gather images ---
    image_files = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, f))
    image_files.sort()

    if not image_files:
        print("No images found!")
        return

    print(f"Found {len(image_files)} images under {image_dir}")

    # --- globals for mouse callback ---
    click_point: dict = {"pt": None, "mask": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point["pt"] = (x, y)
            print(f"Clicked at: {click_point['pt']}")

            # run SAM prediction
            input_point = np.array([click_point["pt"]])
            input_label = np.array([1])
            masks, scores, next_masks = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
         # second iteration, doesn't give super results
            # best_idx = np.argmax(scores)
            # masks, scores, _ = predictor.predict(
            #     point_coords=input_point,
            #     point_labels=input_label,
            #     mask_input=np.expand_dims(next_masks[best_idx], axis=0),
            #     multimask_output=True,
            # )
            # find mask with highest score
            best_idx = np.argmax(scores)
            prev_mask = click_point["mask"]
            if prev_mask is not None:
                click_point["mask"] = prev_mask + masks[best_idx].astype(np.uint8) * 255
            else:
                click_point["mask"] = masks[best_idx].astype(np.uint8) * 255

    def is_mask(image_path: str):
        return image_path.endswith(".mask.jpg") or image_path.endswith(".mask.png")

    # --- main loop ---
    for idx, image_path in enumerate(x for x in image_files if not is_mask(x)):
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}.mask{ext}"
        if os.path.exists(out_path):
            print(f"[{idx+1}/{len(image_files)}] Skipping (mask exists): {image_path}")
            continue

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.setWindowTitle("image", "SAMGUI")
        cv2.setMouseCallback("image", on_mouse)
        cv2.resizeWindow("image", 1333, 800)

        click_point["pt"] = None
        click_point["mask"] = None

        while True:
            disp = image_bgr.copy()

            if click_point["mask"] is not None:
                # overlay mask with transparency
                color = np.array([0, 255, 0], dtype=np.uint8)  # green
                alpha = 0.5
                disp[click_point["mask"] > 0] = disp[click_point["mask"] > 0] * (1 - alpha) + color * alpha

                # draw clicked point
                if click_point["pt"] is not None:
                    cv2.circle(disp, click_point["pt"], 5, (0, 0, 255), -1)

            cv2.setWindowTitle("image", f"SAMGUI: {image_path}")

            cv2.imshow("image", disp)
            key = cv2.waitKey(50) & 0xFF

            if click_point["mask"] is not None and key == ord("n"):
                cv2.imwrite(out_path, click_point["mask"])
                print(f"[{idx+1}/{len(image_files)}] Saved mask → {out_path}")
                break

            if click_point["mask"] is not None and key == ord("r"):
                click_point["mask"] = None

            if key == ord("q") or cv2.getWindowProperty("image",cv2.WND_PROP_VISIBLE) == 0:
                print("Exiting.")
                return

    cv2.destroyAllWindows()

def guess_model_type(checkpoint_path: str) -> str:
    """Heuristic: infer model type from checkpoint filename."""
    name = os.path.basename(checkpoint_path).lower()
    if "vit_b" in name:
        return "vit_b"
    elif "vit_l" in name:
        return "vit_l"
    elif "vit_h" in name:
        return "vit_h"
    else:
        raise ValueError(
            f"Could not determine model type from checkpoint filename: {checkpoint_path}. "
            "Please pass --model_type explicitly."
        )


if __name__ == "__main__":
    fire.Fire(main)

