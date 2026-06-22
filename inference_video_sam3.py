import os
import torch
import numpy as np
from PIL import Image
from sam3.model.sam3matting_video_predictor import build_sam3matting_video_predictor
import argparse
import cv2
from iopath.common.file_io import g_pathmgr

checkpoint = "checkpoints/SAM2Matting-SAM3.pt"
device = "cuda"

def load_tracker_state_dict(checkpoint):
    with g_pathmgr.open(checkpoint, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    sd = ckpt["model"]
    out = {}
    for k, v in sd.items():
        if k.startswith("detector.backbone.vision_backbone."):
            out[k.removeprefix("detector.")] = v
        elif k.startswith("tracker."):
            out[k.removeprefix("tracker.")] = v
    return out


def build_predictor(checkpoint, device="cuda", compiled=False):
    sd = load_tracker_state_dict(checkpoint)
    predictor = build_sam3matting_video_predictor(checkpoint=None, device=device)
    missing, unexpected = predictor.load_state_dict(sd, strict=False)
    print("missing keys: ", missing)
    print("unexpected keys: ", unexpected)
    if compiled:
        trunk = predictor.backbone.vision_backbone.trunk
        trunk.forward = torch.compile(
            trunk.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        from sam3.model.utils.trt import replace_unknown_alpha_predictor_with_trt
        predictor = replace_unknown_alpha_predictor_with_trt(predictor)
    return predictor


def process_single_video(
    video_dir,
    first_mask_path,
    output_dir,
    ann_frame_idx=0,
    ann_obj_id=1,
    predictor=None,
    save_mp4=False,
):
    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if save_mp4:
        os.makedirs(output_dir, exist_ok=True)
        sample_img = Image.open(os.path.join(video_dir, frame_files[0]))
        w, h = sample_img.size

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        alpha_out = cv2.VideoWriter(os.path.join(output_dir, "pha.mp4"), fourcc, 25.0, (w, h))
        green_out = cv2.VideoWriter(os.path.join(output_dir, "fgr.mp4"), fourcc, 25.0, (w, h))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    m = Image.open(first_mask_path).convert("L")
    m = np.array(m).astype(np.float32) / 255.0
    m = (m > 0.005).astype(np.float32) * 20 - 10
    m = torch.from_numpy(m)[None, None]

    m = torch.nn.functional.interpolate(
        m,
        size=(288, 288),
        mode="bilinear",
        align_corners=False,
    )

    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        mask=m.to(device),
    )

    for out_frame_idx, _, _, alpha, _ in predictor.propagate_in_video(inference_state):
        if save_mp4:
            frame_name = frame_files[out_frame_idx]
            alpha_2d = np.asarray(alpha).squeeze().clip(0, 1)

            alpha_img = (alpha_2d * 255).astype(np.uint8)
            alpha_img = cv2.cvtColor(alpha_img, cv2.COLOR_GRAY2BGR)
            alpha_out.write(alpha_img)

            orig_img = np.array(Image.open(os.path.join(video_dir, frame_name)).convert("RGB"))
            alpha_expand = alpha_2d[..., None]
            green_bg = np.array([0, 255, 0], dtype=np.uint8)
            green_img = (orig_img * alpha_expand + green_bg * (1 - alpha_expand)).astype(np.uint8)
            green_img = cv2.cvtColor(green_img, cv2.COLOR_RGB2BGR)
            green_out.write(green_img)

    if save_mp4:
        alpha_out.release()
        green_out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video matting")
    parser.add_argument("--save_mp4", action="store_false", help="save mp4")
    parser.add_argument("--compiled", action="store_true", help="compile image encoder and alpha predictor")
    args = parser.parse_args()

    single_video_dir = "demo/video/frames"
    first_mask_file = "demo/video/mask.png"
    output_root = "output_video"

    predictor = build_predictor(
        checkpoint, device=device, compiled=args.compiled
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        process_single_video(
            video_dir=single_video_dir,
            first_mask_path=first_mask_file,
            output_dir=output_root,
            ann_frame_idx=0,
            ann_obj_id=1,
            predictor=predictor,
            save_mp4=args.save_mp4,
        )