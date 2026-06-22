import os
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2matting_video_predictor
import argparse
import cv2

variant = "sam2.1tiny"

if variant == "sam2.1tiny":
    checkpoint = "checkpoints/SAM2Matting-SAM2.1Tiny.pt"
    model_cfg = "configs/sam2matting-sam2.1tiny.yaml"
elif variant == "sam2.1base+":
    checkpoint = "checkpoints/SAM2Matting-SAM2.1Base+.pt"
    model_cfg = "configs/sam2matting-sam2.1base+.yaml"
else:
    raise ValueError(f"Invalid variant: {variant}")

device = "cuda"

def build_predictor(model_cfg, checkpoint, device="cuda", compiled=False):
    hydra_overrides_extra = []
    if compiled:
        hydra_overrides_extra.append("++model.compile_image_encoder=True")
    predictor = build_sam2matting_video_predictor(
        model_cfg, checkpoint, device=device,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    if compiled:
        from sam2.utils.trt import replace_unknown_alpha_predictor_with_trt
        predictor = replace_unknown_alpha_predictor_with_trt(predictor)
    return predictor

def process_single_video(
    video_dir,
    output_dir,
    ann_frame_idx=0,
    ann_obj_id=1,
    predictor=None,
    save_mp4=False,
    prompt_type="point",
    point=None,
    bbox=None,
):
    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if save_mp4:
        os.makedirs(output_dir, exist_ok=True)
        sample_img = Image.open(os.path.join(video_dir, frame_files[0]))
        w, h = sample_img.size
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        alpha_out = cv2.VideoWriter(os.path.join(output_dir, "pha.mp4"), fourcc, 25.0, (w, h))
        green_out = cv2.VideoWriter(os.path.join(output_dir, "fgr.mp4"), fourcc, 25.0, (w, h))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    if prompt_type == "point":
        points = np.array([point], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
    elif prompt_type == "box":
        box = np.array(bbox, dtype=np.float32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    for out_frame_idx, _, out_mask_logits, alpha, _ in predictor.propagate_in_video(inference_state):
        if save_mp4:
            frame_name = frame_files[out_frame_idx]
            alpha_2d = alpha.detach().cpu().squeeze().float().numpy().clip(0, 1)
            
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
    parser.add_argument("--prompt_type", choices=["point", "box"], default="point")
    parser.add_argument("--point", type=float, nargs=2, default=[486, 301])
    parser.add_argument("--bbox", type=float, nargs=4, default=[412, 109, 717, 449])
    args = parser.parse_args()
    
    single_video_dir = "demo/video/frames"
    output_root = "output_video"
    
    predictor = build_predictor(
        model_cfg, checkpoint, device=device, compiled=args.compiled
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        process_single_video(
            video_dir=single_video_dir,
            output_dir=output_root,
            ann_frame_idx=0,
            ann_obj_id=1,
            predictor=predictor,
            save_mp4=args.save_mp4,
            prompt_type=args.prompt_type,
            point=args.point,
            bbox=args.bbox,
        )