import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from iopath.common.file_io import g_pathmgr
from sam3.model_builder import build_sam3_video_predictor
from sam3.model.sam3matting_video_predictor import build_sam3matting_video_predictor

CHECKPOINT = "checkpoints/SAM2Matting-SAM3.pt"
DEVICE = "cuda"

def load_tracker_state_dict(checkpoint):
    with g_pathmgr.open(checkpoint, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    if not any(k.startswith("tracker.") for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith("detector.backbone.vision_backbone."):
            out[k.removeprefix("detector.")] = v
        elif k.startswith("tracker."):
            out[k.removeprefix("tracker.")] = v
    return out


def build_language_predictor(checkpoint, compiled=False):
    predictor = build_sam3_video_predictor(
        gpus_to_use=[0],
        checkpoint_path=checkpoint,
        strict_state_dict_loading=False,
        bpe_path=BPE_PATH,
    )
    if compiled:
        trunk = predictor.model.detector.backbone.vision_backbone.trunk
        trunk.forward = torch.compile(
            trunk.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        from sam3.model.utils.trt import replace_unknown_alpha_predictor_with_trt
        predictor.model.tracker = replace_unknown_alpha_predictor_with_trt(
            predictor.model.tracker
        )
    return predictor


def build_tracker_predictor(checkpoint, device="cuda", compiled=False):
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


def write_frame(alpha_out, green_out, video_dir, frame_files, idx, alpha_2d):
    alpha_img = (alpha_2d * 255).astype(np.uint8)
    alpha_img = cv2.cvtColor(alpha_img, cv2.COLOR_GRAY2BGR)
    alpha_out.write(alpha_img)

    orig = np.array(Image.open(os.path.join(video_dir, frame_files[idx])).convert("RGB"))
    green_bg = np.array([0, 255, 0], dtype=np.uint8)
    green_img = (
        orig * alpha_2d[..., None] + green_bg * (1 - alpha_2d[..., None])
    ).astype(np.uint8)
    green_img = cv2.cvtColor(green_img, cv2.COLOR_RGB2BGR)
    green_out.write(green_img)


def pick_mask(outputs):
    masks = outputs["out_binary_masks"]
    if len(masks) == 0:
        raise RuntimeError("no mask from propagate")
    if torch.is_tensor(masks):
        m = masks[0]
    else:
        m = torch.as_tensor(masks[0])
    if m.dim() == 3:
        m = m[0]
    return m


def ensure_frame_cache(model, state, frame_idx):
    if frame_idx not in state["feature_cache"]:
        model._prepare_backbone_feats(state, frame_idx, reverse=False)


def compute_alpha(model, state, frame_idx, mask_hw):
    ensure_frame_cache(model, state, frame_idx)
    image, cache = state["feature_cache"][frame_idx]
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(DEVICE)
    fpn = cache["tracker_backbone_out"]["backbone_fpn"]
    high_res_features = [x for x in fpn]
    mask = torch.as_tensor(mask_hw > 0, device=DEVICE, dtype=torch.float32)
    mask_288 = F.interpolate(
        mask[None, None],
        size=(288, 288),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    binary_mask_288 = (mask_288 > 0.0).float()
    alpha, _, _ = model.tracker._forward_alpha_heads(
        input=image,
        backbone_features=None,
        point_inputs=None,
        mask_inputs=binary_mask_288,
        unknown_region_inputs=None,
        high_res_features=high_res_features,
        image=None,
        trimap_input=None,
    )
    video_h = state["orig_height"]
    video_w = state["orig_width"]
    alpha_up = F.interpolate(
        alpha.float(),
        size=(video_h, video_w),
        mode="bilinear",
        align_corners=False,
    )
    return alpha_up.squeeze().detach().float().cpu().numpy()


def process_language(
    video_dir,
    output_dir,
    predictor,
    save_mp4=True,
    fps=25.0,
    frame_idx=0,
    language=None,
):
    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not frame_files:
        raise RuntimeError(f"no frames in {video_dir}")

    alpha_out = None
    green_out = None
    if save_mp4:
        os.makedirs(output_dir, exist_ok=True)
        sample = Image.open(os.path.join(video_dir, frame_files[0]))
        w, h = sample.size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        alpha_out = cv2.VideoWriter(os.path.join(output_dir, "pha.mp4"), fourcc, fps, (w, h))
        green_out = cv2.VideoWriter(os.path.join(output_dir, "fgr.mp4"), fourcc, fps, (w, h))

    resp = predictor.handle_request(dict(
        type="start_session",
        resource_path=video_dir,
    ))
    session_id = resp["session_id"]

    predictor.model.add_prompt(
        inference_state=predictor._get_session(session_id)["state"],
        frame_idx=frame_idx,
        text_str=language,
    )

    seen = set()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for resp in predictor.handle_stream_request(dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="forward",
        )):
            idx = resp["frame_index"]
            if idx in seen:
                continue
            seen.add(idx)

            if save_mp4:
                mask = pick_mask(resp["outputs"])
                state = predictor._get_session(session_id)["state"]
                alpha_2d = compute_alpha(
                    predictor.model,
                    state,
                    idx,
                    mask.cpu().numpy(),
                ).clip(0, 1)
                write_frame(alpha_out, green_out, video_dir, frame_files, idx, alpha_2d)

    predictor.handle_request(dict(type="close_session", session_id=session_id))

    if save_mp4:
        alpha_out.release()
        green_out.release()


def process_tracker(
    video_dir,
    output_dir,
    predictor,
    save_mp4=True,
    fps=25.0,
    frame_idx=0,
    obj_id=1,
    prompt_type="point",
    point=None,
    bbox=None,
):
    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not frame_files:
        raise RuntimeError(f"no frames in {video_dir}")

    sample = Image.open(os.path.join(video_dir, frame_files[frame_idx]))
    w, h = sample.size
    
    alpha_out = None
    green_out = None
    if save_mp4:
        os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        alpha_out = cv2.VideoWriter(os.path.join(output_dir, "pha.mp4"), fourcc, fps, (w, h))
        green_out = cv2.VideoWriter(os.path.join(output_dir, "fgr.mp4"), fourcc, fps, (w, h))

    state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(state)

    if prompt_type == "point":
        pts = np.array(args.point, dtype=np.float32).reshape(-1, 2)
        points = pts / np.array([w, h], dtype=np.float32)
        labels = np.ones(len(pts), dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        
    elif prompt_type == "box":
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=np.array([bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h], dtype=np.float32),
        )
    else:
        raise ValueError(f"invalid prompt_type: {prompt_type}")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for out_frame_idx, _, _, alpha, _ in predictor.propagate_in_video(state):
            if save_mp4:
                alpha_2d = np.asarray(alpha).squeeze().astype(np.float32).clip(0, 1)
                write_frame(alpha_out, green_out, video_dir, frame_files, out_frame_idx, alpha_2d)

    if save_mp4:
        alpha_out.release()
        green_out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="demo/video/frames")
    parser.add_argument("--output_dir", type=str, default="output_video")
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--save_mp4", action="store_false")
    parser.add_argument("--compiled", action="store_true")
    
    parser.add_argument("--prompt_type", choices=["language", "point", "box"], default="language")
    parser.add_argument("--language", type=str, default="girl")
    parser.add_argument("--point", type=float, nargs=2, default=[457, 155, 484, 369]) # multiple points, single point is also supported
    parser.add_argument("--bbox", type=float, nargs=4, default=[412, 109, 717, 449])
    args = parser.parse_args()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if args.prompt_type == "language": 
            # detector needed
            BPE_PATH = "sam3/bpe_simple_vocab_16e6.txt.gz"
            predictor = build_language_predictor(CHECKPOINT, compiled=args.compiled)
            process_language(
                video_dir=args.video_dir,
                output_dir=args.output_dir,
                predictor=predictor,
                save_mp4=args.save_mp4,
                fps=args.fps,
                frame_idx=args.frame_idx,
                language=args.language,
            )
        else: 
            # visual prompts (no detector needed)
            predictor = build_tracker_predictor(CHECKPOINT, device=DEVICE, compiled=args.compiled)
            process_tracker(
                video_dir=args.video_dir,
                output_dir=args.output_dir,
                predictor=predictor,
                save_mp4=args.save_mp4,
                fps=args.fps,
                frame_idx=args.frame_idx,
                prompt_type=args.prompt_type,
                point=args.point,
                bbox=args.bbox,
            )