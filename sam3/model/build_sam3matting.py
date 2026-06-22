from __future__ import annotations

from typing import Optional

import torch
from iopath.common.file_io import g_pathmgr

from sam3.model_builder import (
    _create_tracker_maskmem_backbone,
    _create_tracker_transformer,
    _create_vision_backbone,
)
from sam3.model.vl_combiner import SAM3VLBackbone

from sam3.model.sam3_tracker_base import Sam3TrackerBase as SAM3MattingModel

from sam3.model.sam3matting_image_predictor import SAM3MattingImagePredictor


def build_sam3matting(
    # model_cfg: Optional[str] = None,
    checkpoint: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    strict: bool = False,
) -> SAM3MattingImagePredictor:
    
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer      = _create_tracker_transformer()
    vision_backbone  = _create_vision_backbone()
    backbone         = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)

    model = SAM3MattingModel(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        max_cond_frames_in_attn=4,
        non_overlap_masks_for_mem_enc=False,
        forward_backbone_per_frame_for_eval=True,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    )

    if checkpoint is not None:
        with g_pathmgr.open(checkpoint, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        missing, unexpected = model.load_state_dict(ckpt, strict=strict)
        print("missing keys: ", missing)
        print("unexpected keys: ", unexpected)
    
    model.to(device=device)
    model.eval()
    return model