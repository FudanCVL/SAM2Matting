from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
from tqdm.auto import tqdm
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor

class SAM3MattingVideoPredictor(Sam3TrackerPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @torch.inference_mode()
    def reset_state(self, inference_state):
        self.clear_all_points_in_video(inference_state)

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx: int,
        obj_id: int,
        mask: torch.Tensor,
        add_mask_to_memory: bool = False,
    ):
        m = mask
        while m.dim() > 2 and m.shape[0] == 1:
            m = m.squeeze(0)
        if m.dim() != 2:
            raise ValueError(
                f"Cannot reduce mask to 2-D; original shape = {mask.shape}"
            )

        binary_mask = (m > 0.0).to(dtype=torch.bool, device=m.device)

        frame_idx_out, obj_ids, _low_res, video_res_masks = super().add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=binary_mask,
            add_mask_to_memory=add_mask_to_memory,
        )
        return frame_idx_out, obj_ids, video_res_masks

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
        tqdm_disable: bool = False,
        obj_ids=None,
        run_mem_encoder: bool = True,
        propagate_preflight: bool = True
    ):
        if propagate_preflight:
            self.propagate_in_video_preflight(inference_state)

        output_dict           = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]

        obj_ids    = inference_state["obj_ids"]
        batch_size = self._get_obj_num(inference_state)

        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError(
                "No prompts found. Use add_new_mask() or add_new_points_or_box() to add prompts first."
            )

        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        processing_order = self._get_processing_order(
            inference_state, start_frame_idx, max_frame_num_to_track, reverse
        )

        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        device  = inference_state["device"]

        for frame_idx in tqdm(
            processing_order,
            desc="video matting",
            disable=tqdm_disable,
        ):
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks  = current_out["pred_masks"]
                if clear_non_cond_mem:
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks  = current_out["pred_masks"]

            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                )
                output_dict[storage_key][frame_idx] = current_out

            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )

            (
                image,                 
                _,
                current_vision_feats,   
                _,
                feat_sizes,
            ) = self._get_image_feature(inference_state, frame_idx, batch_size)
            
            high_res_features = [
                x.permute(1, 2, 0)
                 .view(x.size(1), x.size(2), *s)
                 .detach()
                for x, s in zip(current_vision_feats, feat_sizes)
            ]

            pred_masks_gpu = pred_masks.to(device, non_blocking=True)
            mask_288 = F.interpolate(
                pred_masks_gpu.float(),
                size=(288, 288),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            binary_mask_288 = (mask_288 > 0.0).float()   
            
            alpha, _alpha_ms, unknown = self._forward_alpha_heads(
                input=image,                
                backbone_features=None,
                point_inputs=None,
                mask_inputs=binary_mask_288, 
                unknown_region_inputs=None,
                high_res_features=high_res_features,
                image=None,
                trimap_input=None,
            )
            
            alpha_up = F.interpolate(
                alpha.float(),
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )

            # unknown_up = F.interpolate(
            #     unknown.float(),
            #     size=(video_H, video_W),
            #     mode="bilinear",
            #     align_corners=False,
            # ) 

            alpha_np   = alpha_up.squeeze(1).cpu().numpy().astype(np.float32)
            # unknown_np = unknown_up.squeeze(1).cpu().numpy().astype(np.float32)

            yield (
                frame_idx,
                obj_ids,
                video_res_masks,  
                alpha_np, 
                # unknown_np,
                None,
            )


def build_sam3matting_video_predictor(
    # model_cfg: Optional[str] = None,
    checkpoint: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    strict: bool = False,
    **predictor_kwargs,
) -> SAM3MattingVideoPredictor:

    from sam3.model_builder import (
        _create_tracker_maskmem_backbone,
        _create_tracker_transformer,
        _create_vision_backbone,
    )
    from sam3.model.vl_combiner import SAM3VLBackbone

    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer      = _create_tracker_transformer()
    vision_backbone  = _create_vision_backbone()
    backbone         = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)

    predictor = SAM3MattingVideoPredictor(
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
        **predictor_kwargs,
    )

    if checkpoint is not None:
        with g_pathmgr.open(checkpoint, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        missing, unexpected = predictor.load_state_dict(ckpt, strict=strict)
        print("missing keys: ", missing)
        print("unexpected keys: ", unexpected)

    predictor.to(device=device)
    predictor.eval()
    return predictor