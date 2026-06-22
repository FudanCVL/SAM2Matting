import logging
from typing import Optional, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

class SAM3MattingImagePredictor:
    IMAGE_MEAN = (0.5, 0.5, 0.5)
    IMAGE_STD  = (0.5, 0.5, 0.5)

    def __init__(self, model, image_size: int = 1008):
        self.model = model
        self.image_size = image_size
        self._device = next(model.parameters()).device

        self._transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(self.IMAGE_MEAN), std=list(self.IMAGE_STD)),
        ])

        self._orig_hw: Optional[Tuple[int, int]] = None
        self._features = None
        self._vision_feats = None
        self._vision_pos_embeds = None
        self._feat_sizes = None
        self._img_tensor: Optional[torch.Tensor] = None
        
    @torch.inference_mode()
    def set_image(self, image: PIL.Image.Image) -> np.ndarray:
        if isinstance(image, PIL.Image.Image):
            self._orig_hw = (image.height, image.width)
            img_np = np.array(image.convert("RGB"))
        else:
            raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

        img_t = v2.functional.to_image(image).to(self._device)
        img_t = self._transform(img_t).unsqueeze(0)
        self._img_tensor = img_t

        backbone_out = self.model.forward_image(img_t)
        (
            _,
            self._vision_feats,
            self._vision_pos_embeds,
            self._feat_sizes,
        ) = self.model._prepare_backbone_features(backbone_out)
        return img_np

    @torch.inference_mode()
    def predict(
        self,
        img: np.ndarray,
        raw_mask: torch.Tensor,
        mask_input: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._img_tensor is None:
            raise RuntimeError("Call set_image() before predict().")
        device = self._device
        orig_h, orig_w = self._orig_hw

        mask_input = mask_input.to(device=device, dtype=torch.float32)
        if mask_input.dim() == 2:
            mask_input = mask_input.unsqueeze(0).unsqueeze(0)
        elif mask_input.dim() == 3:
            mask_input = mask_input.unsqueeze(0)
        assert mask_input.dim() == 4, f"mask_input must be 4-D, got {mask_input.shape}"

        low_res_mask = F.interpolate(
            mask_input,
            size=(288, 288),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        current_vision_feats   = self._vision_feats
        current_vision_pos_embeds = self._vision_pos_embeds
        feat_sizes             = self._feat_sizes

        img_t = self._img_tensor
        alpha, alpha_multiscale, unknown_region_upscaled = self.model._forward_alpha_heads(
            input=img_t.squeeze(0),        
            backbone_features=None,         
            point_inputs=None,
            mask_inputs=(low_res_mask > 0.0).float(),
            unknown_region_inputs=None,
            high_res_features=self._build_high_res_features(
                current_vision_feats, feat_sizes
            ),
            image=None,
            trimap_input=None,
        )
        
        def _to_numpy_orig(t: torch.Tensor) -> np.ndarray:
            t = F.interpolate(
                t.float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            return t.squeeze(1).cpu().numpy()
        
        if isinstance(raw_mask, torch.Tensor):
            raw_mask_np = raw_mask.float().cpu().numpy()
        else:
            raw_mask_np = np.asarray(raw_mask, dtype=np.float32)
        if raw_mask_np.ndim == 2:
            raw_mask_np = raw_mask_np[np.newaxis]

        alpha_np     = _to_numpy_orig(alpha)  
        # unknown_np   = _to_numpy_orig(unknown_region_upscaled)

        # if isinstance(alpha_multiscale, list):
        #     alpha_upscaled_t = alpha_multiscale[-1]
        # else:
        #     alpha_upscaled_t = alpha_multiscale
        # alpha_upscaled_np = _to_numpy_orig(alpha_upscaled_t)

        return raw_mask_np, alpha_np, None, None
        # return raw_mask_np, alpha_np, alpha_upscaled_np, unknown_np


    def _build_high_res_features(self, vision_feats, feat_sizes):
        """
        Converts the list of (HW, B, C) vision_feats into the list of
        [B, C, H, W] tensors expected by _forward_alpha_heads.
        vision_feats[0] is the finest scale, vision_feats[-1] the coarsest.
        """
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(vision_feats, feat_sizes)
        ]
        return high_res_features