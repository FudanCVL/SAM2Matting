import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image
from sam2.modeling.sam2matting_base import SAM2MattingBase

from sam2.utils.transforms import SAM2Transforms
import torch.nn.functional as F

class SAM2MattingImagePredictor:
    def __init__(
        self,
        sam_model: SAM2MattingBase,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
        self.mask_threshold = mask_threshold
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2MattingImagePredictor":
        from sam2.build_sam import build_sam2matting_hf
        sam2matting_model = build_sam2matting_hf(model_id, **kwargs)
        return cls(sam2matting_model, **kwargs)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        self.reset_predictor()

        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self.all_features=feats
        self._is_image_set = True
        logging.info("Image embeddings computed.")
        return input_image
    
    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])

        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self.all_features=feats
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
           
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        img=None,
        raw_mask=None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        masks = raw_mask.unsqueeze(0).unsqueeze(0).to(self.device)
        mask_inputs = (mask_input>0.).float().to(self.device)

        hight_features=self.all_features

        alpha, alpha_upscaled, unknown_region = self.model._forward_alpha_heads(mask_inputs=mask_inputs, high_res_features=hight_features, image=img)

        alpha=F.interpolate(
            alpha, size=(masks.shape[-2], masks.shape[-1]), mode="bilinear", align_corners=False
        )
        alpha = alpha.squeeze(0).float().detach().cpu().numpy()

        # unknown_region = unknown_region.sigmoid().squeeze(0).float().detach().cpu().numpy()
        
        # masks = masks.squeeze(0).float().detach().cpu().numpy()

        return None, alpha, None
        # return masks, alpha, unknown_region

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
           
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)

            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
