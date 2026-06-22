import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.utils.misc import mask_to_box

def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
       
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs

def get_1d_sine_pe(pos_inds, dim, temperature=10000):
       
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed

def get_activation_fn(activation):
                                                      
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  
    noise_bound: int = 20,  
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
       
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  

    box_coords = box_coords.reshape(-1, 2, 2)  
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels

def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
       
    if pred_masks is None:  
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    fp_masks = ~gt_masks & pred_masks

    fn_masks = gt_masks & ~pred_masks

    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)

    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels

def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
       
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    fp_masks = ~gt_masks & pred_masks

    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  
        points[b, 0, 1] = pt_idx // W_im  
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels

def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")
