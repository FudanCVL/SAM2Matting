# sam3/model/sam3_tracker_train.py
"""
SAM3 Tracker for VOS training (similar to SAM2Train).
"""

import logging
import numpy as np
import torch

from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.sam3_tracker_utils import get_next_point, sample_box_points
# from sam3.train.data.collator import BatchedDatapoint
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.utils.misc import concat_points

logger = logging.getLogger(__name__)


class Sam3TrackerTrain(Sam3TrackerBase):
    def __init__(
        self,
        # Point/mask input sampling probabilities
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # Number of frames to add correction points
        num_frames_to_correct_for_train=1,
        num_frames_to_correct_for_eval=1,
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # Initial conditioning frames
        num_init_cond_frames_for_train=1,
        num_init_cond_frames_for_eval=1,
        rand_init_cond_frames_for_train=True,
        rand_init_cond_frames_for_eval=False,
        add_all_frames_to_correct_as_cond=False,
        # Correction point sampling
        num_correction_pt_per_frame=7,
        pt_sampling_for_eval="center",
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # Whether to freeze image encoder
        freeze_image_encoder=False,
        prob_to_dropout_spatial_mem=0.0,
        teacher_force_obj_scores_for_mem=False,
        use_memory_selection=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Point sampler settings
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        
        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        
        # Random number generator with fixed seed
        self.rng = np.random.default_rng(seed=42)
        self.prob_to_dropout_spatial_mem = prob_to_dropout_spatial_mem
        self.teacher_force_obj_scores_for_mem = teacher_force_obj_scores_for_mem
        
        self.use_memory_selection = use_memory_selection
        
        if freeze_image_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

        ###############################################
        # SAM3Matting
        for name, p in self.named_parameters():
            if 'alpha' not in name and 'unknown' not in name:
                p.requires_grad = False
                
        # trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        # for name in trainable:
        #     print(f"  {name}")
        ###############################################


    def forward(self, input: BatchedDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            backbone_out = self.forward_image(input.img_batch)
        else:
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)
        return previous_stages_out

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts.
        """
        gt_masks_per_frame = {}
        for stage_id in range(len(input.find_inputs)):
            segments = input.find_targets[stage_id].segments
            gt_masks_per_frame[stage_id] = segments.unsqueeze(1)
        
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = len(input.find_inputs)
        backbone_out["num_frames"] = num_frames
        
        # Decide whether to use point or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        
        if num_frames == 1:
            # Special case: single image (force point input)
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        
        assert num_init_cond_frames >= 1
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        
        #####################################
        # SAM3Matting
        use_pt_input = False
        #####################################
        
        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]
        else:
            if rand_init_cond_frames:
                num_init_cond_frames = self.rng.integers(1, num_init_cond_frames, endpoint=True)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        backbone_out["use_pt_input"] = use_pt_input
        
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}
        backbone_out["point_inputs_per_frame"] = {}
        
        assert len(init_cond_frames) == 1
        for t in init_cond_frames:
            if not use_pt_input:
                # Use mask inputs directly
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                assert False
                # Sample points from GT masks
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(gt_masks_per_frame[t])
                else:
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method="uniform" if self.training else self.pt_sampling_for_eval,
                    )
                
                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs
        
        # Sample frames for correction clicks
        if not use_pt_input:
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            if rand_frames_to_correct and num_frames_to_correct > num_init_cond_frames:
                num_frames_to_correct = self.rng.integers(
                    num_init_cond_frames, num_frames_to_correct, endpoint=True
                )
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt
        return backbone_out

    def back_convert(self, targets):
        """
        Convert targets for loss computation.
        For VOS, we return targets as-is since Sam3LossVOS
        expects BatchedFindTarget format.
        """
        return targets