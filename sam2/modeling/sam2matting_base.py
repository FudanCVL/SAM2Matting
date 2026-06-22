import torch
import torch.nn.functional as F
import torch.nn as nn
from sam2.modeling.sam2_base import SAM2Base

NO_OBJ_SCORE = -1024.0

class SAM2MattingBase(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  
        image_size=512,
        backbone_stride=16,  
        sigmoid_scale_for_mem_enc=1.0,  
        sigmoid_bias_for_mem_enc=0.0,  
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        alpha_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            num_maskmem=num_maskmem,
            image_size=image_size,
            backbone_stride=backbone_stride,
            sigmoid_scale_for_mem_enc=sigmoid_scale_for_mem_enc,
            sigmoid_bias_for_mem_enc=sigmoid_bias_for_mem_enc,
            binarize_mask_from_pts_for_mem_enc=binarize_mask_from_pts_for_mem_enc,
            use_mask_input_as_output_without_sam=use_mask_input_as_output_without_sam,
            max_cond_frames_in_attn=max_cond_frames_in_attn,
            directly_add_no_mem_embed=directly_add_no_mem_embed,
            use_high_res_features_in_sam=use_high_res_features_in_sam,
            multimask_output_in_sam=multimask_output_in_sam,
            multimask_min_pt_num=multimask_min_pt_num,
            multimask_max_pt_num=multimask_max_pt_num,
            multimask_output_for_tracking=multimask_output_for_tracking,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            memory_temporal_stride_for_eval=memory_temporal_stride_for_eval,
            non_overlap_masks_for_mem_enc=non_overlap_masks_for_mem_enc,
            use_obj_ptrs_in_encoder=use_obj_ptrs_in_encoder,
            max_obj_ptrs_in_encoder=max_obj_ptrs_in_encoder,
            add_tpos_enc_to_obj_ptrs=add_tpos_enc_to_obj_ptrs,
            proj_tpos_enc_in_obj_ptrs=proj_tpos_enc_in_obj_ptrs,
            use_signed_tpos_enc_to_obj_ptrs=use_signed_tpos_enc_to_obj_ptrs,
            only_obj_ptrs_in_the_past_for_eval=only_obj_ptrs_in_the_past_for_eval,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            fixed_no_obj_ptr=fixed_no_obj_ptr,
            soft_no_obj_ptr=soft_no_obj_ptr,
            use_mlp_for_obj_ptr_proj=use_mlp_for_obj_ptr_proj,
            no_obj_embed_spatial=no_obj_embed_spatial,
            sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
            compile_image_encoder=compile_image_encoder,
            **kwargs,
        )

        self._build_unknown_region_predictor()
        self._build_unknown_alpha_predictor()

    def _build_unknown_region_predictor(self):
        self.unknown_region_predictor = nn.ModuleDict({
            "scale_64": nn.Sequential(
                nn.Conv2d(32+1+3, 128, kernel_size=3, padding=1),  
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            "scale_128": nn.Sequential(
                nn.Conv2d(64+1+3, 128, kernel_size=3, padding=1),  
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            "scale_256": nn.Sequential(
                nn.Conv2d(256+1+3, 256, kernel_size=3, padding=1),  
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
        })

        self.unknown_fusion = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),

        )

    def _build_unknown_alpha_predictor(self):
        self.unknown_alpha_predictor = nn.ModuleDict({
            "scale_64": nn.Sequential(
                nn.ConvTranspose2d(256+1+3, 512, kernel_size=4, padding=1,stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            "scale_128": nn.Sequential(
                nn.ConvTranspose2d(64+3+1+1, 256, kernel_size=4, padding=1,stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            "scale_256": nn.Sequential(
                nn.ConvTranspose2d(32+1+3+1, 256, kernel_size=4, padding=1,stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
        })

        self.alpha_pred1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.alpha_pred2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.alpha_pred3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _detect_unknown_region(self, high_res_features, mask_inputs, img):
        # ROI Detection
        
        features_4 = high_res_features[0]  
        features_8 = high_res_features[1]  
        features_16 = high_res_features[2] if len(high_res_features) > 2 else high_res_features[1]  

        mask_scale_2 = F.interpolate(mask_inputs, size=features_4.shape[2:4], mode="bilinear", align_corners=False)  
        mask_scale_4 = F.interpolate(mask_inputs, size=features_8.shape[2:4], mode="bilinear", align_corners=False)  
        mask_scale_8 = F.interpolate(mask_inputs, size=features_16.shape[2:4], mode="bilinear", align_corners=False)  
        img_scale_2 = F.interpolate(img, size=features_4.shape[2:4], mode="bilinear", align_corners=False)  
        img_scale_4 = F.interpolate(img, size=features_8.shape[2:4], mode="bilinear", align_corners=False)  
        img_scale_8 = F.interpolate(img, size=features_16.shape[2:4], mode="bilinear", align_corners=False)  

        features_4_with_mask = torch.cat([features_4, mask_scale_2,img_scale_2], dim=1)
        features_8_with_mask = torch.cat([features_8, mask_scale_4,img_scale_4], dim=1)
        features_16_with_mask = torch.cat([features_16, mask_scale_8,img_scale_8], dim=1)

        feat_64 = self.unknown_region_predictor["scale_64"](features_4_with_mask)
        feat_128 = self.unknown_region_predictor["scale_128"](features_8_with_mask)
        feat_256 = self.unknown_region_predictor["scale_256"](features_16_with_mask)

        target_size = feat_64.shape[2:4]
        feat_128_up = F.interpolate(feat_128, size=target_size, mode="bilinear", align_corners=False)
        feat_256_up = F.interpolate(feat_256, size=target_size, mode="bilinear", align_corners=False)

        fused_features = torch.cat([feat_64, feat_128_up, feat_256_up], dim=1)
        unknown_region_logits = self.unknown_fusion(fused_features)

        return unknown_region_logits, img_scale_2, img_scale_4, img_scale_8

    def _forward_alpha_heads(
        self,
        input=None,
        backbone_features=None,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        image=None,
        frame_idx=None
    ):
        if input is not None:
            img = input
        else:
            img = image
        
        if high_res_features is not None and len(high_res_features) >= 2:
            unknown_region_inputs, img_scale_2, img_scale_4, img_scale_8  = self._detect_unknown_region(high_res_features, mask_inputs, img)
            # unknown_region_inputs: the detected ROI
            
        ste_mask=(unknown_region_inputs.sigmoid() > 0.65).float() 
        ste_mask = ste_mask*0.5+(1-ste_mask)*mask_inputs 
        # ste_mask: the pseudo trimap
        
        # Use hard threshold to separate ROI and alpha supervisions, preventing mixed-up signals.

        mask_256=ste_mask
        mask_128=F.interpolate(ste_mask, size=high_res_features[1].shape[2:4], mode="bilinear", align_corners=False)
        mask_64=F.interpolate(ste_mask, size=high_res_features[2].shape[2:4], mode="bilinear", align_corners=False)
        
        m_f_img1=torch.cat([high_res_features[2],mask_64,img_scale_8],dim=1)
        alpha1=self.unknown_alpha_predictor["scale_64"](m_f_img1)
        alpha1=self.alpha_pred1(alpha1)

        m_f_img2=torch.cat([high_res_features[1],mask_128,img_scale_4,alpha1],dim=1)
        alpha2=self.unknown_alpha_predictor["scale_128"](m_f_img2)
        alpha2=self.alpha_pred2(alpha2)

        m_f_img3=torch.cat([high_res_features[0],mask_256,img_scale_2,alpha2],dim=1)
        alpha3=self.unknown_alpha_predictor["scale_256"](m_f_img3)
        alpha3=self.alpha_pred3(alpha3)

        # unknown_region_upscaled = F.interpolate(
        #     unknown_region_inputs, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        # )
        # Uncomment for ROI visualization.

        return alpha3, None, None
        # return alpha3, [alpha1,alpha2,alpha3], unknown_region_upscaled


    def _matting_step(
        self,
        frame_idx,
        input,
        mask_inputs,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        matting_output_dict,
        num_frames,
    ):
        point_inputs = None
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        pix_feat = current_vision_feats[-1].permute(1, 2, 0)
        pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])

        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:], feat_sizes[:])
            ]
        else:
            high_res_features = None

        mask_inputs = (mask_inputs > 0.).float() 
        alpha_outputs = self._forward_alpha_heads(
            input=input,
            backbone_features=pix_feat,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
        )
        alpha, alpha_upscaled, unknown_region_upscaled = alpha_outputs
        return current_out, (None, alpha, alpha_upscaled, unknown_region_upscaled)
