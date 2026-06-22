import os
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2matting
from sam2.sam2matting_image_predictor import SAM2MattingImagePredictor

variant = "sam2.1tiny"

if variant == "sam2.1tiny":
    checkpoint = "checkpoints/SAM2Matting-SAM2.1Tiny.pt"
    model_cfg = "configs/sam2matting-sam2.1tiny.yaml"
elif variant == "sam2.1base+":
    checkpoint = "checkpoints/SAM2Matting-SAM2.1Base+.pt"
    model_cfg = "configs/sam2matting-sam2.1base+.yaml"
else:
    raise ValueError(f"Invalid variant: {variant}")

image_path = "demo/image/image.jpg"
mask_path = "demo/image/mask.png"
output_folder = "output_image"
os.makedirs(output_folder, exist_ok=True)

predictor = SAM2MattingImagePredictor(build_sam2matting(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = Image.open(image_path)
    img = predictor.set_image(image)

    mask_image = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_image)

    raw_mask = (torch.from_numpy(mask_np) / 255) > 0
    mask_input = (torch.from_numpy(mask_np) > 0).float() * 20 - 10
    mask_input = mask_input.unsqueeze(0).unsqueeze(0)

    mask_input = torch.nn.functional.interpolate(
        mask_input,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )

    _, alpha, _ = predictor.predict(
        img=img,
        raw_mask=raw_mask,
        mask_input=mask_input,
        multimask_output=False,
    )

alpha_result = (alpha * 255).astype(np.uint8).squeeze()
Image.fromarray(alpha_result, mode="L").save(os.path.join(output_folder, "pha.png"))

bg = np.full((*alpha_result.shape, 3), [120, 255, 155], dtype=np.uint8)
alpha_green = (np.array(image.convert("RGB")) * (alpha_result[..., None] / 255.0) + bg * (1.0 - alpha_result[..., None] / 255.0)).astype(np.uint8)
Image.fromarray(alpha_green).save(os.path.join(output_folder, "fgr.png"))