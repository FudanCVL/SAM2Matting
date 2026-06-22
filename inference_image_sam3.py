import os
import torch
import numpy as np
from PIL import Image
from sam3.model.build_sam3matting import build_sam3matting
from sam3.model.sam3matting_image_predictor import SAM3MattingImagePredictor
from iopath.common.file_io import g_pathmgr

checkpoint = "checkpoints/SAM2Matting-SAM3.pt"

image_path = "demo/image/image.jpg"
mask_path = "demo/image/mask.png"
output_folder = "output_image"
os.makedirs(output_folder, exist_ok=True)

def load_tracker_state_dict(checkpoint):
    with g_pathmgr.open(checkpoint, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    sd = ckpt["model"]
    out = {}
    for k, v in sd.items():
        if k.startswith("detector.backbone.vision_backbone."):
            out[k.removeprefix("detector.")] = v
        elif k.startswith("tracker."):
            out[k.removeprefix("tracker.")] = v
    return out

sd = load_tracker_state_dict(checkpoint)
model = build_sam3matting(checkpoint=None)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing keys: ", missing)
print("unexpected keys: ", unexpected)
predictor = SAM3MattingImagePredictor(model)


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
        size=(288, 288),
        mode="bilinear",
        align_corners=False,
    )

    _, alpha, _, _ = predictor.predict(
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