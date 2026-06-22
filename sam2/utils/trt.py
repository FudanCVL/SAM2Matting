import torch
import torch.nn as nn

def replace_unknown_alpha_predictor_with_trt(
    predictor: nn.Module,
    workspace_gb: int = 4,
) -> nn.Module:
    import torch_tensorrt.dynamo as trt_dynamo
    from torch.export import export as torch_export

    uap: nn.ModuleDict = predictor.unknown_alpha_predictor
    ex_shapes = {
        "scale_64":  (1, 260,  64,  64),
        "scale_128": (1,  69, 128, 128),
        "scale_256": (1,  37, 256, 256),
    }

    compiled = {}
    for key in ("scale_64", "scale_128", "scale_256"):
        print(f"[Torch-TRT] compiling {key} with input shape {ex_shapes[key]} ...")
        compiled[key] = _compile_one(uap[key], ex_shapes[key], workspace_gb, trt_dynamo, torch_export)
        print(f"[Torch-TRT] {key} done.")

    predictor.unknown_alpha_predictor["scale_64"]  = _ToHalfWrapper(compiled["scale_64"])
    predictor.unknown_alpha_predictor["scale_128"] = _ToHalfWrapper(compiled["scale_128"])
    predictor.unknown_alpha_predictor["scale_256"] = _ToHalfWrapper(compiled["scale_256"])
    return predictor


def _to_half_eval_cuda(m: nn.Module):
    return m.eval().to("cuda").half()


def _compile_one(mod, ex_shape, workspace_gb, trt_dynamo, torch_export):
    example = torch.empty(ex_shape, device="cuda", dtype=torch.half)
    mod = _to_half_eval_cuda(mod)
    exprog = torch_export(mod, (example,))
    return trt_dynamo.compile(
        exprog,
        inputs=[example],
        enabled_precisions={torch.half},
        workspace_size=workspace_gb << 30,
        min_block_size=2,
    )


class _ToHalfWrapper(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    @torch.inference_mode()
    def forward(self, x):
        x = x.to(device="cuda", dtype=torch.half, non_blocking=True).contiguous()
        return self.mod(x)