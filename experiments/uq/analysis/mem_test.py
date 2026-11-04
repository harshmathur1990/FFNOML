import os, sys, torch
sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML")
from config import MODEL_CONFIG
from models.ffno_model import FFNO3D
m = FFNO3D(**dict(MODEL_CONFIG)).cuda()
keep = set(os.environ.get("KEEP", "proj1,proj2").split(","))
for n, p in m.named_parameters():
    p.requires_grad = n.split(".")[0] in keep
print("trainable", sum(p.numel() for p in m.parameters() if p.requires_grad), "of", sum(p.numel() for p in m.parameters()))
m.train()
D = 467
x = torch.randn(1, 6, D, 40, 40, device="cuda"); z = torch.linspace(-1, 14, D, device="cuda")[None, :, None, None].expand(1, D, 40, 40).contiguous()
dx = torch.tensor([9.5e4], device="cuda"); dy = dx.clone()
for mode in ("no_grad", "grad"):
    torch.cuda.reset_peak_memory_stats()
    with torch.set_grad_enabled(mode == "grad"):
        y = m(x, z, dx, dy)
        if mode == "grad":
            y.square().mean().backward()
    print(mode, "peak GB %.2f" % (torch.cuda.max_memory_allocated() / 1e9))
