import os, sys, torch, torch.distributed as dist
from functools import partial
sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML")
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from config import MODEL_CONFIG
from models.ffno_model import FFNO3D, FFNOBlock3dBalanced
dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
m = FFNO3D(**dict(MODEL_CONFIG))
for n, p in m.named_parameters():
    p.requires_grad = n.split(".")[0] in ("proj1", "proj2")
from torch.nn.parallel import DistributedDataParallel
m = DistributedDataParallel(m.cuda(), device_ids=[torch.cuda.current_device()])
m.train()
D = 467
x = torch.randn(1, 6, D, 40, 40, device="cuda"); z = torch.linspace(-1, 14, D, device="cuda")[None, :, None, None].expand(1, D, 40, 40).contiguous()
dx = torch.tensor([9.5e4], device="cuda"); dy = dx.clone()
torch.cuda.reset_peak_memory_stats()
y = m(x, z, dx, dy)
print("after forward GB %.2f, y.requires_grad %s" % (torch.cuda.max_memory_allocated() / 1e9, y.requires_grad))
y.square().mean().backward()
print("after backward peak GB %.2f" % (torch.cuda.max_memory_allocated() / 1e9))
dist.destroy_process_group()
