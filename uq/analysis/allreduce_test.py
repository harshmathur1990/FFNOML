import os, torch, torch.distributed as dist
r = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(r)
dist.init_process_group("nccl")
x = torch.ones(1 << 20, device="cuda") * (r + 1); dist.all_reduce(x); torch.cuda.synchronize()
print(r, "ok", float(x[0])); dist.destroy_process_group()
