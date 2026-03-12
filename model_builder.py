import sys
import torch
import torch.distributed as dist

from models.ffno_model import *
from loss.nlte_composite_loss import NLTECompositeLoss
from loss.weighted_mse_loss import WeightedMSE

from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

class ModelBuilder:

    def __init__(
        self,
        model,
        *,
        model_config,
        chi,
        lines,
        wave,
        levels,
        atom_names,
        device="cuda",
        lr=2e-4,
        weight_decay=1e-4,
        amp=True,
        multi_gpu=False,
        debug_loss=False
    ):

        if model == "FFNO3D":
            self.model_cls = FFNO3D
        else:
            sys.stderr.write(f"Invalid Model class: {model}")
            sys.exit(-1)

        self.model_config = model_config
        self.device = device

        self.lr = lr
        self.weight_decay = weight_decay
        self.amp = amp

        self.chi = chi
        self.lines = lines
        self.wave = wave
        self.levels = levels
        self.atom_names = atom_names

        self.multi_gpu = multi_gpu

        self.rank = 0
        self.world_size = 1

        self.debug_loss = debug_loss

        if self.multi_gpu:
            self._init_distributed()

    # ------------------------------------------------
    # DISTRIBUTED INIT
    # ------------------------------------------------

    def _init_distributed(self):

        if not dist.is_initialized():

            dist.init_process_group("nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        torch.cuda.set_device(self.rank)

        self.device = f"cuda:{self.rank}"

    # ------------------------------------------------
    # MODEL
    # ------------------------------------------------

    def build_model(self):

        model = self.model_cls(**self.model_config)

        model = model.to(self.device)

        if self.multi_gpu:

            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={FFNOBlock3d},
            )

            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
            )

        return model

    # ------------------------------------------------
    # OPTIMIZER / LOSS / AMP
    # ------------------------------------------------

    def build_training_components(self, model):

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        mse_loss = WeightedMSE()
        mse_loss = mse_loss.to(self.device)

        loss_fn = NLTECompositeLoss(
            chi=self.chi,
            lines=self.lines,
            wave=self.wave,
            levels=self.levels,
            data_loss_func=mse_loss,
            atom_names=self.atom_names,
            debug=self.debug_loss
        )

        loss_fn = loss_fn.to(self.device)

        scaler = None
        if self.amp and self.device.startswith("cuda"):
            scaler = torch.amp.GradScaler("cuda")

        return optimizer, loss_fn, scaler

    # ------------------------------------------------
    # BUILD
    # ------------------------------------------------

    def build(self):

        model = self.build_model()

        optimizer, loss_fn, scaler = self.build_training_components(
            model
        )

        return model, optimizer, loss_fn, scaler
