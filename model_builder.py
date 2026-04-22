import os
import sys
import torch
import torch.distributed as dist

from models.ffno_model import FFNO3D, FFNOBlock3dBalanced
from models.ffno_z1d_model import FFNO3DZ1D, FFNOBlock3dZ1D
from loss.nlte_composite_loss import NLTECompositeLoss
from loss.gradient_loss import GradientLoss
from loss.weighted_mse_loss import WeightedMSE_L1

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
        multi_gpu=False,
        debug_loss=False,
        mean_X=0,
        std_X=1,
        mean_Y=0,
        std_Y=1,
        num_epochs=None,
        use_cosine=False,
        lr_min=1e-6
    ):

        if model == "FFNO3D":
            self.model_cls = FFNO3D
            self.transformer_layer_cls = FFNOBlock3dBalanced
        elif model == "FFNO3DZ1D":
            self.model_cls = FFNO3DZ1D
            self.transformer_layer_cls = FFNOBlock3dZ1D
        else:
            sys.stderr.write(f"Invalid Model class: {model}")
            sys.exit(-1)

        self.model_config = model_config
        self.device = device

        self.lr = lr
        self.weight_decay = weight_decay

        self.chi = chi
        self.lines = lines
        self.wave = wave
        self.levels = levels
        self.atom_names = atom_names

        self.multi_gpu = multi_gpu

        self.rank = 0
        self.local_rank = 0
        self.world_size = 1

        self.debug_loss = debug_loss

        self.mean_X = mean_X

        self.std_X = std_X

        self.mean_Y = mean_Y

        self.std_Y = std_Y

        self.lr_min = lr_min

        self.num_epochs = num_epochs

        self.use_cosine = use_cosine

        if self.multi_gpu:
            self._init_distributed()

    # ------------------------------------------------
    # DISTRIBUTED INIT
    # ------------------------------------------------

    def _init_distributed(self):

        if not dist.is_initialized():

            dist.init_process_group("nccl")

        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self.world_size = dist.get_world_size()

        torch.cuda.set_device(self.local_rank)

        self.device = f"cuda:{self.local_rank}"

    # ------------------------------------------------
    # MODEL
    # ------------------------------------------------

    def wrap_model(self, model):

        if self.multi_gpu:

            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={self.transformer_layer_cls},
            )

            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
            )

        return model

    def build_model(self, wrap_fsdp=True):

        model = self.model_cls(**self.model_config)

        model = model.to(self.device)

        if wrap_fsdp:
            model = self.wrap_model(model)

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

        # -------------------------------
        # COSINE SCHEDULER
        # -------------------------------
        scheduler = None
        if self.use_cosine:
            if self.num_epochs is None:
                raise ValueError("num_epochs required for cosine scheduler")

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.lr_min
            )

        mse_loss = WeightedMSE_L1()
        mse_loss = mse_loss.to(self.device)

        gradient_loss = GradientLoss()
        gradient_loss = gradient_loss.to(self.device)

        loss_fn = NLTECompositeLoss(
            chi=self.chi,
            lines=self.lines,
            wave=self.wave,
            levels=self.levels,
            data_loss_func=mse_loss,
            gradient_loss_func=gradient_loss,
            atom_names=self.atom_names,
            debug=self.debug_loss,
            mean_X=self.mean_X,
            std_X=self.std_X,
            mean_Y=self.mean_Y,
            std_Y=self.std_Y
        )

        loss_fn = loss_fn.to(self.device)

        return optimizer, scheduler, loss_fn

    # ------------------------------------------------
    # BUILD
    # ------------------------------------------------

    def build(self):

        model = self.build_model()

        optimizer, scheduler, loss_fn = self.build_training_components(
            model
        )

        return model, scheduler, optimizer, loss_fn
