import os
import time
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import copy

from datasets import DataLoaderLite
from gpt import GPT, GPTConfig

# ------------------------------------------------------------------------------
pl.seed_everything(1337)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")
# -----------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    n_gpu: int = torch.cuda.device_count() # Use all available gpus
    B: int = 16
    T: int = 256 # 1024
    lr_base = 3e-4
    lr_overshoot = 3e-3
    epochs: int = 4
    weight_decay: float = 0.1
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 0.5


class GPTTranner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, config: TrainerConfig):
        super(GPTTranner, self).__init__()
        self.automatic_optimization = False
        self.base_model = model
        self.overshoot_model = copy.deepcopy(model)
        self.config = config
        self.start_time = time.time()
        self.optimizers_configured = False


    def training_step(self, batch, batch_idx):
        for opt in self.optimizers():
            opt.zero_grad()
        
        logits, loss = self.overshoot_model.forward(batch[0], batch[1])
        if batch_idx % 10 == 0:
            with torch.no_grad():
                _, loss_base = self.base_model.forward(batch[0], batch[1])
        
        
        self.manual_backward(loss)
        
        # Gradients OVERSHOOT -> BASE
        for (name1, param1), (name2, param2) in zip(self.overshoot_model.named_parameters(), self.base_model.named_parameters()):
            if param1.grad is not None:
                assert name1 == name2, "Parameter names do not match between models."
                param2.grad = param1.grad.clone()

        # Weights BASE -> OVERSHOOT
        for param1, param2 in zip(self.base_model.parameters(), self.overshoot_model.parameters()):
            param2.data = param1.data.clone()
        
        
        for opt in self.optimizers():
            opt.step()
            
        
        self.base_scheduler.step(); self.overshoot_scheduler.step()
        for device_id in range(self.config.n_gpu):
            torch.cuda.synchronize(device_id)  # wait for the GPUs to finish work

        now = time.time()
        dt = now - self.start_time  # time difference in seconds
        self.start_time = now
        tokens_processed = self.config.B * self.config.T
        tokens_per_sec = tokens_processed / dt
        lr_base = self.base_scheduler.get_last_lr()[-1]
        lr_overshoot = self.overshoot_scheduler.get_last_lr()[-1]

        gpu_info = ""
        for gpu_index in range(self.config.n_gpu):
            max_vram = torch.cuda.memory_reserved(gpu_index) / (1024 * 1024 * 1024)
            utilization = torch.cuda.utilization(gpu_index)
            gpu_info += f" | vram{gpu_index} {max_vram:.2f}GB | util{gpu_index} {utilization:.2f}%"
        if batch_idx % 10 == 0:
            print(
                f"process: {os.environ.get('SLURM_PROCID', 0)} | step {batch_idx:4d} | lr_base: {lr_base:.4f} | lr_overshoot: {lr_overshoot:.4f} | loss_base: {loss_base.item():.6f} | loss_overshoot: {loss.item():.6f}  | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}{gpu_info}", flush=True
            )
        self.log_dict({"train_loss": loss.item(), "lr_base": self.base_scheduler.get_last_lr()[-1], "lr_overshoot": self.overshoot_scheduler.get_last_lr()[-1]})
        return loss

    def configure_optimizers(self):
        if not hasattr(self, 'steps'):
            train_dataset = DataLoaderLite(T=self.config.T)
            self.steps = self.config.epochs * len(train_dataset) // self.config.B // self.config.n_gpu

        for model_name in ['base', 'overshoot']:
            # start with all of the candidate parameters (that require grad)
            param_dict = {pn: p for pn, p in self.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if pn.startswith(model_name)}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # Create AdamW optimizer and use the fused version if it is available
            opt = torch.optim.AdamW(optim_groups, lr=getattr(self.config, f'lr_{model_name}'), betas=(0.9, 0.95), eps=1e-8, fused=False)
            # opt = MomentumOptimizer(optim_groups, lr=getattr(self.config, f'lr_{model_name}'), momentum=0)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=self.steps) 
            setattr(self, f'{model_name}_opt', opt)
            setattr(self, f'{model_name}_scheduler', lr_scheduler)
        return [self.base_opt, self.overshoot_opt]

    def train_dataloader(self):
        train_dataset = DataLoaderLite(T=self.config.T)
        self.steps = self.config.epochs * len(train_dataset) // self.config.B // self.config.n_gpu
        return DataLoader(train_dataset, batch_size=self.config.B, num_workers=15)


# -----------------------------------------------------------------------------
def main():
    model = GPT(GPTConfig(vocab_size=50304))

    # Doesn't work inside devana slurn job
    # model = torch.compile(model)

    trainer_config = TrainerConfig()
    gpt_trainer = GPTTranner(model, trainer_config)
    trainer = pl.Trainer(
        max_epochs=trainer_config.epochs,
        # accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        # gradient_clip_val=trainer_config.gradient_clip_val,
        precision="16-mixed",
        enable_progress_bar=False,
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir="lightning_logs", name="demo"),
        devices=trainer_config.n_gpu,
        strategy='deepspeed_stage_2' if trainer_config.n_gpu > 1 else 'auto',
    )
    # trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    trainer.fit(gpt_trainer)


if __name__ == "__main__":
    main()



# [[p.grad for p in g['params']] for g in opt.param_groups]