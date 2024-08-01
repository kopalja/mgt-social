import os
import time
from dataclasses import dataclass

import pytorch_lightning as pl
import pandas as pd
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

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
    B: int = 32
    T: int = 256 # 1024
    lr: float = 3e-4
    epochs: int = 4
    weight_decay: float = 0.1
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 0.5


class GPTTranner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, config: TrainerConfig):
        super(GPTTranner, self).__init__()
        self.automatic_optimization = False
        self.model = model
        self.config = config
        self.start_time = time.time()
        self.training_stats = {key: [] for key in ["step", "base_lr", "overshoot_lr", "base_loss", "overshoot_loss"]}


    def training_step(self, batch, batch_idx):
        self.optimizers().zero_grad()
        logits, loss = self.model.forward(batch[0], batch[1])
        self.manual_backward(loss)
        self.optimizers().step()
        self.lr_scheduler.step()
        
        for device_id in range(self.config.n_gpu):
            torch.cuda.synchronize(device_id)  # wait for the GPUs to finish work

        now = time.time()
        dt = now - self.start_time  # time difference in seconds
        self.start_time = now
        tokens_processed = self.config.B * self.config.T
        tokens_per_sec = tokens_processed / dt
        lr = self.lr_scheduler.get_last_lr()[-1]

        gpu_info = ""
        for gpu_index in range(self.config.n_gpu):
            max_vram = torch.cuda.memory_reserved(gpu_index) / (1024 * 1024 * 1024)
            utilization = torch.cuda.utilization(gpu_index)
            gpu_info += f" | vram{gpu_index} {max_vram:.2f}GB | util{gpu_index} {utilization:.2f}%"
        if batch_idx % 10 == 0:
            print(
                f"process: {os.environ.get('SLURM_PROCID', 0)} | step {batch_idx:4d} | lr: {lr:.4f} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}{gpu_info}", flush=True
            )
        self.training_stats["step"].append(batch_idx + (self.current_epoch * len(self._train_dataset) // self.config.B))
        self.training_stats["base_lr"].append(lr)
        self.training_stats["overshoot_lr"].append(lr)
        self.training_stats["base_loss"].append(loss.item())
        self.training_stats["overshoot_loss"].append(loss.item())
        
        self.log_dict({"base_lr": lr, "overshoot_lr": lr, "base_loss": loss.item(), "overshoot_loss": loss.item()})
        return loss

    def configure_optimizers(self):
        if not hasattr(self, '_train_dataset'):
            self._train_dataset = DataLoaderLite(T=self.config.T)
            self.steps = int(round(self.config.B + self.config.epochs * len(self._train_dataset) // self.config.B // self.config.n_gpu))
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
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
        self.opt = torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=(0.9, 0.95), eps=1e-8, fused=False)
        return self.opt

    def train_dataloader(self):
        self._train_dataset = DataLoaderLite(T=self.config.T)
        # steps = self.config.epochs * len(self._train_dataset) // self.config.B // self.config.n_gpu
        steps = int(round(self.config.B + self.config.epochs * len(self._train_dataset) // self.config.B // self.config.n_gpu))
        print("Traning steps: ", steps)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, T_0=steps)
        return DataLoader(self._train_dataset, batch_size=self.config.B)


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
        # log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir="lightning_logs", name="baseline"),
        devices=trainer_config.n_gpu,
        strategy='deepspeed_stage_2' if trainer_config.n_gpu > 1 else 'auto',
    )
    # trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    trainer.fit(gpt_trainer)
    pd.DataFrame(gpt_trainer.training_stats).to_csv(os.path.join("results_gutenberg", "overshoot_baseline.csv"), index=False)

if __name__ == "__main__":
    main()
