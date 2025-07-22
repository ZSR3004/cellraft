import os
import gc
import json
import psutil
import threading as th
from typing import final
from contextlib import contextmanager

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models.optical_flow import raft_small  # small for optimization

from src.raft.raft_dataset import MemoryManagedRaftDataSet, custom_collate_fn
import src.raft.raft_memory_manager as ramem


@contextmanager
def context_manager(memory_monitor: ramem.RaftMemoryManager):
    try:
        memory_monitor.memory_cleanup()
        old_deterministic = torch.backends.cudnn.deterministic
        old_benchmark = torch.backends.cudnn.benchmark
    except Exception as e:
        print(f"Error during context manager setup: {e}")
        raise

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark
        memory_monitor.memory_cleanup()


class GradientAccumulator:
    """
    Accumulates gradients for a model.
    """

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_step(self) -> bool:
        """
        Check if the current step count is a multiple of the accumulation steps.
        """
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0

    def scale_loss(self, loss: float) -> float:
        """
        Scale the loss by the number of accumulation steps.

        Args:
            loss (float): The loss value to scale.

        Returns:
            float: The scaled loss value.
        """
        return loss / self.accumulation_steps


def setup_optimizer(model: torch.nn.Module,
                    lr: float = 1e-4, weight_decay: float = 1e-4,
                    num_epochs=10) -> tuple[optim.AdamW, optim.lr_scheduler.CosineAnnealingLR]:
    """
    Setup the optimizer and learning rate scheduler for the model.

    Args:
        model (torch.nn.Module): The model to optimize.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        num_epochs (int): Number of epochs for the learning rate scheduler.

    Returns:
        tuple: A tuple containing the optimizer and the learning rate scheduler.
    """ 
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,
        amsgrad=False,
        foreach=True
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr * 0.01
    )

    return optimizer, scheduler

def compute_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, batch_idx: int = None) -> float:
    pass

def init_model():
    pass

def train_model():
    pass

def train_epoch():
    pass 

def run_validation_epoch():
    pass


