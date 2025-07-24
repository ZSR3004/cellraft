"""Monitors and manages memory usage for RAFT models."""
import os
import gc
import glob
import json
import torch
import psutil


class RaftMemoryManager:
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        self.peak_memory = 0

    @staticmethod
    def enable_memory_efficient_mode():
        """Enable memory-efficient settings for PyTorch."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(2)
        gc.set_threshold(100, 5, 5)

    def get_memory_usage(self):
        """
        Get the current memory usage of the process in MB.
        """
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, memory_mb)
        return memory_mb

    def memory_cleanup(self, n: int = 3):
        """
        Perform memory cleanup by invoking garbage collection multiple times.

        Args:
            n (int): Number of times to invoke garbage collection.
        """
        for _ in range(n):
            gc.collect()

        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = False

    def checkup_and_clean(self):
        """Check memory usage and perform cleanup if it exceeds threshold"""
        current_memory = self.get_memory_usage()
        if current_memory > self.threshold_mb:
            print(
                f"Memory usage is high: {current_memory:.1f} MB, cleaning up...")
            self.memory_cleanup()
            new_memory = self.get_memory_usage()
            print("Memory after cleanup: {new_memory:.1f} MB")


class CheckpointManager:
    @staticmethod
    def find_latest_checkpoint() -> str | None:
        """
        Find the latest checkpoint file in the current directory.

        Returns:
            str or None: The path to the latest checkpoint file, or None 
                if no checkpoints exist.
        """
        checkpoint_files = glob.glob("checkpoint_*.pt")
        if not checkpoint_files:
            return None

        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint

    @staticmethod
    def cleanup_old_checkpoints(keep_last_n: int = 3, pattern: str = "_*.pt"):
        """
        Clean up old checkpoint files, keeping only the last `keep_last_n` files.

        Args:
            keep_last_n (int): Number of recent checkpoints to keep.
            pattern (str): Glob pattern to match checkpoint files.
        """
        checkpoint_files = glob.glob(pattern)
        if len(checkpoint_files) <= keep_last_n:
            return

        sorted_files = sorted(checkpoint_files, key=os.path.getmtime)

        for old_checkpoint in sorted_files[:-keep_last_n]:
            try:
                os.remove(old_checkpoint)
            except Exception as e:
                print(f"Warning: Could not remove {old_checkpoint}: {e}")

    def __init__(self, memory_monitor: RaftMemoryManager):
        latest_checkpoint = self.find_latest_checkpoint()
        print(f"Latest checkpoint: {latest_checkpoint}")
        resume_training_bool = input(
            "Resume training from latest checkpoint? (y/n): ").lower() == 'y'
        self.latest_checkpoint = latest_checkpoint if resume_training_bool else None

        self.memory_monitor = memory_monitor

    def save_checkpoint(self,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler._LRScheduler,
                        epoch: int,
                        batch_idx: int,
                        loss_value: float,
                        best_loss: float,
                        successful_batches: int):
        """
        Save the current state of the model, optimizer, and scheduler to a checkpoint file.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to save.
            epoch (int): The current epoch number.
            batch_idx (int): The current batch index.
            loss_value (float): The current loss value.
            best_loss (float): The best loss value observed so far.
            successful_batches (int): The number of successful batches processed.
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
        try:
            torch.save({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_value,
                'best_loss': best_loss,
                'successful_batches': successful_batches,
                'peak_memory_mb': self.memory_monitor.peak_memory
            }, checkpoint_name)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")


class JSONManager:
    def __init__(self, path: str):
        self.path = path
        try:
            with open(path, 'r') as f:
                self.log = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print('No valid JSON log found, initializing new log.')
            
            with open(path, 'w') as f:
                json.dump({}, f)
            self.log = {
                "epochs": [],
                "batch_losses": [],
                "overall_stats": {
                    "total_epochs": 0,
                    "total_batches": 0,
                    "best_train_loss": -1.0,
                    "best_val_loss": -1.0
                }
            }

    def save_training_log(self):
        """Save the current log to the JSON file."""
        try:
            with open(self.path, 'w') as f:
                json.dump(self.log, f, indent=2)
                return True
        except Exception as e:
            raise OSError(f"Warning: Failed to save training log: {e}")

    def update_epoch_log(self, epoch_data: dict):
        pass

    def update_batch_log(self, batch_loss: float):
        pass 
