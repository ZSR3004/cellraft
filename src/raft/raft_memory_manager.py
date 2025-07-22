"""Monitors and manages memory usage for RAFT models."""
import os
import gc
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
