import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.raft.raft_memory_manager import RaftMemoryManager

class RaftDataset(Dataset):
    def __init__(self, input_path: list[str], target_path: list[str],
                 seed: int, max_samples: int | None = None):
        """
        Initializes the RaftDataset.
        Args:
            input_path (list[str]): List of paths to input images.
            target_path (list[str]): List of paths to target images.
            max_samples (int | None): Maximum number of samples to use. If None, uses all samples.
            seed (int | None): Random seed for reproducibility. If None, a random seed is generated.
        """
        if len(input_path) != len(target_path):
            raise ValueError(
                "Input and target paths must have the same length.")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be a positive integer or None.")
        if seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        self.input_path = input_path
        self.target_path = target_path
        self.max_samples = max_samples if max_samples is not None else len(
            input_path)

        self.paths = [(self.input_path[i], self.target_path[i])
                      for i in range(len(self.input_path))]
        random.shuffle(self.paths)
        self.paths = self.paths[:self.max_samples]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.paths)

    def _make_divsible_by(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Pads the input tensor to make its height and width divisible by 8.

        Args:
            frame (torch.Tensor): Input tensor representing a frame.

        Returns:
            torch.Tensor: Padded tensor with height and width divisible by 8.
        """
        h, w = frame.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            frame = F.pad(frame, (0, pad_w, 0, pad_h), mode='reflect')
        return frame

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_pair_path, flow_path = self.paths[idx]
        try:
            frame_pair = torch.load(
                frame_pair_path, weights_only=False, map_location='cpu')
            flow = torch.load(flow_path, weights_only=False,
                              map_location='cpu')

            f1 = frame_pair[0]
            f2 = frame_pair[1]

            f1 = self._make_divsible_by(f1)
            f2 = self._make_divsible_by(f2)
            frame_pair = torch.stack([f1, f2], dim=0)
            flow = self._make_divsible_by(flow)
            return frame_pair, flow
        except Exception as e:
            print(
                f"Error loading data from {frame_pair_path} or {flow_path}: {e}")
            raise e


def custom_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-sized inputs in a batch.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor]]): List of tuples containing frame pairs and flows.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Stacked tensors for frame1, frame2, and flows.
    """
    try:
        frame_pairs = [item[0] for item in batch]
        flows = [item[1] for item in batch]

        max_h = max(fp.shape[-2] for fp in frame_pairs)
        max_w = max(fp.shape[-1] for fp in frame_pairs)

        padded_frame1_list = []
        padded_frame2_list = []
        padded_flows = []

        for fp, flow in zip(frame_pairs, flows):
            frame1, frame2 = fp[0], fp[1]

            padded_frame1 = F.pad(
                frame1, (0, max_w - frame1.shape[-1], 0, max_h - frame1.shape[-2]), mode='reflect')
            padded_frame2 = F.pad(
                frame2, (0, max_w - frame2.shape[-1], 0, max_h - frame2.shape[-2]), mode='reflect')

            padded_flow = F.pad(
                flow, (0, max_w - flow.shape[-1], 0, max_h - flow.shape[-2]), mode='reflect')

            padded_frame1_list.append(padded_frame1)
            padded_frame2_list.append(padded_frame2)
            padded_flows.append(padded_flow)

        frame1_batch = torch.stack(padded_frame1_list, dim=0)
        frame2_batch = torch.stack(padded_frame2_list, dim=0)
        batch_flows = torch.stack(padded_flows, dim=0)

        return frame1_batch, frame2_batch, batch_flows

    except Exception as e:
        raise RuntimeError(f"Error in custom collate function: {e}")


class MemoryManagedRaftDataSet():
    """A wrapper around RaftDataset that manages memory usage by limiting
    the number of samples loaded at once."""
    def __init__(self, dataloader: RaftDataset, memory_monitor: 
                 RaftMemoryManager, 
                 cleanup_frequency: int = 5): 
        self.dataloader = dataloader
        self.memory_monitor = memory_monitor
        self.cleanup_frequency = cleanup_frequency
        self.batch_count = 0

    def __iter__(self):
        for batch in self.dataloader:
            self.batch_count += 1
            if self.batch_count % self.cleanup_frequency == True:
                self.memory_monitor.checkup_and_clean()
            yield batch

    def __len__(self):
        """wrapper around dataloader length"""
        return len(self.dataloader)
