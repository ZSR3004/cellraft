import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.raft.raft_memory_manager import RaftMemoryManager


class RaftDataset(Dataset):
    """
    RaftDataset is a PyTorch Dataset for loading pairs of frames and their corresponding optical flow.
    Primarily used as a wrapper around the Dataloader class to handle training and validation splits.
    """

    def __init__(self, input_paths: list[str], target_paths: list[str],
                 memory_monitor: RaftMemoryManager,
                 seed: int | None = None, training_split: float = 0.8,
                 start_index: int = 0,
                 batch_size: int = 1,
                 drop_last: bool = False,
                 cleanup_frequency: int = 5):
        """
        Initializes the RaftDataset.

        Args:
            input_paths (list[str]): List of paths to input images.
            target_paths (list[str]): List of paths to target images.
            memory_monitor (RaftMemoryManager): Memory manager for monitoring and cleaning up memory.
            seed (int | None): Random seed for reproducibility. If None, a random seed
                is generated.
            training_split (float): Fraction of the dataset to use for training. The rest is used
                for validation. Default is 0.8 (80% training, 20% validation).
            start_index (int): Starting index for the dataset. Default is 0.
            batch_size (int): Number of samples per batch. Default is 1.
            drop_last (bool): Whether to drop the last incomplete batch. Default is False.
            cleanup_frequency (int): Frequency of memory cleanup operations. Default is 5.
        """
        assert len(input_paths) == len(
            target_paths), "Input and target paths must have the same length."
        self.seed = seed if seed is not None else random.randint(
            0, 1000000)  # can't do 1e6 for some reason
        random.seed(self.seed)

        combined_paths = list(zip(input_paths, target_paths))
        random.shuffle(combined_paths)
        training_split_idx = int(len(combined_paths) * training_split)

        self.training_loader = Dataloader(combined_paths[:training_split_idx],
                                          memory_monitor=memory_monitor,
                                          start_index=start_index,
                                          batch_size=batch_size,
                                          drop_last=drop_last,
                                          cleanup_frequency=cleanup_frequency)

        self.validation_loader = Dataloader(combined_paths[training_split_idx:],
                                            memory_monitor=memory_monitor,
                                            start_index=start_index,
                                            batch_size=batch_size,
                                            drop_last=drop_last,
                                            cleanup_frequency=cleanup_frequency)


class Dataloader(Dataset):
    """
    Dataloader is a PyTorch Dataset for loading images.
    """

    def __init__(self, paths: list[tuple[str, str]],
                 memory_monitor: RaftMemoryManager,
                 start_index: int = 0,
                 batch_size: int = 1,
                 drop_last: bool = False,
                 cleanup_frequency: int = 5):
        """
        Initializes the Dataloader.

        Args:
            paths (list[tuple[str, str]]): List of tuples containing paths to frame pairs
                and their corresponding optical flow.
            memory_monitor (RaftMemoryManager): Memory manager for monitoring and cleaning up memory.
            start_index (int): Starting index for the dataset. Default is 0.
            batch_size (int): Number of samples per batch. Default is 1.
            drop_last (bool): Whether to drop the last incomplete batch. Default is False.
            cleanup_frequency (int): Frequency of memory cleanup operations. Default is 5.
        """
        self.paths = paths
        self.memory_monitor = memory_monitor
        self.start_index = start_index
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.cleanup_frequency = cleanup_frequency
        self.batch_count = 0

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.paths)

    def __iter__(self):
        """
        Returns an iterator over the dataset, yielding batches of data.
        Each batch is processed using the custom collate function to handle variable-sized inputs.
        """
        for idx in range(self.start_index, len(self), self.batch_size):
            if self.drop_last and idx + self.batch_size > len(self):
                break
            batch_indices = list(
                range(idx, min(idx + self.batch_size, len(self))))
            batch = [self[i] for i in batch_indices]
            yield self.custom_collate_fn(batch)
            self.batch_count += 1
            if self.batch_count % self.cleanup_frequency == 0:
                self.memory_monitor.memory_cleanup()

    def _make_divisible_by(self, frame: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def custom_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
