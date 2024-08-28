from itertools import cycle
from typing import Callable, Dict, List, Optional, Tuple

import gc
import pynvml
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from llmcompressor.pytorch.utils import tensors_module_forward, tensors_to_device

__all__ = ["EarlyStopException", "apply_pad_mask_to_batch", "run_calibration_forward"]


class EarlyStopException(Exception):
    """
    Exception for stopping execution of a PyTorch model early, and saving the
    inputs of the stopped module offloaded to cpu

    :param args: inputs passed to the layer where the exception was raised
    :param kwargs: keyword inputs passed to the layer where the excetion was raised
    """

    def __init__(self, args: Tuple, kwargs: Dict):
        self.args = tensors_to_device(args, "cpu")
        self.kwargs = kwargs


def apply_pad_mask_to_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Apply a mask to the input ids of a batch. This is used to zero out
    padding tokens so they do not contribute to the hessian calculation in the
    SparseGPT algorithm

    :param batch: batch to apply padding to if it exists
    :return: batch with padding zeroed out in the input_ids
    """
    batch["input_ids"] = batch["input_ids"] * batch["attention_mask"]
    return batch


def run_calibration_forward(
    model: Module,
    calibration_dataloader: DataLoader,
    num_calibration_steps: Optional[int] = None,
    calibration_function: Optional[Callable] = None,
    device: Optional[str] = None,
    mask_padding: bool = False,
) -> List[torch.Tensor]:
    """
    Helper function used by one-shot modifiers, runs calibration data through a model to
    update modifier statistics and trigger hooks

    :param model: PyTorch model to run
    :param calibration_dataloader: data to use for calibration
    :param num_calibration_steps: number of items in calibration_dataloader to process,
    None or a negative number to process all available data
    :param calibration_function: option to pass a custom forward function for model
    :param device: option to move the model to a specific device before calibration
    :param mask_padding: whether to zero out padding tokens during calibration
    :returns: list of last calculated model output if early stopping is triggered
    """
    model.eval()

    forward_fn: Callable = (
        calibration_function if calibration_function else tensors_module_forward
    )

    # move model to optional specified device if it is not already there
    model_device = next(model.parameters()).device
    if device is not None and model_device != device:
        model.to(device)
        model_device = next(model.parameters()).device
    _dataloader = (
        calibration_dataloader
        if num_calibration_steps is None
        else cycle(calibration_dataloader)
    )

    # Store any inputs caught from early stopping, used for sequential compression
    # of GPTQ, SparseGPT and WANDA
    intermediates = []

    # run through the calibration data
    for batch_idx, batch in enumerate(tqdm(_dataloader)):
        if num_calibration_steps and batch_idx >= num_calibration_steps:
            break
        if mask_padding:
            batch = apply_pad_mask_to_batch(batch)
        batch = tensors_to_device(batch, model_device)
        with torch.no_grad():
            try:
                forward_fn(batch, module=model)
            except EarlyStopException as e:
                # model was stopped early, save last calculated output and
                # move on to next calibration sample
                intermediates.append((e.args, e.kwargs))

        # TODO: not ideal, figure out where we aren't freeing memory instead
        # currently without this we run OOM on the 2nd forward pass
        print_gpu_memory()
        print_memory_free_MiB()

        import os
        import psutil
        pid = os.getpid()
        python_process = psutil.Process(pid)
        memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
        print('memory use:', memoryUse)

        #gc.collect()
        torch.cuda.empty_cache()

    return intermediates


def print_gpu_memory():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()

    print(f"Allocated: {allocated_memory / (1024 ** 2):.2f} MB")
    print(f"Cached: {cached_memory / (1024 ** 2):.2f} MB")
    print(f"Total available: {total_memory / (1024 ** 2):.2f} MB")
    print(torch.cuda.mem_get_info())
    print()

def print_memory_free_MiB(gpu_index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(mem_info.free // 1024 ** 2)