import tqdm as tq
import torch
import random


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def enable_tqdm(it, *args, **kwargs):
    return tq.tqdm(it, *args, **kwargs)


def disable_tqdm(it, *args, **kwargs):
    return it
