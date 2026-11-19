import numpy as np
import logging
import random
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, win_size):
        self.vals = []
        self.win_size = win_size
        self.avg = 0

    def reset(self):
        self.vals.clear()
        self.avg = 0

    def update(self, val, n=1):
        self.vals.append(val)
        if len(self.vals) > self.win_size:
            self.vals.pop(0)
        self.avg = np.mean(self.vals)

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.enabled = True  # 默认值
    torch.backends.cudnn.benchmark = False  # 默认为False
    torch.backends.cudnn.deterministic = True  # 默认为False