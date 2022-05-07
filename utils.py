import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import datetime
import os
import sys
import cv2
from math import exp
from pytorch_msssim import ssim
import importlib


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    image = image / 255. ## image in range (0, 1)
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    cb: torch.Tensor = -37.797 * r + -74.203 * g + 112.0 * b + 128.0
    cr: torch.Tensor = 112.0 * r + -93.786 * g + -18.214 * b + 128.0

    return torch.stack((y, cb, cr), -3)

def prepare_qat(model):
    ## fuse model
    model.module.fuse_model()
    ## qconfig and qat-preparation & per-channel quantization
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    # model.qconfig = torch.quantization.QConfig(
    #     activation=torch.quantization.FakeQuantize.with_args(
    #         observer=torch.quantization.MinMaxObserver, 
    #         quant_min=-128,
    #         quant_max=127,
    #         qscheme=torch.per_tensor_symmetric,
    #         dtype=torch.qint8,
    #         reduce_range=False),
    #     weight=torch.quantization.FakeQuantize.with_args(
    #         observer=torch.quantization.MinMaxObserver, 
    #         quant_min=-128, 
    #         quant_max=+127, 
    #         dtype=torch.qint8, 
    #         qscheme=torch.per_tensor_symmetric, 
    #         reduce_range=False)
    # )
    model = torch.quantization.prepare_qat(model, inplace=True)
    return model

def import_module(name):
    return importlib.import_module(name)

def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)

def calc_ssim(sr, hr):
    ssim_val = ssim(sr, hr, size_average=True)
    return float(ssim_val)
    
def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content


class ExperimentLogger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')
    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_stat_dict():
    stat_dict = {
        'epochs': 0,
        'losses': [],
        'ema_loss': 0.0,
        'set5': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'set14': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'b100': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'u100': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'manga109': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        }
    }
    return stat_dict

if __name__ == '__main__':
    timestamp = cur_timestamp_str()
    print(timestamp)