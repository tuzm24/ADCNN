import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class torchUtil:
    def __init__(self):
        assert 0

    @staticmethod
    def psnr_nomalized(loss):
        if loss <= 0:
            return 100
        return loss.reciprocal().log10()*10

    @staticmethod
    def psnr(gt, output):
        return ((gt-output)**2).mean().reciprocal().log10() * 10

    @staticmethod
    def psnr_by_list(gt, output_list, length):
        mse = torch.tensor(0.0)
        start = 0
        for output in output_list:
            mse += ((gt[start:start+length]-output)**2).mean()
            start += length
        return mse.reciprocal().log10() * 10

    @staticmethod
    def psnr_from_mse(mse):
        if mse <= 0:
            return 100
        return mse.reciprocal().log10() * 10




class mathUtil:
    def __init__(self):
        assert 0

    @staticmethod
    def psnr(loss):
        if loss <= 0:
            return 100
        return math.log10(1 / loss) * 10