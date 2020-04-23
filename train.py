import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, ConcatDataset, Sampler, RandomSampler, BatchSampler, DataLoader
import torch
import torch.nn as nn
import shutil
from PIL import Image

import numpy as np
import os
from collections import namedtuple
from collections import OrderedDict
from importlib import import_module
from collections.abc import Iterable

from help_func.logging import LoggingHelper
from help_func.help_torch_parallel import DataParallelCriterion, DataParallelModel
from help_func.warmup_scheduler import GradualWarmupScheduler
from help_func.help_func import torchUtil
from help_func.CompArea import  PictureFormat, LearningIndex
from help_func.help_torch import NetTrainAndTest


from CfgEnv.loadCfg import NetManager
from CfgEnv.loadData import DataBatch
from model import Model


class CommonDatatSetting():
    def __init__(self):
        self.input_data_channel = 3
        self.output_data_channel = 3
        self.data_padding = 0
        self.qplist = [22,27,32,37]
        self.input_transform = False

class _TrainAndValidDataBatch(DataBatch, CommonDatatSetting):
    """
    input_idx : PictureFormat Class (image)
    others_return_idx : PictureFormat (image)
    input_map_idx : (0:qp, 1:mode, 2: depth, 3: horTr, 4: VerTr)
    scalr_values : (input_map_idx by scalr)
    """
    def __init__(self, istraining, input_idx = 3, others_return_idx = (0,2),
                 input_map_idx = (0,), scalar_values = ()):
        DataBatch.__init__(self, istraining=istraining, batch_size=NetManager.BATCH_SIZE)
        CommonDatatSetting.__init__(self)
        self.input_idx = input_idx
        self.others_return_idx = others_return_idx
        self.input_map_idx = input_map_idx
        self.scalar_values = scalar_values

    def __getitem__(self, index):
        if self.data[index] is None:
            self.data[index] = self.unpackData(self.batch[index])
        return self.data[index]

    def unpackData(self, info):
            DataBatch.unpackData(self, info)
            input = np.stack([*self.reshapeFuncs[self.input_idx]()]+
                              [self.tulist.getTuMaskFromIndex(x, info[2], info[1])
                               for x in self.input_map_idx], axis=0)
            others = []
            for x in self.others_return_idx:
                others.append(np.stack([*self.reshapeFuncs[x]()],
                                       axis=0))
            scalar_list = [self.tulist.getMeanTuValue(x) for x in self.scalar_values]
            return (input, *others, *scalar_list)

    def __len__(self):
        return len(self.data)


if __name__== '__main__':
    logger = LoggingHelper.get_instance().logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.__version__)
    logger.info('device {}'.format(device))


    train_dataset = _TrainAndValidDataBatch(istraining=LearningIndex.TRAINING, input_idx=PictureFormat.UNFILTEREDRECON,
                                            others_return_idx=(PictureFormat.ORIGINAL, PictureFormat.RECONSTRUCTION),
                                            input_map_idx=(), scalar_values=())
    valid_dataset = _TrainAndValidDataBatch(istraining=LearningIndex.VALIDATION, input_idx=PictureFormat.UNFILTEREDRECON,
                                            others_return_idx=(PictureFormat.ORIGINAL, PictureFormat.RECONSTRUCTION),
                                            input_map_idx=(), scalar_values=())

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=NetManager.BATCH_SIZE,
                                  drop_last=True, shuffle=True, num_workers=NetManager.NUM_WORKER)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=NetManager.BATCH_SIZE,
                                  drop_last=True, shuffle=True, num_workers=NetManager.NUM_WORKER)
    module = Model()
    netmanage = NetTrainAndTest(net=module, train_loader=train_dataloader, valid_loader=valid_dataloader, test_loader=None)
    netmanage.train()



