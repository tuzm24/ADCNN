

import os

if '__main__' == __name__:

    os.chdir("../")
    print(os.getcwd())

from CfgEnv.loadCfg import NetManager
from help_func.CompArea import TuList
from help_func.CompArea import UnitBuf
# from help_func.CompArea import Component
from help_func.CompArea import ChromaFormat
from help_func.CompArea import PictureFormat
# from help_func.CompArea import UniBuf
from help_func.CompArea import Size
from help_func.CompArea import Area
from help_func.help_torch import torchUtil
from help_func.CompArea import LearningIndex
import numpy as np
import struct
import pandas as pd
# import random
import sklearn
# import threading
import copy
# import torch
# from CfgEnv.loadCfg import LoggingHelper


from help_func.help_python import myUtil

# from multiprocessing import Queue

class  DataBatch(NetManager):
    #istraining :: 0 : test, 1 : training, 2 : validation


    def __init__(self, istraining, batch_size = 1): #
        if istraining == LearningIndex.TRAINING:
            self.data_path = self.TRAINING_PATH
            self.csv_path = os.path.join(self.TRAINING_PATH, self.CSV_NAME)
        elif istraining == LearningIndex.VALIDATION:
            self.data_path = self.VALIDATION_PATH
            self.csv_path = os.path.join(self.VALIDATION_PATH, self.CSV_NAME)
        rs = np.random.RandomState(42)
        self.istraining = istraining
        self.batch_size = batch_size
        self.csv = pd.read_csv(self.csv_path)
        self.csv = self.csv.sample(frac=NetManager.cfg.USE_DATASET_SUBSET, random_state=rs)
        print("[%s DataNum : %d]" %(LearningIndex.INDEX_DIC[istraining], len(self.csv)))
        self.sizeDic = {}
        self.tulen = len(self.TU_ORDER)
        self.csv = self.csv.dropna(axis='columns')
        if istraining>0:
            self.csv = sklearn.utils.shuffle(self.csv, random_state=rs)
        self.batch = self.MakeBatch()
        self.batch_num = len(self.batch)
        print("[%s NumberOfBatchs : %d]" %(LearningIndex.INDEX_DIC[istraining], self.batch_num))
        self.batch = np.array(self.batch).reshape(-1, len(self.batch[0][0]))
        self.iter = 0
        # self.data = Queue()
        # t = threading.Thread(target=self.setNextData, daemon = True)
        # t.start()

    def MakeBatch(self):
        sizedic = {}
        batch_list = []
        for index, row in self.csv.iterrows():
            sizetuple = (row['HEIGHT'], row['WIDTH'])
            if sizetuple not in sizedic:
                self.SetSizeDic(sizetuple)
                sizedic[sizetuple] = [[]]
            sizedic[sizetuple][-1].append(row.values)
            if len(sizedic[sizetuple][-1]) == self.batch_size:
                sizedic[sizetuple].append(list())
        for v in sizedic.values():
            if len(v[-1]) < self.batch_size:
                del(v[-1])
            batch_list += v
        return batch_list

    def SetSizeDic(self, sizetuple):
        size = Size(sizetuple[1], sizetuple[0])
        area = size.getArea()
        carea = size.getCArea()
        datanum = 0
        split_bin = []
        for i in range(PictureFormat.MAX_NUM_COMPONENT):
            if self.PEL_DATA[i]:
                split_bin.append(area)
                datanum += area
                if not self.IS_ONLY_LUMA:
                   split_bin.append(carea)
                   split_bin.append(carea)
                   datanum += carea*2
            else:
                split_bin.append(0)
                if not self.IS_ONLY_LUMA:
                    split_bin.append(0)
                    split_bin.append(0)
        del split_bin[-1]
        split_bin = np.cumsum(split_bin)
        self.sizeDic[sizetuple] = (split_bin, '<'  + str(datanum) + 'h', datanum*2)
        return


    def getNextData(self):
        data = self.data.get()
        return data

    def setNextData(self):
        while True:
            for i in range(self.iter, self.iter + self.batch_size):
                pred, gt = self.unpackData(self.batch[i])
                self.data.put((pred, gt))
            self.iter +=self.batch_size
            while self.data.qsize() != 0:
                pass

    def isReadySetData(self):
        if self.data.qsize() == self.batch_size:
            return True
        return False


    # self.info - 0 : filename, 1 : width, 2: height, 3: qp, 4: mode, 5: depth ...
    # self : info, orgY, orgCb, orgCr, predY, predCb, predCr, reconY, reconCb, reconCr, unfiltredY, unfiltredCb, unfiltredCr
    # self.tulist
    def unpackData(self, info):
        self.info = info
        filepath = os.path.join(self.data_path, info[0])
        split_bin, strdatanum, shortdatanum = self.sizeDic[(info[2], info[1])]
        with open(filepath, 'rb') as data:
            self.orgY, self.orgCb, self.orgCr,\
            self.predY, self.predCb, self.predCr,\
            self.reconY, self.reconCb, self.reconCr,\
            self.unfilteredY, self.unfilteredCb, self.unfilteredCr\
                = np.split(np.array(struct.unpack(strdatanum, data.read(shortdatanum)),
                                    dtype='float32'), split_bin, axis=0)
            self.cwidth = self.info[1] // 2
            self.cheight = self.info[2] // 2
            if not self.IS_CONST_TU_DATA:
                self.tulist = TuList(np.array([[*info[:2], 0, 0, *info[3:] ]]))
            else:
                self.tulist = TuList.loadTuList(data)
            if self.IS_CONST_CTU_DATA:
                self.ctulist = TuList.loadTuList(data)


    def reshapeOrg(self):
        return self.orgY.reshape((self.info[2], self.info[1])), myUtil.UpSamplingChroma(
            self.orgCb.reshape((self.cheight, self.cwidth))), myUtil.UpSamplingChroma(
            self.orgCr.reshape((self.cheight, self.cwidth)))

    def reshapePred(self):
        return self.predY.reshape((self.info[2], self.info[1])), myUtil.UpSamplingChroma(
            self.predCb.reshape((self.cheight, self.cwidth))), myUtil.UpSamplingChroma(
            self.predCr.reshape((self.cheight, self.cwidth)))

    def reshapeRecon(self):
        return self.reconY.reshape((self.info[2], self.info[1])), myUtil.UpSamplingChroma(
            self.reconCb.reshape((self.cheight, self.cwidth))), myUtil.UpSamplingChroma(
            self.reconCr.reshape((self.cheight, self.cwidth)))

    def reshapeUnfiltered(self):

        return self.unfilteredY.reshape((self.info[2], self.info[1])), myUtil.UpSamplingChroma(
            self.unfilteredCb.reshape((self.cheight, self.cwidth))), myUtil.UpSamplingChroma(
            self.unfilteredCr.reshape((self.cheight, self.cwidth)))

    def dropPadding(self, x, pad, isDeepCopy = False):
        if isDeepCopy:
            return copy.deepcopy(x[:,pad:-pad,pad:-pad])
        else:
            return x[:,pad:-pad,pad:-pad]

    def TFdropPadding(self, x, pad, isDeepCopy = False):
        if isDeepCopy:
            return copy.deepcopy(x[pad:-pad,pad:-pad,:])
        else:
            return x[pad:-pad,pad:-pad,:]

    def loadMeanStd(self, loader, isGetNew = False):
        mean = 0
        std = 0
        self.mean = 0
        self.std = 1
        if not isGetNew:
            if self.cfg.isExist('DATAMEAN'):
                mean = self.cfg.DATAMEAN
            if self.cfg.isExist('DATASTD'):
                std = self.cfg.DATASTD
            if not mean or not std:
                print('There is no mean or standard deviation data present.')
                mean, std = torchUtil.online_mean_and_sd(loader)
            elif len(mean)!=self.data_channel_num or len(std)!=self.data_channel_num:
                print('The mean and std already exist and the number\
                 of channels in the current data does not match.')
                mean, std = torchUtil.online_mean_and_sd(loader)
            else:
                for i, s in enumerate(std):
                    if not s:
                        std[i] = mean[i]*10
                return (np.array(list(mean))), (np.array(list(std)))
        else:
            mean, std = torchUtil.online_mean_and_sd(loader)
        print('Save mean and std')
        self.cfg.member['DATAMEAN'] = mean.numpy().tolist()
        self.cfg.member['DATASTD'] = std.numpy().tolist()
        self.cfg.write_yml()
        self.mean = mean.numpy().reshape((len(mean), 1 , 1))
        self.std = std.reshape((len(std), 1 , 1))
        return (mean.numpy()), (std.numpy())



class TestDataBatch(NetManager):
    def __init__(self):
        self.batch = myUtil.getleafDirs(self.TEST_PATH)
        self.sizeDic = {}
        self.SetSizeDic()
        self.batch_num = len(self.batch)

    def SetSizeDic(self):
        for folderpath in self.batch:
            csv = pd.read_csv(os.path.join(folderpath, self.CSV_NAME)).dropna(axis='columns')
            for index, row in csv.iterrows():
                sizetuple = (row['HEIGHT'], row['WIDTH'])
                if sizetuple not in self.sizeDic:
                    size = Size(sizetuple[1], sizetuple[0])
                    area = size.getArea()
                    carea = size.getCArea()
                    datanum = 0
                    split_bin = []
                    for i in range(PictureFormat.MAX_NUM_COMPONENT):
                        if self.PEL_DATA[i]:
                            split_bin.append(area)
                            datanum += area
                            if not self.IS_ONLY_LUMA:
                               split_bin.append(carea)
                               split_bin.append(carea)
                               datanum += carea*2
                        else:
                            split_bin.append(0)
                            if not self.IS_ONLY_LUMA:
                                split_bin.append(0)
                                split_bin.append(0)
                    del split_bin[-1]
                    split_bin = np.cumsum(split_bin)
                    self.sizeDic[sizetuple] = (split_bin, '<'  + str(datanum) + 'h', datanum*2)
        return
    @staticmethod
    def appendNone(num, mlist):
        for _ in range(num):
            mlist.append(None)
        return

    #NAME,WIDTH,HEIGHT,X_POS,Y_POS,QP,MODE,DEPTH,HOR_TR,VER_TR
    def unpackData(self, testFolderPath):
        # self.cur_path = testFolderPath
        csv = pd.read_csv(os.path.join(testFolderPath, self.CSV_NAME)).dropna(axis='columns').values
        # width = np.max(csv[:,1].astype('int32') + csv[:, 3].astype('int32'))
        # height = np.max(csv[:,2].astype('int32') + csv[:, 4].astype('int32'))
        # cwidth = width//2
        # cheight = height//2
        # pic = []
        # for i in range(PictureFormat.MAX_NUM_COMPONENT):
        #     if self.PEL_DATA[i]:
        #         pic.append(np.zeros((height, width)))
        #         if not self.IS_ONLY_LUMA:
        #             pic.append(np.zeros(cheight, cwidth))
        #             pic.append(np.zeros(cheight, cwidth))
        #         else:
        #             self.appendNone(2, pic)
        #     else:
        #         self.appendNone(3, pic)
        # picarea = Area(width, height, 0, 0)
        # self.pic = UnitBuf(ChromaFormat.YCbCr4_2_0, picarea,*pic)
        # self.tulist = TuList(None)
        # self.ctulist = TuList(None)
        for info in csv:
            filepath = os.path.join(testFolderPath, info[0])
            split_bin, strdatanum, shortdatanum = self.sizeDic[(info[2], info[1])]
            with open(filepath, 'rb') as data:
                pels = np.split(np.array(struct.unpack(strdatanum, data.read(shortdatanum)),
                                        dtype='float32'), split_bin, axis=0)
                pelarea = Area(*info[1:5])
                self.pic = UnitBuf(ChromaFormat.YCbCr4_2_0, pelarea, *pels)
                # self.pic.CopyAll(pels)
                if not self.IS_CONST_TU_DATA:
                    self.tulist = TuList(np.array([[*info[:2], 0, 0, *info[3:]]]))
                else:
                    self.tulist = TuList.loadTuList(data)
                if self.IS_CONST_CTU_DATA:
                    self.ctulist = TuList.loadTuList(data)

    def dropPadding(self, x, pad, isDeepCopy = False):
        if isDeepCopy:
            return copy.deepcopy(x[:,pad:-pad,pad:-pad])
        else:
            return x[:,pad:-pad,pad:-pad]

if '__main__' == __name__:
    df = DataBatch(2, 3)
    df.unpackData(df.batch[0])
    # df = TestDataBatch()
    # df.unPackOneFrameTestSet(df.testfolders[0])