import os
import time
import subprocess
import struct
import numpy as np
from PIL import Image
import math
from threading import Thread
import threading
import queue
from help_func.logging import LoggingHelper
from CfgEnv.config import Config
from help_func.help_python import myUtil
import csv
import copy
import random

from help_func.CompArea import TuList
from help_func.CompArea import UnitBuf
from help_func.CompArea import Component
from help_func.CompArea import ChromaFormat
from help_func.CompArea import PictureFormat
from help_func.CompArea import UniBuf
from help_func.CompArea import Area

if __name__ == '__main__':
    os.chdir("../")


class BuildData(object):
    COLOR_BLACK = [0, 512, 512]
    COLOR_RED = [304, 336, 1020]
    COLOR_GREEN = np.array([149, 43, 21], dtype='int16') << 2
    COLOR_BLUE = np.array([29, 255, 107], dtype='int16') << 2
    logger = LoggingHelper.get_instance().logger
    configpath = './build_data/data_config.yml'
    cfg = Config(configpath, logger)
    os.makedirs(cfg.DATASET_PATH, exist_ok=True)
    os.makedirs(cfg.TEMP_PATH, exist_ok=True)
    os.makedirs(cfg.DECODE_LOG_PATH, exist_ok=True)
    dataset_path = os.path.join(cfg.DATASET_PATH, cfg.DATASET_NAME)
    if not cfg.ADDITION_DATA:
        os.makedirs(dataset_path)
    trainingset_path = os.path.join(dataset_path, cfg.TRAININGSET_PATH)
    validation_path = os.path.join(dataset_path, cfg.VALIDATIONSET_PATH)
    testset_path = os.path.join(dataset_path, cfg.TESTSET_PATH)
    sample_path = os.path.join(dataset_path, cfg.SAMPLE_IMAGE_PATH)
    csv_name = cfg.CSV_NAME
    others_data = cfg.TU_DATA_OTHERS
    tu_data_num = len(others_data) + 4

    # Data Format Config
    isOnlyLuma = cfg.ONLY_LUMA
    usePelDataList = cfg.PEL_DATA
    # tu_map = cfg.TU_MAP

    # assert len(tu_map) == len(others_data)

    training_datacnt = 0
    validation_datacnt = 0
    # test_datacnt = 0

    training_datacnt += myUtil.xgetFileNum(trainingset_path)
    validation_datacnt += myUtil.xgetFileNum(validation_path)
    # test_datacnt += myUtil.xgetFileNum(testset_path)

    os.makedirs(trainingset_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(testset_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    temp_path = cfg.TEMP_PATH
    depth = cfg.DECODER_BIT_DEPTH
    if depth == 10:
        datatype = 'int16'  # Output Data Type
    else:
        datatype = 'uint8'

    # per_training = 100 - (cfg.VALIDATION_SPLIT * 100) // 1
    target_test_POC = cfg.TARGET_TESTSET_NUM

    splitmode = cfg.IS_TU_SPLIT
    # others setting
    qpList = cfg.QP
    mode = cfg.MODE
    intraList = cfg.INTRA_MODE
    interList = cfg.INTER_MODE
    combineList = intraList + interList

    # data shape setting
    use_const_block = cfg.USE_CONST_BLOCK
    const_width = cfg.CONST_WIDTH
    const_height = cfg.CONST_HEIGHT
    save_tu_const_block_info = cfg.SAVE_TU_CONST_BLOCK_INFO

    get_testset_by_picture = cfg.GET_TESTSET_BY_PICTURE


    # Tu Shape
    min_width = cfg.MIN_WIDTH_SIZE
    max_width = cfg.MAX_WIDTH_SIZE
    min_height = cfg.MIN_HEIGHT_SIZE
    max_height = cfg.MAX_HEIGHT_SIZE
    only_square = cfg.ONLY_SQUARE

    # padding setting
    boundary_pad = cfg.PIC_BOUNDARY_PAD
    luma_pad = ((cfg.LUMA_PAD[0], cfg.LUMA_PAD[1]),
                (cfg.LUMA_PAD[2], cfg.LUMA_PAD[3]))
    chroma_pad = ((cfg.CHROMA_PAD[0], cfg.CHROMA_PAD[1]),
                  (cfg.CHROMA_PAD[2], cfg.CHROMA_PAD[3]))
    pad_opt = [luma_pad, chroma_pad, chroma_pad]

    if splitmode < 2:
        min_width += cfg.LUMA_PAD[2] + cfg.LUMA_PAD[3]
        max_width += cfg.LUMA_PAD[2] + cfg.LUMA_PAD[3]
        min_height += cfg.LUMA_PAD[0] + cfg.LUMA_PAD[1]
        max_height += cfg.LUMA_PAD[0] + cfg.LUMA_PAD[1]

    else:
        min_width += cfg.CHROMA_PAD[2] + cfg.CHROMA_PAD[3]
        max_width += cfg.CHROMA_PAD[2] + cfg.CHROMA_PAD[3]
        min_height += cfg.CHROMA_PAD[0] + cfg.CHROMA_PAD[1]
        max_height += cfg.CHROMA_PAD[0] + cfg.CHROMA_PAD[1]
    getopt = cfg.OPT
    comp_opt = cfg.COMPARE_OPT
    yuv_opt = cfg.YUV_OPT

    ctulist = cfg.CTU_DTA_OTHERS
    ctu_data_num = len(ctulist)
    # training_tu_data = queue.Queue()
    # validation_tu_data = queue.Queue()
    test_data_order = ['NAME', 'WIDTH', 'HEIGHT', 'X_POS', 'Y_POS']
    test_data_order += others_data

    csv_header = ['NAME', 'WIDTH', 'HEIGHT'] + others_data
    training_csv_path = os.path.join(trainingset_path, csv_name)
    validation_csv_path = os.path.join(validation_path, csv_name)

    myUtil.initHeaderCSV(training_csv_path, csv_header)
    myUtil.initHeaderCSV(validation_csv_path, csv_header)

    training_file = open(training_csv_path, 'a', newline='')
    validation_file = open(validation_csv_path, 'a', newline='')
    training_csv = csv.writer(training_file)
    validation_csv = csv.writer(validation_file)

    if use_const_block:
        splitmode = 0


class imgInfo(BuildData):
    thlock = threading.Lock()

    # os.makedirs(os.path.join(TrainingSetPath, '32x32'), exist_ok=True)
    # os.makedirs(os.path.join(TestSetPath, '32x32'), exist_ok=True)
    def __init__(self, name, data_opt, targetnum):
        self.example_image_get = False
        binpath = "./ywkim_" + name + ".bin"
        self.name = name
        self.num = 0
        self.img = open(binpath, 'rb')
        self.qp = struct.unpack('B', self.img.read(1))[0]
        self.width = struct.unpack('<h', self.img.read(2))[0]
        self.height = struct.unpack('<h', self.img.read(2))[0]
        self.POC = struct.unpack('<h', self.img.read(2))[0]
        self.ofPOC = (targetnum + self.POC) // self.POC
        self.cwidth = (int)(self.width / 2)
        self.cheight = (int)(self.height / 2)
        self.area = self.width * self.height
        self.carea = self.cwidth * self.cheight
        self.pelCumsum = []
        if self.cfg.IS_ONLY_ONE_INTRA and random.randrange(0,30)%29 != 0:
            self.example_image_get = True
        for _ in range(len(self.usePelDataList)):
            self.pelCumsum.append(self.area)
            self.pelCumsum.append(self.carea)
            self.pelCumsum.append(self.carea)
        self.totalpels = np.sum(self.pelCumsum)
        if len(self.pelCumsum):
            self.pelCumsum = np.cumsum(np.array(self.pelCumsum), dtype='int32')
        self.dataopt = data_opt  # 0 : test, 1 : training, 2 : validation

    # def Check_Cfg_Setting(self):
    #     if self.boundary_pad !=2 and (self.luma_pad !=0 or self.chroma_pad !=0):
    #         logger.error("PIC_BOUNDARY_PAD is must '2', if you want use pad data")

    def getPSNR(self, org, control):
        if self.depth == 10:
            MAX_VALUE = 1023.0
        else:
            MAX_VALUE = 255.0
        try:
            mse = np.square(np.subtract(org, control)).mean()
            if mse == 0:
                return 100
            return 10 * math.log10((MAX_VALUE * MAX_VALUE) / mse)
        except:
            msum = np.array([])
            for i in range(len(org)):
                msum = np.concatenate((msum, np.square(np.subtract(org[i], control[i])).flatten()), axis=0)
            return 10 * math.log10((MAX_VALUE * MAX_VALUE) / msum.mean())

    def GetFilecnt(self, path):
        return len(os.listdir(path))

    def getTrainingset(self):
        self.logger.info("%s binfile get training set.." % self.name)
        if self.cfg.SKIP_TU_DEPENDENT_QP == 0:
            if self.qp not in self.qpList:
                self.logger.info("  Not matched qp..")
                self.logger.info("  Skip binfile..")
        for poc in range(self.POC):
            # st_time = time.time()
            # Pic = []  # piclist {0:Original, 1:prediction, 2:reconstruction, 3:unfiltered }
            # PicUV = []
            # pocMode = struct.unpack('<h', self.img.read(2))[0]
            if self.ctu_data_num > 0:
                CTUInfo = self.getCTUInfo()
            YSplitInfo = self.getTuInfo()
            CSplitInfo = self.getTuInfo()
            # print("Until Make Pic&TU : %s", time.time() - st_time)
            # self.imgUnpack(Pic)
            # print("Until Make Pic&TU : %s", time.time() - st_time)
            pic = self.imgUnpack()
            # tumaps = self.getTuMap(YSplitInfo = YSplitInfo,
            #                       CSplitInfo = CSplitInfo)

            if np.sum(pic.original[0]) == 0:
                self.logger.error('   %s poc %s Original is zero' % (self.name, poc))
                return
            try:
                self.logger.info("  %s POC %s" % (self.name, poc))
                self.logger.debug("  %s Prediction PSNR [YUV : %s], [Y : %s], [U : %s], [V : %s]"
                                  % (
                                      self.name, self.getPSNR(pic.original, pic.prediction),
                                      self.getPSNR(pic.original[0], pic.prediction[0]),
                                      self.getPSNR(pic.original[1], pic.prediction[1]),
                                      self.getPSNR(pic.original[2], pic.prediction[2])))

                self.logger.debug("  %s Reconstruction PSNR [YUV : %s], [Y : %s], [U : %s], [V : %s]"
                                  % (
                                      self.name, self.getPSNR(pic.original, pic.reconstruction),
                                      self.getPSNR(pic.original[0], pic.reconstruction[0]),
                                      self.getPSNR(pic.original[1], pic.reconstruction[1]),
                                      self.getPSNR(pic.original[2], pic.reconstruction[2])))

                self.logger.debug("  %s Unfiltered PSNR [YUV : %s], [Y : %s], [U : %s], [V : %s]"
                                  % (
                                      self.name, self.getPSNR(pic.original, pic.unfilteredRecon),
                                      self.getPSNR(pic.original[0], pic.unfilteredRecon[0]),
                                      self.getPSNR(pic.original[1], pic.unfilteredRecon[1]),
                                      self.getPSNR(pic.original[2], pic.unfilteredRecon[2])))
            except Exception as e:
                self.logger.error(e)
                self.logger.error('   %s poc %s cannot calc PSNR' % (self.name, poc))
                return
            # if self.mode or self.mode != pocMode:
            #     self.logger.info("  Skip POC...")
            #     continue
            # print("Until Make Calc PSNR : %s", time.time() - st_time)

            if self.use_const_block:
                cbi = self.cfg.CONST_BLOCK_INTERVAL
                pos_list = [
                    np.array((self.const_width if (x + self.const_width) <= self.width else (self.width - x),
                              self.const_height if (y + self.const_height) <= self.height else (self.height - y),
                              x, y, self.qp))
                    for y in range(0, ((self.height-1)//self.const_height)*self.const_height, self.const_height)
                    for x in range(0, ((self.width-1)//self.const_width)*self.const_width, self.const_width)]

                pos_list = TuList(np.array(pos_list).T)
                # pic.tulist = copy.deepcopy(YSplitInfo)
            elif self.splitmode == 1:
                pos_list = copy.deepcopy(YSplitInfo)
                # pic.tulist = copy.deepcopy(YSplitInfo)
            elif self.splitmode == 2:
                CSplitInfo[2:, :] = CSplitInfo[2:, :] << 1
                pos_list = copy.deepcopy(CSplitInfo)
                # pic.tulist = copy.deepcopy(CSplitInfo)
            else:
                self.logger.error("  Config IS_TU_SPLIT is Unknown")
                return

            # 0: width, 1: height, 2: x_pos 3: y_pos
            # pos_list = np.array(pos_list)
            if not self.example_image_get:
                real_pos_list = copy.deepcopy(pos_list)
                if self.splitmode < 2:
                    real_pos_list.tulist[2] += self.cfg.LUMA_PAD[2]
                    real_pos_list.tulist[3] += self.cfg.LUMA_PAD[0]
                else:
                    real_pos_list.tulist[2] += self.cfg.CHROMA_PAD[2]
                    real_pos_list.tulist[3] += self.cfg.CHROMA_PAD[0]

            pad_width = 0
            pad_height = 0

            if self.boundary_pad != 2:
                for i in range(4):
                    for j in range(3):
                        if self.boundary_pad != 1:
                            pic.pelBuf[i][j] = np.pad(pic.pelBuf[i][j], self.pad_opt[j]
                                                      # + ((0,0),)
                                                      , mode='constant', constant_values=(self.boundary_pad))
                        else:
                            pic.pelBuf[i][j] = np.pad(pic.pelBuf[i][j], self.pad_opt[j]
                                                      # + ((0,0),)
                                                      , mode='edge')
                if self.splitmode < 2:  # split mode is luma or constant
                    pad_width = self.cfg.LUMA_PAD[2] + self.cfg.LUMA_PAD[3]
                    pad_height = self.cfg.LUMA_PAD[0] + self.cfg.LUMA_PAD[1]
                else:
                    pad_width = self.cfg.CHROMA_PAD[2] + self.cfg.CHROMA_PAD[3]
                    pad_height = self.cfg.CHROMA_PAD[0] + self.cfg.CHROMA_PAD[1]
                pos_list.tulist[0] += pad_width
                pos_list.tulist[1] += pad_height

                if self.ctu_data_num>0:
                    CTUInfo.tulist[0] += pad_width
                    CTUInfo.tulist[1] += pad_height

                if self.use_const_block and self.save_tu_const_block_info:
                    if self.save_tu_const_block_info % 2 == 1:
                        YSplitInfo.tulist[2] += self.cfg.LUMA_PAD[2]
                        YSplitInfo.tulist[3] += self.cfg.LUMA_PAD[0]
                        YSplitInfo.tulist[0, YSplitInfo.tulist[2] == self.cfg.LUMA_PAD[2]] += self.cfg.LUMA_PAD[2]
                        YSplitInfo.tulist[2, YSplitInfo.tulist[2] == self.cfg.LUMA_PAD[2]] = 0
                        YSplitInfo.tulist[1, YSplitInfo.tulist[3] == self.cfg.LUMA_PAD[0]] += self.cfg.LUMA_PAD[0]
                        YSplitInfo.tulist[3, YSplitInfo.tulist[3] == self.cfg.LUMA_PAD[0]] = 0
                        YSplitInfo.tulist[0, (YSplitInfo.tulist[2] + YSplitInfo.tulist[0]) == (self.width - 1)] += \
                        self.cfg.LUMA_PAD[3]
                        YSplitInfo.tulist[1, (YSplitInfo.tulist[3] + YSplitInfo.tulist[1]) == (self.height - 1)] += \
                        self.cfg.LUMA_PAD[1]

                    if self.save_tu_const_block_info > 1:
                        CSplitInfo.tulist[2] += self.cfg.CHROMA_PAD[2]
                        CSplitInfo.tulist[3] += self.cfg.CHROMA_PAD[0]
                        CSplitInfo.tulist[0, CSplitInfo.tulist[2] == self.cfg.CHROMA_PAD[2]] += self.cfg.CHROMA_PAD[2]
                        CSplitInfo.tulist[2, CSplitInfo.tulist[2] == self.cfg.CHROMA_PAD[2]] = 0
                        CSplitInfo.tulist[1, CSplitInfo.tulist[3] == self.cfg.CHROMA_PAD[0]] += self.cfg.CHROMA_PAD[0]
                        CSplitInfo.tulist[3, CSplitInfo.tulist[3] == self.cfg.CHROMA_PAD[0]] = 0
                        CSplitInfo.tulist[0, (CSplitInfo.tulist[2] + CSplitInfo.tulist[0]) == (self.width - 1)] += \
                        self.cfg.CHROMA_PAD[3]
                        CSplitInfo.tulist[1, (CSplitInfo.tulist[3] + CSplitInfo.tulist[1]) == (self.height - 1)] += \
                        self.cfg.CHROMA_PAD[1]

            # print("Until Tu Padding : %s", time.time() - st_time)
            # pos_list = pos_list.transpose((1,0))

            if not self.use_const_block:
                if len(self.combineList) != 0:
                    pos_list.tulist = pos_list.tulist[:, np.isin(pos_list.tulist[5], self.combineList)]

            # 0: width, 1: height, 2: x_pos 3: y_pos
            # tulist cut
            if self.cfg.SKIP_TU_DEPENDENT_QP == 1:
                pos_list.tulist = pos_list.tulist[:, np.isin(pos_list.tulist[4], self.qpList)]

            pos_list.tulist = pos_list.tulist[:, pos_list.tulist[0] >= self.min_width + pad_width]
            pos_list.tulist = pos_list.tulist[:, pos_list.tulist[0] <= self.max_width + pad_width]
            pos_list.tulist = pos_list.tulist[:, pos_list.tulist[1] >= self.min_height + pad_height]
            pos_list.tulist = pos_list.tulist[:, pos_list.tulist[1] <= self.max_height + pad_height]
            if self.only_square:
                pos_list.tulist = pos_list.tulist[:, pos_list.width == pos_list.height]

            try:
                if len(pos_list.tulist[0]) == 0:
                    self.logger.info("  No match block in POC about config..")
                    self.logger.info("  Skip POC...")
                    return
            except:
                self.logger.error("  No match block in POC about config..")
                self.logger.error("  Skip POC...")
                return

            # 0: width, 1: height, 2: x_pos 3: y_pos, 4 : qp, 5 : mode
            perm = np.arange(pos_list.tulist.shape[1])
            np.random.shuffle(perm)
            pos_list.tulist = pos_list.tulist[:, perm]
            # np.random.shuffle(pos_list)
            pos_list.resetMember()
            if len(pos_list.tulist[0]) <= self.ofPOC:
                pass
            elif self.getopt == 0:
                pos_list.tulist = pos_list.tulist[:, :self.ofPOC]
            else:
                if self.getopt == 1:
                    prob_list = pos_list.width * pos_list.height
                else:
                    prob_list = []
                    for pos in pos_list.tulist:
                        if self.yuv_opt == 0:
                            se = (np.square(
                                pic.original[0][pos[1]:pos[1], pos[0]:pos[0]]
                                - pic.pelBuf[self.comp_opt][0][pos[1]:pos[1], pos[0]:pos[0]]).sum()
                                  + np.square(
                                        pic.original[1:][pos[1]:pos[1], pos[0]:pos[0]]
                                        - pic.pelBuf[self.comp_opt][1:][pos[1]:pos[1], pos[0]:pos[0]]).sum())
                        elif self.yuv_opt == 1:
                            se = np.square(
                                pic.original[0][pos[1]:pos[1], pos[0]:pos[0]]
                                - pic.pelBuf[self.comp_opt][0][pos[1]:pos[1], pos[0]:pos[0]])
                        else:
                            se = np.square(
                                pic.original[1:][pos[1]:pos[1], pos[0]:pos[0]]
                                - pic.pelBuf[self.comp_opt][1:][pos[1]:pos[1], pos[0]:pos[0]])
                        if self.getopt == 2:
                            se = se.mean()
                        else:  # self.getopt ==3:
                            se = se.sum()
                        prob_list.append(se)
                    prob_list = np.array(prob_list)
                pos_list.tulist = pos_list.tulist[:,
                                  np.random.choice(np.arange(len(prob_list)),
                                                   self.ofPOC, p=prob_list / prob_list.sum())]

            # 0: width, 1: height, 2: x_pos 3: y_pos, 4 : qp, 5 : mode
            pos_list.resetMember()
            # print("Until Tu Extracting : %s", time.time() - st_time)
            for block in pos_list.tulist.T:
                self.thlock.acquire()
                if self.dataopt == 1:  # training
                    BuildData.training_datacnt += 1
                    block_path = os.path.join(self.trainingset_path, str(BuildData.training_datacnt) + '.bin')
                    self.training_csv.writerow([str(BuildData.training_datacnt) + '.bin', *block[:2], *block[4:]])
                else:
                    BuildData.validation_datacnt += 1
                    block_path = os.path.join(self.validation_path, str(BuildData.validation_datacnt) + '.bin')
                    self.validation_csv.writerow([str(BuildData.validation_datacnt) + '.bin', *block[:2], *block[4:]])
                self.thlock.release()
                blockdata = (block[3], block[3] + block[1], block[2], block[2] + block[0],
                             block[3] >> 1, (block[1] >> 1) + (block[3] >> 1),
                             block[2] >> 1, (block[0] >> 1) + (block[2] >> 1))
                imgdata = []
                for (i, isUsePel) in enumerate(self.usePelDataList):
                    if isUsePel:
                        imgdata.append(self.getBlockArea(pic.pelBuf[i], blockdata=blockdata))
                imgdata = np.concatenate(imgdata, axis=0)
                with open(block_path, 'wb') as f:
                    f.write(struct.pack('<' + str(len(imgdata)) + 'h', *imgdata))
                    if self.use_const_block and self.save_tu_const_block_info:
                        if self.save_tu_const_block_info % 2:
                            TuList(YSplitInfo.containTuList(Area(*block[:4]))).saveTuList(f)
                        if self.save_tu_const_block_info > 1:
                            TuList(CSplitInfo.containTuList(Area(*block[:4]))).saveTuList(f)
                    if self.ctu_data_num > 0:
                        TuList(CTUInfo.containTuList(Area(*block[:4]))).saveTuList(f)

            # print("Save Tu : %s", time.time() - st_time)
            if self.example_image_get == False:
                self.example_image_get = True
                self.logger.info("  Save to Image : %s", self.name)
                self.saveImage(pic.original, real_pos_list.tulist, pos_list.tulist, 'orig_')
                self.saveImage(pic.prediction, real_pos_list.tulist, pos_list.tulist, 'pred_')
                self.saveImage(pic.reconstruction, real_pos_list.tulist, pos_list.tulist, 'recon_')
                self.saveImage(pic.unfilteredRecon, real_pos_list.tulist, pos_list.tulist, 'unfiltered_')

            self.logger.info("  POC_%d finished(%s Cnt : %d)" % (poc, self.name, len(pos_list.tulist[0])))
        self.img.close()
        os.remove('./ywkim_' + self.name + '.bin')

    def getBlockArea(self, pic, blockdata):
        y = pic[0][blockdata[0]:blockdata[1],
            blockdata[2]: blockdata[3]].flatten()
        u = pic[1][blockdata[4]:blockdata[5], blockdata[6]:blockdata[7]].flatten()
        v = pic[2][blockdata[4]:blockdata[5], blockdata[6]:blockdata[7]].flatten()
        return np.concatenate((y, u, v), axis=0).astype(self.datatype)

    def getTuData(self, tumap, blockArea):
        return tumap[blockArea[0]:blockArea[1],
               blockArea[2]:blockArea[3]].flatten().astype(self.datatype)

    def getTestSet(self):
        self.logger.info("%s binfile get test set.." % self.name)
        test_seq_path = os.path.join(self.testset_path, self.name)
        os.makedirs(test_seq_path, exist_ok=True)
        for poc in range(self.POC):
            # st_time = time.time()
            if self.target_test_POC != 0 and poc >= self.target_test_POC:
                self.logger.info("  %s POC is break(POC : %s)" % (self.name, poc))
                break
            test_poc_path = os.path.join(test_seq_path,
                                         'poc' + str(poc) + '_' + str(self.height) + '_' + str(self.width))
            test_poc_csv_path = os.path.join(test_poc_path, self.csv_name)
            os.makedirs(test_poc_path, exist_ok=True)
            myUtil.initHeaderCSV(test_poc_csv_path, self.test_data_order)
            test_poc_file = open(test_poc_csv_path, 'a', newline='')
            test_csv = csv.writer(test_poc_file)
            # test_poc_data_path = os.path.join(test_poc_path, 'control')
            # other_poc_data_path = os.path.join(test_poc_path, 'others')
            # os.makedirs(test_poc_data_path, exist_ok=True)
            # os.makedirs(other_poc_data_path, exist_ok=True)

            # Pic = []  # piclist {0:Original, 1:prediction, 2:reconstruction, 3:unfiltered }
            # PicUV = []
            # pocMode = struct.unpack('<h', self.img.read(2))[0]
            if self.ctu_data_num > 0:
                CTUInfo = self.getCTUInfo()
            YSplitInfo = self.getTuInfo()
            CSplitInfo = self.getTuInfo()

            # tu_map = self.getTuMap(YSplitInfo, CSplitInfo)

            # print("Until Make Pic&TU : %s", time.time() - st_time)
            pic = self.imgUnpack()
            # print("Until Make Pic&TU : %s", time.time() - st_time)
            if np.sum(pic.original[0]) == 0:
                self.logger.error('   %s poc %s Original is zero' % (self.name, poc))
                return
            try:
                self.logger.info("  %s POC %s" % (self.name, poc))
                self.logger.debug("  %s Prediction PSNR [YUV : %s], [Y : %s], [U : %s], [V : %s]"
                                  % (
                                      self.name, self.getPSNR(pic.original, pic.prediction),
                                      self.getPSNR(pic.original[0], pic.prediction[0]),
                                      self.getPSNR(pic.original[1], pic.prediction[1]),
                                      self.getPSNR(pic.original[2], pic.prediction[2])))

                self.logger.debug("  %s Reconstruction PSNR [YUV : %s], [Y : %s], [U : %s], [V : %s]"
                                  % (
                                      self.name, self.getPSNR(pic.original, pic.reconstruction),
                                      self.getPSNR(pic.original[0], pic.reconstruction[0]),
                                      self.getPSNR(pic.original[1], pic.reconstruction[1]),
                                      self.getPSNR(pic.original[2], pic.reconstruction[2])))

                self.logger.debug("  %s Unfiltered PSNR [YUV : %s], [Y : %s], [U : %s], [V : %s]"
                                  % (
                                      self.name, self.getPSNR(pic.original, pic.unfilteredRecon),
                                      self.getPSNR(pic.original[0], pic.unfilteredRecon[0]),
                                      self.getPSNR(pic.original[1], pic.unfilteredRecon[1]),
                                      self.getPSNR(pic.original[2], pic.unfilteredRecon[2])))
            except Exception as e:
                self.logger.error(e)
                self.logger.error('   %s poc %s cannot calc PSNR' % (self.name, poc))
                return

            if self.use_const_block:
                pos_list = [
                    np.array((self.const_width if (x + self.const_width) <= self.width else (self.width - x),
                              self.const_height if (y + self.const_height) <= self.height else (self.height - y),
                              x, y, self.qp))
                    for y in range(0, self.height-1, self.const_height) for x in range(0, self.width-1, self.const_width)]
                pos_list = np.array(pos_list).T

                pos_list = TuList(pos_list)
                # pic.tulist = copy.deepcopy(YSplitInfo)
            elif self.splitmode == 1:
                pos_list = copy.deepcopy(YSplitInfo)
                # pic.tulist = copy.deepcopy(YSplitInfo)
            elif self.splitmode == 2:
                CSplitInfo[2:, :] = CSplitInfo[2:, :] << 1
                pos_list = copy.deepcopy((CSplitInfo))
                # pic.tulist = copy.deepcopy(CSplitInfo)
            else:
                self.logger.error("  Config IS_TU_SPLIT is Unknown")
                return

            if not self.example_image_get:
                real_pos_list = copy.deepcopy(pos_list)
                if self.splitmode < 2:
                    real_pos_list.tulist[2] += self.cfg.LUMA_PAD[2]
                    real_pos_list.tulist[3] += self.cfg.LUMA_PAD[0]
                else:
                    real_pos_list.tulist[2] += self.cfg.CHROMA_PAD[2]
                    real_pos_list.tulist[3] += self.cfg.CHROMA_PAD[0]

            pad_width = 0
            pad_height = 0

            if self.boundary_pad != 2:
                for i in range(4):
                    for j in range(3):
                        if self.boundary_pad != 1:
                            pic.pelBuf[i][j] = np.pad(pic.pelBuf[i][j], self.pad_opt[j]
                                                      # + ((0,0),)
                                                      , mode='constant', constant_values=(self.boundary_pad))
                        else:
                            pic.pelBuf[i][j] = np.pad(pic.pelBuf[i][j], self.pad_opt[j]
                                                      # + ((0,0),)
                                                      , mode='edge')
                if self.splitmode < 2:  # split mode is luma or constant
                    pad_width = self.cfg.LUMA_PAD[2] + self.cfg.LUMA_PAD[3]
                    pad_height = self.cfg.LUMA_PAD[0] + self.cfg.LUMA_PAD[1]
                else:
                    pad_width = self.cfg.CHROMA_PAD[2] + self.cfg.CHROMA_PAD[3]
                    pad_height = self.cfg.CHROMA_PAD[0] + self.cfg.CHROMA_PAD[1]
                pos_list.tulist[0] += pad_width
                pos_list.tulist[1] += pad_height

                if self.ctu_data_num>0:
                    CTUInfo.tulist[0] += pad_width
                    CTUInfo.tulist[1] += pad_height

                if self.use_const_block and self.save_tu_const_block_info:
                    if self.save_tu_const_block_info % 2 == 1:
                        YSplitInfo.tulist[2] += self.cfg.LUMA_PAD[2]
                        YSplitInfo.tulist[3] += self.cfg.LUMA_PAD[0]
                        YSplitInfo.tulist[0, YSplitInfo.tulist[2] == self.cfg.LUMA_PAD[2]] += self.cfg.LUMA_PAD[2]
                        YSplitInfo.tulist[2, YSplitInfo.tulist[2] == self.cfg.LUMA_PAD[2]] = 0
                        YSplitInfo.tulist[1, YSplitInfo.tulist[3] == self.cfg.LUMA_PAD[0]] += self.cfg.LUMA_PAD[0]
                        YSplitInfo.tulist[3, YSplitInfo.tulist[3] == self.cfg.LUMA_PAD[0]] = 0
                        YSplitInfo.tulist[0, (YSplitInfo.tulist[2] + YSplitInfo.tulist[0]) == (self.width - 1)] += \
                        self.cfg.LUMA_PAD[3]
                        YSplitInfo.tulist[1, (YSplitInfo.tulist[3] + YSplitInfo.tulist[1]) == (self.height - 1)] += \
                        self.cfg.LUMA_PAD[1]

                    if self.save_tu_const_block_info > 1:
                        CSplitInfo.tulist[2] += self.cfg.CHROMA_PAD[2]
                        CSplitInfo.tulist[3] += self.cfg.CHROMA_PAD[0]
                        CSplitInfo.tulist[0, CSplitInfo.tulist[2] == self.cfg.CHROMA_PAD[2]] += self.cfg.CHROMA_PAD[2]
                        CSplitInfo.tulist[2, CSplitInfo.tulist[2] == self.cfg.CHROMA_PAD[2]] = 0
                        CSplitInfo.tulist[1, CSplitInfo.tulist[3] == self.cfg.CHROMA_PAD[0]] += self.cfg.CHROMA_PAD[0]
                        CSplitInfo.tulist[3, CSplitInfo.tulist[3] == self.cfg.CHROMA_PAD[0]] = 0
                        CSplitInfo.tulist[0, (CSplitInfo.tulist[2] + CSplitInfo.tulist[0]) == (self.width - 1)] += \
                        self.cfg.CHROMA_PAD[3]
                        CSplitInfo.tulist[1, (CSplitInfo.tulist[3] + CSplitInfo.tulist[1]) == (self.height - 1)] += \
                        self.cfg.CHROMA_PAD[1]

            condition_arr = []

            condition_arr.append(np.ones(len(pos_list.tulist[0])))
            if self.splitmode != 0:
                if len(self.combineList) != 0:
                    condition_arr.append(np.isin(pos_list.tulist[5], self.combineList))

            if self.cfg.SKIP_TU_DEPENDENT_QP == 1:
                condition_arr.append(np.isin(pos_list.tulist[0], self.qpList))

            if not self.use_const_block:
                condition_arr.append(pos_list.tulist[0] >= self.min_width + pad_width)
                condition_arr.append(pos_list.tulist[0] <= self.max_width + pad_width)
                condition_arr.append(pos_list.tulist[1] >= self.min_height + pad_height)
                condition_arr.append(pos_list.tulist[1] <= self.max_height + pad_height)

            # mask = (pos_list.tulist[0] + pos_list.tulist[2]) > self.width
            # condition_arr.append(~mask)
            # pos_list.tulist[0, mask] = pos_list.tulist[0, mask] - (self.width - pos_list.tulist[2, mask])
            # mask = (pos_list.tulist[1] + pos_list.tulist[3]) > self.height
            # condition_arr.append(~mask)
            # pos_list.tulist[1, mask] = pos_list.tulist[1, mask] - (self.height - pos_list.tulist[3, mask])
            if self.only_square:
                condition_arr.append(pos_list.width == pos_list.height)
            try:
                if len(pos_list.tulist[0]) == 0:
                    self.logger.info("  No match block in POC about config..")
                    self.logger.info("  Skip POC...")
                    return
            except:
                self.logger.error("  No match block in POC about config..")
                self.logger.error("  Skip POC...")
                return
            condition_arr = np.all(condition_arr, axis=0)
            pos_list.tulist = np.concatenate((pos_list.tulist, condition_arr[np.newaxis, :]), axis=0)

            if self.get_testset_by_picture:
                pos_list.tulist = np.array([[np.max(pos_list.tulist[0] + pos_list.tulist[2])], [np.max(pos_list.tulist[1] + pos_list.tulist[3])],
                                            [0], [0], [self.qp], [1]])

            pos_list.resetMember()
            for idx, block in enumerate(pos_list.tulist.T):

                block_name = str(block[3]) + '_' + str(block[2]) \
                             + '_' + str(block[1]) + '_' + str(block[0]) + '.bin'
                block_path = os.path.join(test_poc_path, block_name)

                test_csv.writerow([block_name, *block])

                blockdata = (block[3], block[3] + block[1], block[2], block[2] + block[0],
                             block[3] >> 1, (block[1] >> 1) + (block[3] >> 1),
                             block[2] >> 1, (block[0] >> 1) + (block[2] >> 1))
                # qp_mode = np.array((block[:2]), dtype= self.datatype)
                imgdata = []
                for (i, isUsePel) in enumerate(self.usePelDataList):
                    if isUsePel:
                        imgdata.append(self.getBlockArea(pic.pelBuf[i], blockdata=blockdata))
                imgdata = np.concatenate(imgdata, axis=0)
                with open(block_path, 'wb') as f:
                    f.write(struct.pack('<' + str(len(imgdata)) + 'h', *imgdata))
                    if self.use_const_block and self.save_tu_const_block_info:
                        if self.save_tu_const_block_info % 2:
                            TuList(YSplitInfo.containTuList(Area(*block[:4]))).saveTuList(f)
                        if self.save_tu_const_block_info > 1:
                            TuList(CSplitInfo.containTuList(Area(*block[:4]))).saveTuList(f)
                    if self.ctu_data_num > 0:
                        TuList(CTUInfo.containTuList(Area(*block[:4]))).saveTuList(f)
            if self.example_image_get == False:
                self.example_image_get = True
                self.logger.info("  Save to Image : %s", self.name)
                self.saveImage(pic.original, real_pos_list.tulist, pos_list.tulist, 'orig_', select_mark=False)
                self.saveImage(pic.prediction, real_pos_list.tulist, pos_list.tulist, 'pred_', select_mark=False)
                self.saveImage(pic.reconstruction, real_pos_list.tulist, pos_list.tulist, 'recon_', select_mark=False)
                self.saveImage(pic.unfilteredRecon, real_pos_list.tulist, pos_list.tulist, 'unfiltered_',
                               select_mark=False)
        self.img.close()
        os.remove('./ywkim_' + self.name + '.bin')

        # def SaveByTIFF(self, YUV, name):
        #     y = YUV[0]
        #     u = self.UpSamplingChroma(YUV[1])
        #     v = self.UpSamplingChroma(YUV[2])
        #     tiff = TIFF.open(name, mode = 'w')
        #     tiff.write_image([y,u,v], write_rgb=True)
        #     return

    def imgUnpack(self):
        if self.depth == 10:
            endian = '<'
            fmt = 'h'
            perbyte = 2
        else:  # bitdepth is 8
            endian = ''
            fmt = 'B'
            perbyte = 1
        image = np.array(struct.unpack(endian + str(self.totalpels) + fmt,
                                       self.img.read((perbyte) * self.totalpels)),
                         dtype='int16')
        image = np.split(image, self.pelCumsum)
        buflist = []
        for (i, isUse) in enumerate(self.usePelDataList):
            for ch in range(Component.MAX_NUM_COMPONENT):
                if isUse and not (ch and self.isOnlyLuma):
                    if not ch:
                        buflist.append(image[i * Component.MAX_NUM_COMPONENT + ch].reshape((self.height, self.width)))
                    else:
                        buflist.append(image[i * Component.MAX_NUM_COMPONENT + ch].reshape((self.cheight, self.cwidth)))
                else:
                    buflist.append(None)

        return UnitBuf(ChromaFormat.YCbCr4_2_0, Area(self.width, self.height, 0, 0), *buflist)

        # imgY = np.array(struct.unpack(endian + str(self.area) + fmt, self.img.read((perbyte) * self.area)),
        #                 dtype='float32').reshape((self.height, self.width))
        # # .reshape((self.height, self.width, 1))
        # imgUV = np.array(struct.unpack(endian + str(2 * self.carea) + fmt, self.img.read((perbyte) * 2 * self.carea)),
        #                  dtype='float32').reshape((2, self.cheight, self.cwidth)).transpose((1, 2, 0))

    def getTuInfo(self):
        tuCnt = struct.unpack('<i', self.img.read(4))[0]
        # tulist = []
        # print(tuCnt)
        strTuCnt = '<' + str(tuCnt * self.tu_data_num) + 'h'
        tudatanum_mul_two = self.tu_data_num * 2
        # for i in range(tuCnt):
        tulist = np.array(struct.unpack(strTuCnt, self.img.read(tudatanum_mul_two * tuCnt)), dtype='int16').reshape(
            (-1, self.tu_data_num)).transpose(
            (1, 0))
        # tu1 = (struct.unpack('2B', self.img.read(2)))
        # tu2 = (struct.unpack('<4h', self.img.read(8)))
        # tulist.append(np.concatenate((tu1, tu2), axis=0).astype('int16'))
        return TuList(tulist)

    def getCTUInfo(self):
        ctuCnt = struct.unpack('<i', self.img.read(4))[0]
        strCTUCnt = '<' + str(ctuCnt * (self.ctu_data_num + 4)) + 'h'
        tudatanum_mul_two = (self.ctu_data_num + 4) * 2
        ctulist = np.array(struct.unpack(strCTUCnt, self.img.read(tudatanum_mul_two * ctuCnt)),
                           dtype='int16').reshape((-1, (self.ctu_data_num + 4))).transpose(
            (1, 0))
        return TuList(ctulist)

    def DownSamplingLuma(self, lumaPic):
        pos00 = lumaPic[::2, ::2, :]
        pos10 = lumaPic[1::2, ::2, :]
        return (pos00 + pos10 + 1) >> 1

    def UpSamplingChroma(self, UVPic):
        return UVPic.repeat(2, axis=0).repeat(2, axis=1)

    def saveImage(self, YUV, real_tulist, candi_list, name, select_mark=True):
        uv = self.UpSamplingChroma(np.concatenate((YUV[1].reshape(1, YUV[1].shape[0], YUV[1].shape[1]),
                                                   YUV[2].reshape(1, YUV[2].shape[0], YUV[2].shape[1])),
                                                  axis=0).transpose((1, 2, 0)))
        y = YUV[0].reshape((1, YUV[0].shape[0], YUV[0].shape[1])).transpose((1, 2, 0))
        YUV = np.concatenate((y, uv), axis=2)
        if self.depth == 10:
            YUV = np.uint8(YUV >> 2)
        else:
            YUV = np.uint8(YUV)

        # 0: width, 1: height, 2: x_pos 3: y_pos, 4 : qp, 5 : mode
        for tu in real_tulist.T:
            YUV[tu[3]:tu[3] + tu[1], tu[2], 0] = 0
            YUV[tu[3], tu[2]:tu[2] + tu[0], 0] = 0
            if tu[2]+tu[0] <YUV.shape[1] and tu[3] + tu[1] < YUV.shape[0]:
                YUV[tu[3] + tu[1], tu[2]:tu[2] + tu[0], 0] = 0
                YUV[tu[3]:tu[3] + tu[1], tu[2] + tu[0], 0] = 0
        if select_mark:
            for tu in candi_list.T:
                YUV[tu[3]:tu[3] + tu[1], tu[2]:tu[2] + tu[0], 1] = self.COLOR_BLUE[1]
                YUV[tu[3]:tu[3] + tu[1], tu[2]:tu[2] + tu[0], 2] = self.COLOR_BLUE[2]

        rgbimg = Image.fromarray(YUV, 'YCbCr')
        rgbimg.convert('RGBA').save(os.path.join(self.sample_path, name + self.name + ".png"))
        # self.logger.info("%s image file save." % self.name)
        # plt.imshow(rgbimg)
        # plt.show()
        # rgbimg.show()


class SplitManager(BuildData):
    corelock = threading.Lock()

    def __init__(self):
        self.cores = 0
        self.corenum = self.cfg.PARALLEL_DECODE
        self.filelist = self.getFileListsFromList(self.cfg.TRAINING_BIN_PATH)
        self.testlist = self.getFileListsFromList(self.cfg.TEST_BIN_PATH)
        self.validationlist = self.getFileListsFromList(self.cfg.VALIDATION_BIN_PATH)
        self.orglist = self.getFileListsFromList(self.cfg.TRAINING_ORG_PATH, pattern='.yuv')
        self.testOrgList = self.getFileListsFromList(self.cfg.TEST_ORG_PATH, pattern='.yuv')
        self.validationOrgList = self.getFileListsFromList(self.cfg.VALIDATION_ORG_PATH, pattern='.yuv')
        self.logpath = self.cfg.DECODE_LOG_PATH
        os.makedirs(self.logpath, exist_ok=True)
        # self.myftp = self.tryReconnectFTP()
        # self.seq = self.initSeqeuences()

    def getFileListsFromList(self, list, pattern='.bin'):
        filelist = []
        for files in list:
            filelist += myUtil.getFileList(files, pattern)
        return filelist

    def OnlyOneIntra(self, filepath, yuvpath):
        commands = []
        orgdic = {}
        filelist = self.getFileListsFromList(filepath, pattern='.bin')
        orglist = self.getFileListsFromList(yuvpath, pattern='.yuv')

        for org in orglist:
            orgdic['PNG' + str(os.path.basename(org).split('_')[0])] = org
        for file in filelist:
            if os.path.basename(file).split('_')[1] in orgdic:
                command = self.cfg.DECODER_PATH + ' -b ' + file + ' -i ' + orgdic[
                    str(os.path.basename(file).split('_')[1])] + ' -bd ' + '8'
                commands.append((os.path.basename(file), command))
        return commands

    def initCommand(self, filelist, orglist):
        seqs = []
        bdDic = {}
        # frameDic = {}
        # framenum = 0
        with open("./build_data/Sequences.csv", 'r') as reader:
            # with open("./SequenceSetting/CTCSequences.csv", 'r') as reader:
            data = reader.read()
            lines = data.strip().split('\n')
            for line in lines:
                seqs.append(line.split(','))
        seqs = seqs[1:]
        for seq in seqs:
            # if seq[0].split("_")[0].lower() != "netflix":
            bdDic[seq[0].lower()] = seq[2]
            # else:
            #
            #     bdDic[''.join(seq[0].split('_')[1:]).lower()] = seq[2]
            # frameDic[seq[0].lower()] = seq[8]
        commands = []
        for org in orglist:
            tmp = str(os.path.basename(org).split(".")[0])
            if tmp.split("_")[0].lower() != "netflix":
                tmp = '_'.join(tmp.split("_")[:3]).lower()
            else:
                tmp = '_'.join(tmp.split("_")[:4]).lower()
            for binfile in filelist:
                if binfile.split('_')[1] == 'PNG' and tmp == binfile.split('_')[2]:
                    command = self.cfg.DECODER_PATH + ' -b ' + binfile + ' -i ' + org + ' -bd 8'
                    commands.append((os.path.basename(binfile), command))
                    # framenum +=1
                else:
                    if os.path.basename(binfile).lower().split('_')[1] != 'netflix':
                        if '_'.join(os.path.basename(binfile).lower().split("_")[1:4]).lower() == tmp:
                            # command = Decoderpath + ' -b ' + binfile + ' -i ' + org + ' -bd ' + bdDic[tmp]
                            command = self.cfg.DECODER_PATH + ' -b ' + binfile + ' -i ' + org + ' -bd ' + bdDic[tmp]
                            commands.append((os.path.basename(binfile), command))
                    else:
                        if '_'.join(os.path.basename(binfile).lower().split("_")[1:5]).lower() == tmp:
                            # command = Decoderpath + ' -b ' + binfile + ' -i ' + org + ' -bd ' + bdDic[tmp]
                            command = self.cfg.DECODER_PATH + ' -b ' + binfile + ' -i ' + org + ' -bd ' + bdDic[tmp]
                            commands.append((os.path.basename(binfile), command))
                            # framenum += frameDic[tmp]
        # return commands, framenum
        return commands

    def runDecoder(self, command):
        (name, command) = command
        logpath = name.replace(".bin", ".log")
        logpath = os.path.join(self.logpath, logpath)
        self.logger.info(command)
        self.logger.info("%s start" % name)
        with open(logpath, 'w') as fp:
            sub_proc = subprocess.Popen(command, stdout=fp)
            sub_proc.wait()
        self.logger.info("%s done" % name)
        return name.replace(".bin", "")

    def runThreading(self, command, isTraining, targetnum):
        name = self.runDecoder(command)
        splitimg = imgInfo(name, isTraining, targetnum)
        if isTraining:
            splitimg.getTrainingset()
        else:
            splitimg.getTestSet()
        self.corelock.acquire()
        self.cores -= 1
        self.corelock.release()

    # def CalculateTargetNum(self, cur_num, target_num, remain):
    #     return (target_num - cur_num) //  remain

    def getTrainingset(self):
        if not len(self.filelist):
            return
        targetnum = self.cfg.TARGET_DATASET_NUM // len(self.filelist)  # 각 bin file에서 데이터를 몇개 뽑을 것인지 체크
        if self.cfg.IS_ONLY_ONE_INTRA:
            commands = self.OnlyOneIntra(self.cfg.TRAINING_PNG_PATH, self.cfg.PNG_ORG_PATH)
        else:
            commands = self.initCommand(self.filelist, self.orglist)
        for command in commands:
            while self.cores >= self.corenum:
                time.sleep(3)
            self.corelock.acquire()
            self.cores += 1
            self.corelock.release()
            t = Thread(target=self.runThreading, args=(command, 1, targetnum))
            t.start()
        return

    def getValidationset(self):
        if not len(self.validationlist):
            return
        targetnum = self.cfg.VALIDATION_DATSET_NUM // len(self.validationlist)  # 각 bin file에서 데이터를 몇개 뽑을 것인지 체크

        if self.cfg.IS_ONLY_ONE_INTRA:
            commands = self.OnlyOneIntra(self.cfg.VALIDATION_PNG_PATH, self.cfg.PNG_ORG_PATH)
        else:
            commands = self.initCommand(self.validationlist, self.validationOrgList)
        for command in commands:
            while self.cores >= self.corenum:
                time.sleep(3)
            self.corelock.acquire()
            self.cores += 1
            self.corelock.release()
            t = Thread(target=self.runThreading, args=(command, 2, targetnum))
            t.start()
        return

    def getTestset(self):
        if not len(self.testlist):
            return
        targetnum = self.cfg.TARGET_TESTSET_NUM // len(self.testlist)  # 각 bin file에서 데이터를 몇개 뽑을 것인지 체크
        commands = self.initCommand(self.testlist, self.testOrgList)
        for command in commands:
            while self.cores >= self.corenum:
                time.sleep(3)
            self.cores += 1
            t = Thread(target=self.runThreading, args=(command, 0, targetnum))
            t.start()

        return


if __name__ == '__main__':
    # os.chdir("../")
    print(os.getcwd())
    sp = SplitManager()
    sp.getTrainingset()
    sp.getTestset()
    sp.getValidationset()
    # st = time.time()
    # fsize = os.path.getsize('./ywkim_AI_MAIN10_FoodMarket4_3840x2160_60_10b_S02_27.bin')
    # im =  open('./ywkim_AI_MAIN10_CatRobot1_3840x2160_60_10b_S04_22.bin', 'rb')
    # img = im.read(fsize)
    # print(time.time()-st)
    # splitimg = imgInfo("AI_MAIN10_CatRobot1_3840x2160_60_10b_S04_22", 1, 50)
    # splitimg.getTrainingset()
    # splitimg.getTestSet()
    # sp.getTestset()
    # sp.getTestset()
# binpath1 = "./bin_AI_MAIN10_BQTerrace_1920x1080_60_S11_32.bin_Q32_Split.bin"
# img = imgInfo(name=binpath1,targetnum=1000)
# img.getTrainingset()
