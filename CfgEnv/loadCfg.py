from CfgEnv.config import Config

import os
from torch.cuda import is_available as checkGPU




class NetManager(object):
    #dataset
    #logging
    # logger = LoggingHelper.get_instance().logger

    #config load
    cfg = Config("config.yml", None)
        #path
    DATASET_PATH = cfg.DATASET_PATH
    TENSORBOARD_PATH = cfg.TENSORBOARD_PATH
    TRAINING_PATH = os.path.join(DATASET_PATH, 'Training')
    VALIDATION_PATH = os.path.join(DATASET_PATH, 'Validation')
    TEST_PATH = os.path.join(DATASET_PATH, 'TEST')
    CSV_NAME = cfg.CSV_NAME
    MODEL_PATH = cfg.SAVE_PATH
    os.makedirs(MODEL_PATH, exist_ok=True)
        #Framework
    IS_TENSORFLOW = cfg.FRAMEWORK
        #tensorboard
    # TENSORBOARD_BASIC_PLOT = ['Prediction', 'Reconstruction', 'Unfiltered', 'CNN_Recon']
    # TB_PLOT_LIST = [TENSORBOARD_BASIC_PLOT[i] for i in cfg.PLOT_LIST]
        #netInfo
    OBJECT_EPOCH = cfg.OBJECT_EPOCH
    SET_NEW_MEAN_STD = cfg.SET_NEW_MEAN_STD
    #PELDATA
    PEL_DATA = cfg.PEL_DATA # Original, Prediction, Reconstruction, UnfiltredReconstruction
    BIT_DEPTH = cfg.BIT_DEPTH
    IS_ONLY_LUMA = cfg.IS_ONLY_LUMA


    #TUDATA
    IS_CONST_TU_DATA = cfg.IS_CONST_TU_DATA
    IS_CONST_CTU_DATA = cfg.IS_CONST_CTU_DATA
    TU_ORDER = cfg.TU_ORDER

    PRINT_PERIOD = cfg.PRINT_PERIOD
    BATCH_SIZE = cfg.BATCH_SIZE

    TEST_BY_BLOCKED = cfg.TEST_BY_BLOCKED
    TEST_BY_BLOCKED_WIDTH = cfg.TEST_BY_BLOCKED_WIDTH
    TEST_BY_BLOCKED_HEIGHT = cfg.TEST_BY_BLOCKED_HEIGHT

    #tensorboard
    if cfg.isExist('step'):
        step = cfg.step
    else:
        step = 0
        cfg.member['step'] = 0
    object_step = 0
    print('Call NetManager')
    #TRAINING_CFG
    NUM_WORKER = cfg.NUM_WORKER
    RESULT_DIR = cfg.RESULT_DIR

    if not checkGPU():
        PRINT_PERIOD = 5
        BATCH_SIZE = 6
        cfg.USE_DATASET_SUBSET = 1.0
        NUM_WORKER = 1
        OBJECT_EPOCH = 1



