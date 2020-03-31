import logging
import os
import time
from help_func.__init__ import ExecFileName

def get_str_time():
    return time.strftime('%a %d %b %Y, %Hh%Mm%S', time.localtime(time.time()))



class LoggingHelper(object):
    INSTANCE = None
    filename = ExecFileName.filename
    p_start_time = time.time()
    # if LoggingHelper.INSTANCE is not None:
    #     raise ValueError("An instantiation already exists!")
    name = ''
    if filename is not None:
        name = filename + '_'
    os.makedirs("./logs", exist_ok=True)
    logger = logging.getLogger()

    logging.basicConfig(filename='./logs/' + name + get_str_time() + '_LOGGER_basic.log', level=logging.INFO)

    fileHandler = logging.FileHandler("./logs/"+ name + get_str_time() + '_msg.log')
    streamHandler = logging.StreamHandler()

    fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    fileHandler.setFormatter(fomatter)
    streamHandler.setFormatter(fomatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)


    def __init__(self):
        pass
    @classmethod
    def get_instance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = LoggingHelper()
        return cls.INSTANCE

    @staticmethod
    def diff_time_logger(messege, start_time):
        LoggingHelper.get_instance().logger.info("[{}] :: running time {}".format(messege, time.time() - start_time))



    def log_cur_time(self):
        self.logger.info("Clock : %s", self.get_str_time())

    def log_diff_start_time(self):
        self.logger.info("Elapsed Time : %a %m %b %Y, %Hh%Mm%S", time.localtime(time.time() - self.p_start_time))