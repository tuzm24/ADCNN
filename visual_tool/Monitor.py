import GPUtil
from threading import Thread
import time
from help_func.logging import LoggingHelper
from CfgEnv.loadCfg import NetManager

class MonitorGPU(Thread):
    def __init__(self, delay):
        super(MonitorGPU, self).__init__()
        self.stopped = False
        self.delay = delay
        self.logger = LoggingHelper.get_instance().logger
        try:
            self.GPU = GPUtil.getGPUs()
            self.writer = NetManager.writer
            # self.writerLayout = {'GPU':{'Load': ['Margin', ['gpuMonitor/load_gpu_' + str(i) for i in range(len(self.GPU))].append('gpuMonitor/load_gpu_total')],
            #                             'Memory': ['Margin', ['gpuMonitor/memory_gpu_' + str(i) for i in range(len(self.GPU))].append('gpuMonitor/memory_gpu_total')]}}
            self.loadLayout = {'GPU/Load' : dict.fromkeys('gpu_' + str(i) for i in range(len(self.GPU))),
                               'GPU/Memory' : dict.fromkeys('gpu_' + str(i) for i in range(len(self.GPU)))}
            self.loadLayout['GPU/Load']['gpu_total'] = None
            self.loadLayout['GPU/Memory']['gpu_total'] = None
        except Exception as e:
            self.logger.info(e)
            self.stopped = True
        self.start()

    def run(self):
        while not self.stopped:
            try:
                gpus = GPUtil.getGPUs()
                loadsum = 0
                memorysum = 0
                for i, gpu in enumerate(gpus):
                    self.loadLayout['GPU/Load']['gpu_'+ str(i)] = gpu.load
                    self.loadLayout['GPU/Memory']['gpu_'+ str(i)] = gpu.memoryUtil
                    loadsum += gpu.load
                    memorysum += gpu.memoryUtil
                self.loadLayout['GPU/Load']['gpu_total'] = loadsum
                self.loadLayout['GPU/Memory']['gpu_total'] = memorysum
                self.writer.add_scalars('GPU/Load', self.loadLayout['GPU/Load'], NetManager.step)
                self.writer.add_scalars('GPU/Memory', self.loadLayout['GPU/Memory'], NetManager.step)
                # self.writer.add_custom_scalars(layout=self.writerLayout)                           8
                time.sleep(self.delay)
            except Exception as e:
                self.logger.error('GPU logging Error : ' + str(e))
                self.stopped = True

    def stop(self):
        self.stopped = True



