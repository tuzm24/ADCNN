from CfgEnv.loadCfg import NetManager
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import math
from PIL import Image
import torch


class Mytensorboard(NetManager):
    INSTANCE = None

    def __init__(self, comment=''):
        self.writer = SummaryWriter(comment='_' + comment)
        self.writerLayout = {'Loss': {},
                             'PSNR': {}}
        self.step = 0

    @classmethod
    def get_instance(cls, comment=''):
        if cls.INSTANCE is None:
            cls.INSTANCE = Mytensorboard(comment=comment)
        return cls.INSTANCE

    def plotToTensorboard(self, fig, name):
        self.writer.add_figure(name, fig, global_step=self.step, close=True)

    def imgToTensorboard(self, img, name):
        # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8 or use tensorflow
        self.writer.add_image(name, img, global_step=self.step)

    def batchImageToTensorBoard(self, recon, resi, name):
        if recon is not None:
            img = (recon.cpu().detach().numpy() + resi.cpu().detach().numpy()) * 255.0
        else:
            img = (resi.cpu().detach().numpy()) * 255.0
        img = np.clip(img, 0, 255).astype(int)
        self.writer.add_image(name, img, global_step=self.step)

    def SaveImageToTensorBoard(self, name, image):
        image = np.clip(image * 255.0, 0, 255).astype(int)
        self.writer.add_image(name, image, global_step=self.step)

    def saveImageFromTest(self, recon, resi, name):
        img = (recon.cpu().detach().numpy() + resi.cpu().detach().numpy()) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img[0, 0])
        img.save(name + 'png')

    @staticmethod
    def Makegrid(imgs, nrow=None):
        if nrow is None:
            nrow = math.ceil(math.sqrt(imgs.shape[0]))
        return vutils.make_grid(imgs, nrow)

    def setObjectStep(self, num_set):
        self.object_step = num_set * self.OBJECT_EPOCH

    def plotScalars(self):
        for key, values in self.writerLayout.items():
            self.writer.add_scalars(key, values, self.step)

    def plotDifferent(self, img, name, percentile=90):
        if isinstance(img, torch.Tensor):
            img = (img.cpu().detach().numpy()) * 1023.0
        else:
            img = img * 1023.0
        percentile = percentile + (100 - percentile) // 2
        img = np.clip(img,
                      np.percentile(img, 100 - percentile, interpolation='higher'),
                      np.percentile(img, percentile, interpolation='lower'))
        img = np.clip(img, -1023.0, 1023.0)
        fig, ax = plt.subplots()
        if img.min() < 0 and img.max() > 0:
            mymax = max(abs(img.min()), img.max())
            mymin = -mymax
        else:
            mymin = img.min()
            mymax = img.max()
        imgs = ax.imshow((img[0]).astype(int), vmin=mymin, vmax=mymax, interpolation='nearest',
                         cmap=plt.cm.get_cmap('seismic'))
        v1 = np.linspace(mymin, mymax, 10, endpoint=True)
        cb = fig.colorbar(imgs, ticks=v1)
        cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
        self.plotToTensorboard(fig, name + '_percentile' + str(percentile))
        return

    def plotMSEImage(self, resi, name):
        img = ((resi.cpu().detach().numpy()) ** 2) * 1023.0 * 1023.0
        fig, ax = plt.subplots()
        if img.min() < 0 and img.max() > 0:
            mymax = max(abs(img.min()), img.max())
            mymin = -mymax
        else:
            mymin = img.min()
            mymax = img.max()
        imgs = ax.imshow((img[0]).astype(int), vmin=mymin, vmax=mymax, interpolation='nearest',
                         cmap=plt.cm.get_cmap('seismic'))
        v1 = np.linspace(mymin, mymax, 10, endpoint=True)
        cb = fig.colorbar(imgs, ticks=v1)
        cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
        self.plotToTensorboard(fig, name)
        return

    def plotMAEImage(self, resi, name, percentile=90):
        img = (resi.cpu().detach().numpy()) * 1023.0
        np.abs(img, out=img)
        fig, ax = plt.subplots()
        if percentile < 100:
            np.clip(img,
                    0,
                    np.percentile(img, percentile, interpolation='lower'), out=img)
        mymin = img.min()
        mymax = img.max()
        imgs = ax.imshow((img[0]).astype(int), vmin=mymin, vmax=mymax, interpolation='nearest',
                         cmap=plt.cm.get_cmap('seismic'))
        v1 = np.linspace(mymin, mymax, 10, endpoint=True)
        cb = fig.colorbar(imgs, ticks=v1)
        cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
        self.plotToTensorboard(fig, name + '_percentile' + str(percentile))
        return

    """
    Use to plot input data 2d
    vminmax is range as list or tuple, example [22, 37]
    for example is qp is 22, 27, 32, 37
    plotMap(qpmap, 'QPMap', vminmax = [22, 37], color_num = 4)
    """

    def plotMap(self, img, name, vminmax=None, color_num=None):
        if vminmax is None:
            vminmax = (img.min().cpu(), img.max().cpu())

        img = self.Makegrid(img)
        fig, ax = plt.subplots()
        img = img.cpu()

        if color_num is None:
            color_num = len(img.unique())
        imgs = ax.imshow((img.numpy()[0]).astype(int), vmin=vminmax[0], vmax=vminmax[1], interpolation='nearest',
                         cmap=plt.cm.get_cmap('viridis', color_num))
        v1 = np.round(np.linspace(vminmax[0], vminmax[1], 10, endpoint=True))
        cb = fig.colorbar(imgs, ticks=v1)
        cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
        self.plotToTensorboard(fig, name)
        return

    def SetLoss(self, name, value):
        self.writerLayout['Loss'][name] = value

    def SetPSNR(self, name, value):
        self.writerLayout['PSNR'][name] = value

    def SetLearningRate(self, value):
        self.writer.add_scalars('LearningRate', {'lr': value}, self.step)

[0,	1,	5,	10,	30,	70,	100,	125,	150]
[39.42,	39.42,	39.42,	39.42,	39.42,	39.42,	39.42,	39.42,	39.42]
[0,	39.4,	39.43,	39.53,	39.54,	39.57,	39.57,	39.58,	39.59]
[0,	39.43,	39.49,	39.51,	39.51,	39.53,	39.54,	39.55,	39.555]
