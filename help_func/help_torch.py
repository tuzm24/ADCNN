import torch
from help_func.logging import LoggingHelper
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CfgEnv.loadCfg import NetManager
from collections import OrderedDict
from help_func.help_torch_parallel import DataParallelModel, DataParallelCriterion
import torch.optim as optim
from visual_tool.Tensorboard import Mytensorboard
from itertools import cycle
from help_func.help_python import myUtil
from help_func.warmup_scheduler import GradualWarmupScheduler
# from copy import deepcopy
# from torchsummary import summary
from help_func.my_torchsummary import summary_to_file
import os
import pickle
from help_func.CompArea import LearningIndex

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}

class torchUtil:

    @staticmethod
    def online_mean_and_sd(loader):
        """Compute the mean and sd in an online fashion

            Var[x] = E[X^2] - E^2[X]
        """
        input_channel = loader.dataset.input_data_channel
        print('Calculating data mean and std')
        cnt = 0
        fst_moment = torch.empty(input_channel).float().to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        snd_moment = torch.empty(input_channel).float().to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        for _, data, _ in tqdm(loader):
            b, c, h, w = data.shape
            data = data.to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            nb_pixels = b * h * w
            sum_ = torch.sum(data, dim=[0, 2, 3])
            sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels
        print('Finish calculate data mean and std')
        return fst_moment.cpu(), torch.sqrt(snd_moment - fst_moment ** 2).cpu()

    @staticmethod
    def Calc_Pearson_Correlation(loader, dataidx, opt='mean'):
        def maxCountValue(arr1d):
            brr, idxs = np.unique(arr1d, return_counts=True)
            return brr[np.argmax(idxs)]

        print('Calculating data mean and std')
        dataidx +=3
        x = []
        y = []
        z = []
        for _, data, gt in tqdm(loader):
            b, c, h, w = data.shape
            data = data.to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            gt = gt.to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            gtmse = torch.mean(gt[:,0,:,:]**2, dim = [1,2])
            x.append(gtmse.view(-1).cpu().numpy())
            if opt =='mean':
                datamean = torch.mean(data[:,dataidx[0],:,:], dim=[1,2])
                y.append(datamean.view(-1).cpu().numpy())
            if opt == 'max':
                data = data[:,dataidx[1],:,:].cpu().numpy().reshape((b,-1))
                y.append(np.apply_along_axis(maxCountValue, 1, data))
            if opt =='mean':
                datamean = torch.mean(data[:,dataidx[1],:,:], dim=[1,2])
                z.append(datamean.view(-1).cpu().numpy())

        y = np.array(y).reshape(-1)
        x = np.array(x).reshape(-1)
        z = np.array(z).reshape(-1)
        return x, y, z

    @staticmethod
    def _RoundChannels(c, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
        if new_c < 0.9 * c:
            new_c += divisor
        return new_c

    @staticmethod
    def _SplitChannels(channels, num_groups):
        split_channels = [channels // num_groups for _ in range(num_groups)]
        split_channels[0] += channels - sum(split_channels)
        return split_channels

    @staticmethod
    def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU', padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def UnPixelShuffle(x, r):
        b, c, h, w = x.shape
        out_channel = c * (r ** 2)
        out_h = h // r
        out_w = w // r
        return x.contiguous().view(b, c, out_h, r, out_w, r).permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)


    @staticmethod
    def psnr(mse):
        if mse <= 0:
            return 100
        return torch.log10(1 / mse) * 10

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out



class WeightedMSELoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, weight):
        super(WeightedMSELoss, self).__init__()
        self.weight = [x/np.sum(weight) for x in weight]
    def forward(self, input, target):
        return torch.sum(torch.stack([F.mse_loss(input[:,x:x+1,...], target[:,x:x+1,...], reduction=self.reduction) * self.weight[x]
                                       for x in range(len(self.weight))]))

class WeightedL1Loss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, weight):
        super(WeightedL1Loss, self).__init__()
        self.weight = [x/np.sum(weight) for x in weight]
    def forward(self, input, target):
        return torch.sum(torch.stack([F.l1_loss(input[:,x:x+1,...], target[:,x:x+1,...], reduction=self.reduction) * self.weight[x]
                                       for x in range(len(self.weight))]))


class result_collection:
    def __init__(self, istraining):
        self.istraining = istraining
        self.loss = []
        self.mse = []
        self.recon_mse = []
        self.result_dic = {'loss':self.loss, 'mse':self.mse, 'recon_mse':self.recon_mse}
        for x in self.result_dic.values():
            x.append(list())


    def __len__(self):
        return len(self.loss)



    def strlastResult(self, epoch):
        return  '({})[Epoch : {:>4}] Loss : {:k>8.7f}, ' \
                'MSE : {:k>8.7f}, PSNR : {:k>5.4f}, ' \
                'gtPSNR : {:k>5.4f}'.format(LearningIndex.INDEX_DIC[self.istraining],epoch, self.loss[-1],
                                           self.mse[-1], myUtil.psnr(self.mse[-1]),
                                           myUtil.psnr(self.recon_mse[-1]))


    def endofEpoch(self, logger, tb, epoch):
        for x in self.result_dic.values():
            x[-1] = np.mean(x[-1])
        self.printAndPlot(logger, tb, epoch)
        self.loss.append(list())
        self.mse.append(list())

    def printAndPlot(self, logger, tb, epoch):
        learning_str = LearningIndex.INDEX_DIC[self.istraining]
        logger.info(self.strlastResult(epoch))
        tb.step = epoch
        tb.SetLoss('{}_CNN_LOSS'.format(learning_str), self.loss[-1])
        tb.SetMSE('{}_VVC-Inloop_MSE'.format(learning_str), self.recon_mse[-1])
        tb.SetMSE('{}_CNN_MSE'.format(learning_str), self.mse[-1])
        tb.SetPSNR('{}_VVC-Inloop_PSNR'.format(learning_str), myUtil.psnr(self.recon_mse[-1]))
        tb.SetPSNR('{}_CNN_PSNR'.format(learning_str), myUtil.psnr(self.mse[-1]))
        tb.plotScalars()

    def save(self, path):
        with open(path+'_'+LearningIndex.INDEX_DIC[self.istraining], 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path, istraining):
        with open(path+'_'+LearningIndex.INDEX_DIC[istraining], 'rb') as f:
            return pickle.load(f)





class NetTrainAndTest:
    def __init__(self, net, train_loader, valid_loader, test_loader, mainloss = 'wl1', opt = 'adam', gpunum = None):

        self.logger = LoggingHelper.get_instance().logger
        self.net = net
        self.name = NetManager.NET_NAME
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # self.data_padding = self.train_loader.dataset.data_padding
        if train_loader is not None:
            self.data_channel_num = self.train_loader.dataset.input_data_channel
            self.output_channel_num = self.train_loader.dataset.output_data_channel
        else:
            self.data_channel_num = self.test_loader.dataset.input_data_channel
            self.output_channel_num = self.test_loader.dataset.output_data_channel

        self.train_batch_num, self.train_iter = self.getBatchNumAndCycle(self.train_loader)
        self.valid_batch_num, self.valid_iter = self.getBatchNumAndCycle(self.valid_loader)
        self.test_batch_num, self.test_iter = self.getBatchNumAndCycle(self.test_loader)
        self.criterion = self.setloss(mainloss, [8,1,1])
        self.ResultMSELoss = self.setloss('wl2', [4,1,1])
        self.GTMSELoss = self.setloss('wl2', [4,1,1])
        self.setGPUnum(gpunum)
        if self.iscuda:
            self.GTMSELoss = self.GTMSELoss.cuda()
            if self.cuda_device_count>1:
                # self.net = DataParallelModel(self.net).cuda()
                self.criterion = DataParallelCriterion(self.criterion).cuda()
                self.ResultMSELoss = DataParallelCriterion(self.ResultMSELoss).cuda()
            else:
                # self.net = self.net.cuda()
                self.criterion = self.criterion.cuda()
                self.ResultMSELoss = self.ResultMSELoss.cuda()
        self.optimizer = self.setopt(self.net.parameters(), NetManager.cfg.INIT_LEARNING_RATE*0.1, opt)
        # self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
        #                                               milestones=[int(NetManager.cfg.OBJECT_EPOCH * 0.5),
        #                                                           int(NetManager.cfg.OBJECT_EPOCH * 0.75)],
        #                                               gamma=0.1, last_epoch=-1)
        self.tb = Mytensorboard.get_instance(self.name)
        # summary_to_file(self.net,
        #                 (self.data_channel_num, NetManager.TEST_BY_BLOCKED_HEIGHT, NetManager.TEST_BY_BLOCKED_WIDTH),
        #                 os.path.join(self.tb.writer.logdir, 'model_summary.txt'))
        self.lr_after_dscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, NetManager.OBJECT_EPOCH)
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=10, total_epoch=10, after_scheduler=self.lr_after_dscheduler)
        self.highestScore = 0
        self.epoch = 0
        self.trc = result_collection(LearningIndex.TRAINING)
        self.vrc = result_collection(LearningIndex.VALIDATION)
        self.testrc = result_collection(LearningIndex.TEST)
        self.load_model()


    def setGPUnum(self, gpunum = None):
        if gpunum is not None:
            self.iscuda = True
            self.cuda_device_count = gpunum
            return
        if torch.cuda.is_available():
            self.iscuda = True
            self.cuda_device_count = torch.cuda.device_count()
        else:
            self.iscuda = False
            self.cuda_device_count = 0

    @staticmethod
    def setopt(netparameters, learning_rate, opt = 'adam'):
        if opt == 'adam':
            return optim.Adam(params = netparameters, lr = learning_rate)
        elif opt == 'momentum':
            return optim.SGD(netparameters, learning_rate, momentum=0.9)
        else:
            assert 0, 'The Optimizer is ambiguous'

    @staticmethod
    def getBatchNumAndCycle(dataloader):
        if dataloader is not None:
            return dataloader.dataset.batch_num, cycle(dataloader)
        else:
            return None, None


    @staticmethod
    def setloss(loss='l1', weight=None):
        loss = loss.lower()
        if loss == 'l1':
            return nn.L1Loss()
        elif loss == 'l2':
            return nn.MSELoss()
        elif loss == 'wl1':
            return WeightedL1Loss(weight)
        elif loss == 'wl2':
            return WeightedMSELoss(weight)
        else:
            assert 0, 'The loss is ambiguous'

    def test(self):
        def block_based_test():
            _, c, h, w = recons.shape
            data_list = list()
            pos_list = [
                np.array((NetManager.TEST_BY_BLOCKED_WIDTH if (x + NetManager.TEST_BY_BLOCKED_WIDTH) <= w else (w - x),
                          NetManager.TEST_BY_BLOCKED_HEIGHT if (y + NetManager.TEST_BY_BLOCKED_HEIGHT) <= h else (h - y),
                          x, y))
                for y in range(0, h - 1, NetManager.TEST_BY_BLOCKED_HEIGHT)
                for x in range(0, w - 1, NetManager.TEST_BY_BLOCKED_WIDTH)]
            # inputs = F.pad(inputs, pad_opt)
            # for pos in pos_list:

        self.load_model(gpunum=1)
        MSE_loss = nn.MSELoss()
        recon_MSE_loss = nn.MSELoss()
        if self.iscuda:
            MSE_loss.cuda()
            recon_MSE_loss.cuda()
        self.net.eval()

        if NetManager.TEST_BY_BLOCKED:
            (h_pad, w_pad, no_h_pad, no_w_pad) = self.invest_net_pad(self.net, (self.data_channel_num, 100,100), set_zero=False, device=self.iscuda)
            pad_opt = [w_pad, w_pad, h_pad, h_pad]
        seq_paths = myUtil.getDirlist(NetManager.TEST_PATH)
        seqs_dic = {}
        for path in seq_paths:
            seqs_dic[path] = []
        test_psnr_mean = []
        with torch.no_grad():
            for i in range(len(self.test_loader)):
                current_path = self.test_loader.dataset.batch[i]
                if current_path in '3840x2160':
                    pass
                (recons, inputs, gts) = next(self.test_iter)
                if self.iscuda:
                    recons = recons.cuda()
                    inputs = inputs.cuda()
                    gts = gts.cuda()
                outputs = self.net(inputs)
                mse = MSE_loss(outputs, gts)
                recon_mse = torch.mean((gts) ** 2)
                print('%s : %s' %(current_path, myUtil.psnr(mse)))
                seqs_dic[os.path.dirname(current_path)].append((myUtil.psnr(mse), myUtil.psnr(recon_mse)))
        for key, values in seqs_dic.items():
            mean = np.mean(np.array(values))
            print('%s : %s' %(key, mean))




    def load_model(self, gpunum = None, result_collection=True):
        if gpunum is not None:
            self.cuda_device_count = gpunum
        if NetManager.cfg.LOAD_SAVE_MODEL:
            PATH = './' + NetManager.MODEL_PATH + '/' + self.name + '_model.pth'

            if self.iscuda:
                checkpoint = torch.load(PATH)
            else:
                checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
            try:
                self.net.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                self.logger.info('Not equal GPU Number.. %s' % e)
                if self.cuda_device_count == 1:
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model_state_dict'].items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.net.load_state_dict(new_state_dict)
                else:
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model_state_dict'].items():
                        name = 'module.' + k  # remove `module.`
                        new_state_dict[name] = v
                    self.net.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if NetManager.cfg.LOAD_SAVE_MODEL == 1:
                self.epoch = checkpoint['epoch']
                self.tb.step = checkpoint['TensorBoardStep']
                if 'valid_psnr' in checkpoint:
                    self.highestScore = checkpoint['valid_psnr']
                self.logger.info('It is Transfer Learning...')
            if result_collection:
                self.trc.load(os.path.join('./', NetManager.MODEL_PATH, self.name), LearningIndex.TRAINING)
                self.vrc.load(os.path.join('./', NetManager.MODEL_PATH, self.name), LearningIndex.VALIDATION)
            self.logger.info('Load the saved checkpoint')
            # for g in optimizer.param_groups:
            #     g['lr'] = 0.0001
        if gpunum is not None:
            self.setGPUnum(None)




    def save_model(self, isBest=False):
        if isBest:
            PATH = './' + NetManager.MODEL_PATH + '/' + self.name + '_model_best.pth'
        else:
            PATH = './' + NetManager.MODEL_PATH + '/' + self.name + '_model.pth'
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'TensorBoardStep': self.tb.step,
            'valid_psnr': self.highestScore
        }, PATH)
        self.trc.save(os.path.join('./', NetManager.MODEL_PATH, self.name))
        self.vrc.save(os.path.join('./', NetManager.MODEL_PATH, self.name))



    def train(self, input_channel_list = None):
        dataset = self.train_loader.dataset
        self.logger.info('Training Start')
        for epoch_iter, epoch in enumerate(range(NetManager.OBJECT_EPOCH), self.epoch):
            for i in range(dataset.batch_num):
                (inputs, gts, recons) = next(self.train_iter)
                if self.iscuda:
                    recons = recons.cuda()
                    inputs = inputs.cuda()
                    gts = gts.cuda()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, gts)
                if epoch_iter==0:
                    self.trc.recon_mse[-1].append(self.GTMSELoss(recons, gts).item())
                self.trc.mse[-1].append(self.ResultMSELoss(outputs, gts).item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.trc.loss[-1].append(loss.item())
                # self.tb.SetLoss('CNN_Filter', MSE.item())
                # self.tb.SetLoss('VVC_In-loop', recon_MSE.item())
                # self.tb.plotScalars()
                # if i % NetManager.PRINT_PERIOD == NetManager.PRINT_PERIOD - 1:
                #     self.logger.info('[Epoch : %d, %5d/%d] loss: %.7f' %
                #                 (epoch_iter, i + 1,
                #                  dataset.batch_num, running_loss / dataset.PRINT_PERIOD))
                #     running_loss = 0.0
                # del recons, inputs, gts
            self.trc.endofEpoch(self.logger, self.tb, self.epoch)
            self.tb.step += 1  # Must Used
            if self.valid_loader is not None:
                self.valid(epoch_iter, input_channel_list)
            self.lr_scheduler.step(epoch_iter)
            self.epoch += 1

    def valid(self, epoch_iter, input_channel_list = None):

        valid_dataset = self.valid_loader.dataset
        # cumsum_valid = torch.zeros(1).to(
        #     torch.device("cuda:0" if self.iscuda else "cpu"))

        if input_channel_list is None:
            input_channel_list = list()
            for i in range(valid_dataset.input_data_channel):
                input_channel_list.append(str(i) + '_channel')


        for i in range(valid_dataset.batch_num):
            with torch.no_grad():
                (inputs, gts, recons) = next(self.valid_iter)
                if self.iscuda:
                    recons = recons.cuda()
                    inputs = inputs.cuda()
                    gts = gts.cuda()
                outputs = self.net(inputs)
                self.vrc.loss[-1].append(self.criterion(outputs, gts).item())
                if epoch_iter==0:
                    self.vrc.recon_mse[-1].append(self.GTMSELoss(recons, gts).item())
                self.vrc.mse[-1].append(self.ResultMSELoss(outputs, gts).item())
                if self.cuda_device_count > 1:
                    outputs = torch.cat(outputs, dim=0)
                # cumsum_valid += torch.abs(outputs).sum(dim=0)
                if i == 0:
                    self.tb.batchImageToTensorBoard(self.tb.Makegrid(recons), self.tb.Makegrid(outputs), 'CNN_Reconstruction')
                    self.tb.plotDifferent(self.tb.Makegrid(gts - outputs), 'CNN_Residual')
                    self.tb.plotDifferent(self.tb.Makegrid(gts - outputs), 'CNN_Residual', percentile=100)
        #             if epoch_iter == 1:
        #                 for channel, channel_name in enumerate(input_channel_list):
        #                     self.tb.plotMap(valid_dataset.ReverseNorm(
        #                         inputs.split(1, dim=1)[3], idx=channel).narrow(dim=2, start=0,                                                                                              length=128).narrow(
        #                         dim=3, start=0, length=64), channel_name)
        #                     # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[4], idx=4).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Mode_Map', [0, 3], 4)
        #                     # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[5], idx=5).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Depth_Map', [1, 6], 6)
        #                     # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[6], idx=6).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Hor_Trans', [0, 2], 2)
        #                     # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[7], idx=7).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Ver_Trans', [0, 2], 2)
        #                     # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[8], idx=8).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'ALF_IDX', [0, 16], 17)
        #             self.logger.info("[epoch:%d] Finish Plot Image" % epoch_iter)
        # cumsum_valid /= (valid_dataset.batch_num * valid_dataset.batch_size)
        # self.tb.plotMSEImage(cumsum_valid, 'Error_MSE')
        # self.tb.plotMAEImage(cumsum_valid, 'Error_MAE', percentile=90)
        # self.tb.plotMAEImage(cumsum_valid, 'Error_MAE', percentile=95)
        # self.tb.plotMAEImage(cumsum_valid, 'Error_MAE', percentile=100)
        self.vrc.endofEpoch(self.logger, self.tb, self.epoch)
        if self.highestScore > self.vrc.mse[-2]:
            self.save_model(isBest=True)
            self.highestScore = self.vrc.mse[-2]
            self.logger.info('is Highest Model')
        else:
            self.save_model(isBest=False)



    @staticmethod
    def invest_net_pad(model, input_size, batch_size=-1, set_zero=False, device=True):

        def register_hook(module):

            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                if class_name == 'Conv2d':
                    if module.kernel_size[0] != 1 and module.padding[0] == 0:
                        no_height_pad.append((module.kernel_size[0] + 0.5) // 2)
                    if module.kernel_size[1] != 1 and module.padding[1] == 0:
                        no_width_pad.append((module.kernel_size[1] + 0.5) // 2)
                    h, w = module.padding
                    width_pad.append(w)
                    height_pad.append(h)
                    if set_zero:
                        module.padding = (0, 0)
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))
        if device:
            device = 'cuda'
        else:
            device = 'cpu'
        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        # print(type(x[0]))

        # create properties
        width_pad = []
        height_pad = []
        no_height_pad = []
        no_width_pad = []
        summary = OrderedDict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()
        return int(np.sum(height_pad)), int(np.sum(width_pad)), int(np.sum(no_height_pad)), int(np.sum(no_width_pad))




class LowerBound(torch.autograd.Function):
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 device,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.FloatTensor([reparam_offset])

        self.build(ch, torch.device(device))

    def build(self, ch, device):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta.to(device))

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma.to(device))
        self.pedestal = self.pedestal.to(device)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound()(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound()(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs