import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from CfgEnv.loadCfg import NetManager
from help_func.__init__ import ExecFileName
import time
import os
from help_func.logging import get_str_time
from help_func.help_python import myUtil
from pathlib import Path
import pandas as pd
import csv




class Plot_detail_setting:
    _font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()
    unfilled_markers = [m for m, func in Line2D.markers.items()
                        if func != 'nothing' and m not in Line2D.filled_markers]
    filled_markers = list(Line2D.filled_markers)

    _detail_setting = namedtuple('_detail_setting',
                                 ['color', 'linewidth', 'linestyle',
                                  'marker', 'markersize', 'markeredgecolor',
                                  'markeredgewidth', 'markerfacecolor', 'markevery'
                                  ])
    _training_color = _detail_setting(color='blue', linewidth=None, linestyle='--', marker='o',
                                      markersize=10, markeredgecolor='black', markeredgewidth=None, markerfacecolor='blue',
                                      markevery=10)
    _valid_color = _detail_setting(color='orange', linewidth=None, linestyle='--', marker='o',
                                      markersize=10, markeredgecolor='black', markeredgewidth=None, markerfacecolor='orange',
                                      markevery=10)
    _test_color = _detail_setting(color='darkred', linewidth=None, linestyle='--', marker='o',
                                      markersize=10, markeredgecolor='black', markeredgewidth=None, markerfacecolor='darkred',
                                      markevery=10)
    style_label = 'seaborn-notebook'
    def setdefault(self):
        return self._detail_setting(*([None]*8+[10]))

    def getStrMarker(self, string, mathcal=False):
        if mathcal:
            return r"$\mathcal{%s}$" %string
        return "$" + string + "$"


class Plot_Sequentially_By_Epoch:
    PLOT_EPOCH_INTERVAL = 10
    DATAPATH = 'plot_epoch_data.csv'
    DEFAULT_EPOCH_DATA_LIST = ['Epoch', 'EpochData', 'Class', 'Value']
    STYLE = 'seaborn-notebook'
    def __init__(self, name=None, previousdatapath=None, additional_name=None):
        self.set_plt_style(self.STYLE)
        if name is None:
            name = ExecFileName.filename
        if additional_name is None:
            additional_name = get_str_time()
        self.path = os.path.join(NetManager.RESULT_DIR, name+additional_name)
        self.path = os.path.join(self.path, self.DATAPATH)
        if previousdatapath is not None:
            self.data = self.load_csv_data(previousdatapath)
        elif os.path.exists(self.path):
            self.data = self.load_csv_data(self.path)
        else:
            self.data = pd.DataFrame(columns=self.DEFAULT_EPOCH_DATA_LIST)
            self.filehandler = csv.writer(open(self.path, 'ab'))
            self.filehandler.writerow(['epoch', 'epoch_data', 'data_class', 'value'])
        self.filehandler = csv.writer(open(self.path, 'ab'))
        return

    #for example (epoch=1, epoch_data='loss', data_class='ours', 0.00023)
    def add_and_save_data(self, epoch, epoch_data, data_class, value):
        self.data.loc[len(self.data)] = [epoch, epoch_data, data_class, value]
        self.filehandler.writerow([epoch, epoch_data, data_class, value])
        return

    def load_csv_data(self, path):
        return pd.read_csv(path)

    @staticmethod
    def getBasenameWOext(path):
        return os.path.splitext(os.path.basename(path))[0]

    def slice_ylim(self, data):
        random_data_std = np.std(data)
        random_data_mean = np.mean(data)
        anomaly_cut_off = random_data_std
        lower_limit = random_data_mean - anomaly_cut_off
        upper_limit = random_data_mean + anomaly_cut_off
        slicedata = data[np.all((data>lower_limit, data<upper_limit), axis=0)]
        data_mean = np.mean(slicedata)
        data_min = min(slicedata)
        data_max = max(slicedata)
        return data_min - (data_mean - data_min), data_max + (data_max - data_mean)

    def filteringData(self, dics):
        data = self.data
        try:
            for key, value in dics.items():
                data = data[data[:,key]==value]
        except Exception as e:
            print(e)
        return data


    def save_data_by_graph(self, epoch_data, data_class, fig = None, ax = None):
        assert isinstance(epoch_data, str)
        assert isinstance(data_class, str)
        if fig is None:
            fig, ax = plt.subplots(num = self.STYLE)
        data = self.filteringData({'epoch_data':epoch_data, 'data_class':data_class}).sort_values(by='epoch')
        ax.plot(data['epoch'], data['value'])
        return fig, ax





    def set_plt_style(self, style):
        plt.style.use(style)

    def save_datas_by_graph(self, datadicts):
        assert isinstance(datanames, dict)


