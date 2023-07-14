#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 17:21
# @Author  :
# @File    :
# @Description : 这个函数是用来balabalabala自己写

import pandas as pd
import numpy as np
import pywt              #Python中的小波分析库
import os                #operating system的缩写，顾名思义， os模块提供的就是各种Python 程序与操作系统进行交互的接口
from scipy import signal
import matplotlib.pyplot as plt
import math
import pylab as pl
from scipy import fftpack
import scipy.signal as signal
from scipy import interpolate
from pylab import mpl
from sklearn.preprocessing import normalize
from math import *


class FeatureExtraction:
    def __init__(self, strat_file, end_file, OutProcessedFilename):
        self.strat_file = strat_file
        self.end_file = end_file
        num = strat_file
        self.data_path = 'E:/1研究生/9.实验/0.T00/T00/T00_{num:03}.xlsx'
        self.data_original = pd.read_excel(f'E:/1研究生/9.实验/0.T00/T00/T00_{num:03}.xlsx')
        self.OutProcessedFilename = OutProcessedFilename

        self.data_original_used = self.data_original.iloc[:, :]  # 设置选择数据量的大小,数据类型为pandas.core.frame.DataFrame,
        self.size = self.data_original_used.shape  # 求解表格中数据的个数,行数=size[0]，列数=size[1]
        self.size_row = self.size[0]
        self.size_colunm = self.size[1]

        self.window_size = 5000
        self.num_window = (self.size_row + self.window_size - 1) // self.window_size

    def signal_stats_bywindow(self, data, num, col):
        stats_list = []
        for i in range(self.num_window):
            start = i * self.window_size
            end = min(start + self.window_size, len(data))
            window_data = data[start:end]

            # 24个特征公式：
            # time_domain:1-12
            # 绝对均值，特征1
            absolute_mean_value = np.sum(np.fabs(window_data)) / self.window_size
            # 峰值，特征2
            max = np.max(window_data)
            # 均方根值，特征3
            root_mean_score = np.sqrt(np.sum(np.square(window_data)) / self.window_size)
            # 方根幅值，特征4
            Root_amplitude = np.square(np.sum(np.sqrt(np.fabs(window_data))) / self.window_size)
            # 歪度值，特征5
            skewness = np.sum(np.power((np.fabs(window_data) - absolute_mean_value), 3)) / self.window_size
            # 峭度值，特征6
            Kurtosis_value = np.sum(np.power(window_data, 4)) / self.window_size
            # 波形因子，特征7
            shape_factor = root_mean_score / absolute_mean_value
            # 脉冲因子，特征8
            pulse_factor = max / absolute_mean_value
            # 歪度因子，特征9
            skewness_factor = skewness / np.power(root_mean_score, 3)
            # 峰值因子，特征10
            crest_factor = max / root_mean_score
            # 裕度因子，特征11
            clearance_factor = max / Root_amplitude
            # 峭度因子，特征12
            Kurtosis_factor = Kurtosis_value / np.power(root_mean_score, 4)

            # frequency_domain:13-16
            data_fft = np.fft.fft(window_data)  # fft,得到一系列的复数值，如a+bi（实部和虚部）:指实现用很多正弦波取表示这个信号
            Y = np.abs(data_fft)  # 对复数取绝对值=为该复数的模
            freq = np.fft.fftfreq(self.window_size, 1 / 50000)  # 获得频率；第一个参数是FFT的点数，一般取FFT之后的数据的长度（size）；
            # 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs，针对回转支承，采用频率为20KHZ
            ps = Y ** 2 / self.window_size  # 频域特性：指频率值及其对应的幅值
            # 重心频率，特征13
            FC = np.sum(freq * ps) / np.sum(ps)
            # 均方频率，特征14
            MSF = np.sum(ps * np.square(freq)) / np.sum(ps)
            # 均方根频率，特征15
            RMSF = np.sqrt(MSF)
            # 频率方差，特征16
            VF = np.sum(np.square(freq - FC) * ps) / np.sum(ps)

            # time and frequency domain:17-24
            wp = pywt.WaveletPacket(window_data, wavelet='db3', mode='symmetric', maxlevel=3)  # 小波包变换
            aaa = wp['aaa'].data
            aad = wp['aad'].data
            ada = wp['ada'].data
            add = wp['add'].data
            daa = wp['daa'].data
            dad = wp['dad'].data
            dda = wp['dda'].data
            ddd = wp['ddd'].data
            ret1 = np.linalg.norm(aaa, ord=None)  # 第一个节点系数求得的范数/ 矩阵元素平方和开方
            ret2 = np.linalg.norm(aad, ord=None)  # ord=None：默认情况下，是求整体的矩阵元素平方和，再开根号
            ret3 = np.linalg.norm(ada, ord=None)
            ret4 = np.linalg.norm(add, ord=None)
            ret5 = np.linalg.norm(daa, ord=None)
            ret6 = np.linalg.norm(dad, ord=None)
            ret7 = np.linalg.norm(dda, ord=None)
            ret8 = np.linalg.norm(ddd, ord=None)


            stats = {
                'file': num,
                'signal': col,
                'absolute_mean_value': absolute_mean_value,
                'max': max,
                'root_mean_score': root_mean_score,
                'root_amplitude': Root_amplitude,
                'skewness': skewness,
                'Kurtosis_value': Kurtosis_value,
                'shape_factor': shape_factor,
                'pulse_factor': pulse_factor,
                'skewness_factor': skewness_factor,
                'crest_factor': crest_factor,
                'clearance_factor': clearance_factor,
                'Kurtosis_factor': Kurtosis_factor,
                'FC': FC,
                'MSF': MSF,
                'RMSF': RMSF,
                'ret1': ret1,
                'ret2': ret2,
                'ret3': ret3,
                'ret4': ret4,
                'ret5': ret5,
                'ret6': ret6,
                'ret7': ret7,
                'ret8': ret8,
            }
            stats_list.append(stats)
        return stats_list

    def signal_stats_all(self, data, num, col):
        # 24个特征公式：
        # time_domain:1-12
        # 绝对均值，特征1
        absolute_mean_value = np.sum(np.fabs(data)) / self.size_row
        # 峰值，特征2
        max = np.max(data)
        # 均方根值，特征3
        root_mean_score = np.sqrt(np.sum(np.square(data)) / self.size_row)
        # 方根幅值，特征4
        Root_amplitude = np.square(np.sum(np.sqrt(np.fabs(data))) / self.size_row)
        # 歪度值，特征5
        skewness = np.sum(np.power((np.fabs(data) - absolute_mean_value), 3)) / self.size_row
        # 峭度值，特征6
        Kurtosis_value = np.sum(np.power(data, 4)) / self.size_row
        # 波形因子，特征7
        shape_factor = root_mean_score / absolute_mean_value
        # 脉冲因子，特征8
        pulse_factor = max / absolute_mean_value
        # 歪度因子，特征9
        skewness_factor = skewness / np.power(root_mean_score, 3)
        # 峰值因子，特征10
        crest_factor = max / root_mean_score
        # 裕度因子，特征11
        clearance_factor = max / Root_amplitude
        # 峭度因子，特征12
        Kurtosis_factor = Kurtosis_value / np.power(root_mean_score, 4)

        # frequency_domain:13-16
        data_fft = np.fft.fft(data)  # fft,得到一系列的复数值，如a+bi（实部和虚部）:指实现用很多正弦波取表示这个信号
        Y = np.abs(data_fft)  # 对复数取绝对值=为该复数的模
        freq = np.fft.fftfreq(self.size_row, 1 / 50000)  # 获得频率；第一个参数是FFT的点数，一般取FFT之后的数据的长度（size）；
        # 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs，针对回转支承，采用频率为20KHZ
        ps = Y ** 2 / self.size_row  # 频域特性：指频率值及其对应的幅值
        # 重心频率，特征13
        FC = np.sum(freq * ps) / np.sum(ps)
        # 均方频率，特征14
        MSF = np.sum(ps * np.square(freq)) / np.sum(ps)
        # 均方根频率，特征15
        RMSF = np.sqrt(MSF)
        # 频率方差，特征16
        VF = np.sum(np.square(freq - FC) * ps) / np.sum(ps)

        # time and frequency domain:17-24
        wp = pywt.WaveletPacket(data, wavelet='db3', mode='symmetric', maxlevel=3)  # 小波包变换
        aaa = wp['aaa'].data
        aad = wp['aad'].data
        ada = wp['ada'].data
        add = wp['add'].data
        daa = wp['daa'].data
        dad = wp['dad'].data
        dda = wp['dda'].data
        ddd = wp['ddd'].data
        ret1 = np.linalg.norm(aaa, ord=None)  # 第一个节点系数求得的范数/ 矩阵元素平方和开方
        ret2 = np.linalg.norm(aad, ord=None)  # ord=None：默认情况下，是求整体的矩阵元素平方和，再开根号
        ret3 = np.linalg.norm(ada, ord=None)
        ret4 = np.linalg.norm(add, ord=None)
        ret5 = np.linalg.norm(daa, ord=None)
        ret6 = np.linalg.norm(dad, ord=None)
        ret7 = np.linalg.norm(dda, ord=None)
        ret8 = np.linalg.norm(ddd, ord=None)

        return {
            'file': num,
            'signal': col,
            'absolute_mean_value': absolute_mean_value,
            'max': max,
            'root_mean_score': root_mean_score,
            'root_amplitude': Root_amplitude,
            'skewness': skewness,
            'Kurtosis_value': Kurtosis_value,
            'shape_factor': shape_factor,
            'pulse_factor': pulse_factor,
            'skewness_factor': skewness_factor,
            'crest_factor': crest_factor,
            'clearance_factor': clearance_factor,
            'Kurtosis_factor': Kurtosis_factor,
            'FC': FC,
            'MSF': MSF,
            'RMSF': RMSF,
            'VF':VF,
            'ret1': ret1,
            'ret2': ret2,
            'ret3': ret3,
            'ret4': ret4,
            'ret5': ret5,
            'ret6': ret6,
            'ret7': ret7,
            'ret8': ret8,

        }


    def get_feature(self):       # 获取特征
        # all_results = pd.DataFrame()
        result_dfs = []
        for num in range(self.strat_file, self.end_file):
            data_path = f'E:/1研究生/9.实验/0.T00/T00/T00_{num:03}.xlsx'
            data_original = pd.read_excel(data_path)
            for col in data_original.columns:
                signal_data = data_original[col]
                stats = self.signal_stats_all(signal_data, num,col)
                result_df = pd.DataFrame([stats])  # 将单个信号统计数据转换为DataFrame
                result_dfs.append(result_df)
        all_results = pd.concat(result_dfs, ignore_index=True)

        with pd.ExcelWriter(self.OutProcessedFilename) as writer:
            for signal, group in all_results.groupby('signal'):
                group.to_excel(writer, index=False, sheet_name=f"{signal}_features")

    def get_feature_bywindow(self):       # 获取特征
        # all_results = pd.DataFrame()
        result_dfs = []
        for num in range(self.strat_file, self.end_file):
            data_path = f'E:/1研究生/9.实验/0.T00/T00/T00_{num:03}.xlsx'
            data_original = pd.read_excel(data_path)
            for col in data_original.columns:
                signal_data = data_original[col]
                stats = self.signal_stats_bywindow(signal_data, num, col)
                result_df = pd.DataFrame(stats)  # 将单个信号统计数据转换为DataFrame
                result_dfs.append(result_df)
        all_results = pd.concat(result_dfs, ignore_index=True)

        with pd.ExcelWriter(self.OutProcessedFilename) as writer:
            for signal, group in all_results.groupby('signal'):
                group.to_excel(writer, index=False, sheet_name=f"{signal}_features")

