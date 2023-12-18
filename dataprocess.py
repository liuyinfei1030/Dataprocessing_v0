#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 17:21
# @Author  : LGC
# @File    : dataprocess.py
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


class DataProcess:
    def __init__(self, data_original, OutProcessedFilename, sampling_rate, num_for_FFT, num_for_periodic_aver, num_sample, num_sensors, num_data_per_cycle):
        """
        目前的处理方法包括：
        (1)时域同步平均法降噪：time_synchronous_averaging(已完成)
        (2)去除趋势项：de_trending_items(已完成)
        (3)移动平均：move_average(已完成)
        (4)滤波：低通滤波/高通滤波/带通滤波/带阻滤波：filter(已完成)
        (5)EMD处理
        (6)自相关分析
        (7)FFT变换fft(已完成)
        (8)数据增强
        :param data_original:
        :param OutProcessedFilename:
        :param sampling_rate:
        :param num_for_FFT:
        :param num_for_periodic_aver:
        :param num_sample:
        :param num_sensors:
        :param num_data_per_cycle:
        """
        self.data_original = data_original
        self.OutProcessedFilename = OutProcessedFilename
        self.data_original_used = self.data_original.iloc[:, :]  # 设置选择数据量的大小,数据类型为pandas.core.frame.DataFrame,
        self.size = self.data_original_used.shape  # 求解表格中数据的个数,行数=size[0]，列数=size[1]
        self.size_row = self.size[0]
        self.size_colunm = self.size[1]

        self.sampling_rate =sampling_rate
        self.num_for_FFT = num_for_FFT
        self.num_for_periodic_aver = num_for_periodic_aver
        self.num_sample = num_sample
        self.num_sensors = num_sensors
        self.num_data_per_cycle = num_data_per_cycle

        # 定义空矩阵，用于存储处理后数据的主要用于绘图、导出致excel表######
        self.data_processed_detrend = np.empty([self.size_row, self.size_colunm])  # 去除趋势项
        self.data_processed_moving_average = np.empty([self.size_row, self.size_colunm])  # 移动平均
        self.data_original_FFT = np.empty([self.num_for_FFT, self.size_colunm])  # 用于FFT变换的原始数据
        self.data_processed_freq = np.empty([int(self.num_for_FFT / 2), self.size_colunm])  # FFT变换后的频率
        self.data_processed_amp = np.empty([int(self.num_for_FFT / 2), self.size_colunm])  # FFT变换后的幅值
        self.data_processed_ps = np.empty([int(self.num_for_FFT / 2), self.size_colunm])  # FFT变换后的功率
        self.data_processed_FFT_allinformation = np.empty([int(self.num_for_FFT / 2), 3 * self.size_colunm])  # FFT变换后的所有信息，分别为“频率/幅值/功率”
        self.data_processed_autocorrcoef = np.empty([self.size_row, self.size_colunm])  # 自相关分析
        self.data_processed_digital_filtering_lowpass = np.empty([self.size_row, self.size_colunm])  # 数字滤波—低通滤波-后的数据
        self.data_processed_digital_filtering_highpass = np.empty([self.size_row, self.size_colunm])  # 数字滤波—高通滤波-后的数据
        self.data_processed_digital_filtering_bandpass = np.empty([self.size_row, self.size_colunm])  # 数字滤波—带通滤波-后的数据
        self.data_processed_digital_filtering_bandstop = np.empty([self.size_row, self.size_colunm])  # 数字滤波—带阻滤波-后的数据


    def time_synchronous_averaging(self):
        """
        处理方法--时域同步平均法降噪：去除低频噪声,程序已经验证，没问题，但具体操作，很难准确的判断出1个周期有多少个采样点，所以实际操作困难，可以让学生专门去研究一下
        :return:
        """
        ''''''
        data_processed_time_synchronous_averaging = np.empty([self.num_data_per_cycle, self.size_colunm])  # 时域同步平均法降噪后的数据
        NTime = int(np.fix(self.size_row / self.num_data_per_cycle))  # 确定分段数，即一共分几段
        print(NTime)
        print(self.num_data_per_cycle)
        for i in range(self.size_colunm):
            data_original_time_synchronous_averaging = self.data_original_used.iloc[:,i].copy()  # 说明：overwrite_data为True,会修改data_original_used,所以这里采用复制数据的方法更好
            xxx = np.array(data_original_time_synchronous_averaging).reshape(self.num_data_per_cycle, -1,order='F')  # 重新组成为num_data_per_cycle行 * 任意列
            data_processed_time_synchronous_averaging[:, i] = xxx.sum(axis=1) / NTime

        print('时域同步平均OK')

    def de_trending_items(self):
        """
        处理方法--处理方法--去除趋势项：去除后的数据存储在data_processed_detrend中
        :return:
        """
        for i in range(self.size_colunm):
            data_original_detrend = self.data_original_used.iloc[:, i].copy()#说明：overwrite_data为True,会修改data_original_used,所以这里采用复制数据的方法更好
            signal.detrend(data_original_detrend, axis=- 1, type='linear', bp=0, overwrite_data=True)
            self.data_processed_detrend[:, i] = data_original_detrend
        print('去除趋势项OK')

    def move_average(self):
        """
        处理方法--移动平均：基于移动平均框（通过卷积）平滑，移动平均后的数据存储在data_processed_moving_average中#########
        :return:
        """
        for i in range(self.size_colunm):
            box_pts = self.num_for_periodic_aver
            data_original_moving_average = self.data_original_used.iloc[:,i].copy()
            box = np.ones(box_pts) / box_pts
            self.data_processed_moving_average[:,i] = np.convolve(data_original_moving_average, box, mode='same')
        print('移动平均OK')

    def filter(self):
        """
        处理方法——滤波：低通滤波/高通滤波/带通滤波/带阻滤波,还没有彻底搞明白
        处理方法--滤波，基本原理说明
        基于scipy模块使用Python实现简单滤波处理，包括内容有1.低通滤波，2.高通滤波，3.带通滤波，4.带阻滤波器。
        具体含义查阅，信号与系统。简单的理解就是低通滤波去除高于某一阈值频率的信号；
        高通滤波去除低于某一频率的信号；带通滤波指的是类似低通高通的结合保留中间频率信号；
        带阻滤波也是低通高通的结合只是过滤掉的是中间部分。
        如何实现的呢？我的理解，是通过时域转换为频域，在频域信号中去除相应频域信号，最后在逆转换还原为时域型号。
        有什么作用呢？可以消除一些干扰信号，以低通滤波为例，例如我们如果只是统计脉搏信号波形，应该在1Hz左右，却发现波形信号上有很多噪音，
        这些噪音都是成百上千Hz的，这些对于脉搏信号波形就属于无用的噪音，我们就可以通过低通滤波器将超出某一阈值的信号过滤掉，
        此时得到的波形就会比较平滑了。

        1).低通滤波
        这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除400hz以上频率成分，即截至频率为400hz,则wn=2*400/1000=0.8
        2).高通滤波
        假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除100hz以下频率成分，即截至频率为100hz,则wn=2*100/1000=0.2
        3).带通滤波
        假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除100hz以下，400hz以上频率成分，
        即截至频率为100，400hz,则wn1=2*100/1000=0.2； wn2=2*400/1000=0.8。Wn=[0.2,0.8]
        4).带阻滤波
        这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除100hz以上，400hz以下频率成分，
        即截至频率为100，400hz,则wn1=2*100/1000=0.2； wn2=2*400/1000=0.8。Wn=[0.02,0.8]，和带通相似，但是带通是保留中间，而带阻是去除。

        阶数：阶数越高，截止频率等参数越精确，但是电路结构也越复杂。 简单说比如你的截止频率是 100HZ， 你只有2阶的话可能实际的截止频率是95-1000HZ，
        衰减比较慢，但如果是20阶的话，可能截止频率就变成了95-105HZ，衰减很快。但是阶数上升，实际电路的结构就会非常的复杂，浪费资源。
        阶数高，就越接近理想滤波器。

        对于“T1 Orthogonal milling of TC4”，采样频率为15000hz,信号本身最大的频率为：4刃*4000rpm/60=266hz，上下限分别设为450hz，150hz，
        则wn1=2*150/15000=0.02;wn2=2*450/15000=0.06
        :return:
        """
        for i in range(self.size_colunm):
            data_original_digital_filtering = self.data_original_used.iloc[:, i].copy()  # 需要进行滤波的数据
            b, a = signal.butter(8, 0.06, 'lowpass')  # 配置滤波器 b: 滤波器的分子系数向量； a: 滤波器的分母系数向量
            self.data_processed_digital_filtering_lowpass[:, i] = signal.filtfilt(b, a,data_original_digital_filtering)  # data为要过滤的信号
            b, a = signal.butter(8, 0.02, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
            self.data_processed_digital_filtering_highpass[:, i] = signal.filtfilt(b, a,data_original_digital_filtering)  # data为要过滤的信号
            b, a = signal.butter(8, [0.02, 0.15], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
            self.data_processed_digital_filtering_bandpass[:, i] = signal.filtfilt(b, a,data_original_digital_filtering)  # data为要过滤的信号
            b, a = signal.butter(8, [0.02, 0.06], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
            self.data_processed_digital_filtering_bandstop[:, i] = signal.filtfilt(b, a,data_original_digital_filtering)  # data为要过滤的信号
        print('数字滤波OK')

        '''
        #绘制滤波前后的信号
        plt.figure(figsize =(12,10))
        plt.subplot(511)
        plt.plot(data_original_digital_filtering[1:2400],'r')
        plt.ylabel("original")
        plt.subplot(512)
        plt.plot(data_processed_digital_filtering_lowpass[1:2400,2],'g')
        plt.ylabel("lowpass")
        plt.subplot(513)
        plt.plot(data_processed_digital_filtering_highpass[1:2400,2],'y')
        plt.ylabel("highpass")
        plt.subplot(514)
        plt.plot(data_processed_digital_filtering_bandpass[1:2400,2],'b')
        plt.ylabel("bandpass")
        plt.subplot(515)
        plt.plot(data_processed_digital_filtering_bandstop[1:2400,2],'ob')
        plt.ylabel("bandstop")

        # plt.figure(figsize =(12,10))
        # plt.plot(data_original_digital_filtering[1:400],'r')
        # plt.plot(data_processed_digital_filtering_lowpass[1:400,2],'g')
        # plt.plot(data_processed_digital_filtering_highpass[1:400,2],'y')
        # plt.plot(data_processed_digital_filtering_bandpass[1:400,2],'b')
        # plt.plot(data_processed_digital_filtering_bandstop[1:400,2],'ob')
        plt.show()
        '''

    def fft(self):
        """
        处理方法--FFT变换：绘制频谱图、功率谱图，并导出频谱/幅值/功率谱数据
        :return:
        """
        ######处理方法--FFT变换：绘制频谱图、功率谱图，并导出频谱/幅值/功率谱数据######

        for i in range(self.size_colunm):
            data_original_FFT_temp=self.data_original_used.iloc[0:self.num_for_FFT,i].copy()
            data_fft = np.fft.fft(data_original_FFT_temp)#快速傅里叶变换,得到一系列的复数值，如a+bi；即实现用很多正玄波取表示这个信号,data_fft格式为numpy.ndarray，为1维数组，元素个数等于data中元素的个数,每个元素是1个复数，代表1个分解的信号信息
            Y = np.abs(data_fft)#对复数取绝对值，即求该复数的模
            Y_real_amp= Y/(self.num_for_FFT/2)#将fft结果的振幅特征转换为原始信号的振幅，即真实幅值
            freq_real = np.arange(self.num_for_FFT) / self.num_for_FFT * self.sampling_rate  #获得真实频率
            ps_real = (Y_real_amp ** 2) / self.num_for_FFT#获得功率谱,直接法
            # 取一半
            freq_real_half = freq_real[range(int(self.num_for_FFT / 2))]
            Y_real_amp_half = Y_real_amp[range(int(self.num_for_FFT / 2))]
            ps_real_half=ps_real[range(int(self.num_for_FFT / 2))]
            #将FFT变换后的数据写入
            self.data_original_FFT[:,i] =data_original_FFT_temp  # 用于FFT变换的原始数据
            self.data_processed_freq[:,i] = freq_real_half # FFT变换后的频率
            self.data_processed_amp[:,i] = Y_real_amp_half # FFT变换后的幅值
            self.data_processed_ps[:,i] = ps_real_half  # FFT变换后的功率
            self.data_processed_FFT_allinformation[:,i*3-2]=freq_real_half
            self.data_processed_FFT_allinformation[:,i*3-1]=Y_real_amp_half
            self.data_processed_FFT_allinformation[:,i*3-0]=ps_real_half
        print('FFT 处理 OK')

        path = os.path.join('E:/1研究生/9.实验/0.T00/6.数据集/', 'data_processed_freq.xlsx')
        df1 = pd.DataFrame(self.data_processed_freq)
        df1.to_excel(path, index=False)

    def data_enhancement(self):
        """
        说明：针对铣削力信号进行数据增强，主要方法包括：
        ①FX与FY的合力；②FX、FY及FZ的合力；③FX/FY比值；④FX/FZ比值；⑤FY/FZ比值;⑥FX/Ftot比值；⑦FY/Ftot比值；⑧FZ/Ftot比值；⑨。。。
        :return:
        """
        # 读入原始数据
        data_used_for_enhancement = self.data_original_used  # 注意，此处必须严格按照FX/FY/FZ的顺序排列
        # data_used_for_enhancement=pd.DataFrame(data_processed_moving_average)
        size_for_enhancement = data_used_for_enhancement.shape  # 求解表格中数据的个数,行数=size[0]，列数=size[1]
        size_for_enhancement_row = size_for_enhancement[0]  # 行数
        size_for_enhancement_colunm = size_for_enhancement[1]  # 列数

        # 定义空矩阵，用于存储增强后的数据
        self.data_processed_enhancement = np.empty([size_for_enhancement_row,6 * size_for_enhancement_colunm])  # 定义用于存储增强数据的空矩阵，注意这里的6*size_for_enhancement_colunm要足够存储增强后的数据量

        # 用于数据增强
        j = 0
        k = 0
        for i in range(int(size_for_enhancement_colunm / 3)):  # 遍历数据的每组样本,i是从0开始的

            j = 3 * i  # 因为原始数据中有3个方向的铣削力，所以3*i，此处用于读取数据
            k = self.num_sensors * i  # 增强后的数据个数，此处用于存数据。
            data_FX = data_used_for_enhancement.iloc[:, j]  # 读取需要求解的列
            data_FY = data_used_for_enhancement.iloc[:, j + 1]  # 读取需要求解的列
            data_FZ = data_used_for_enhancement.iloc[:, j + 2]  # 读取需要求解的列
            size_E = data_FX.size

            # 数据增强处理
            data_FX_FY = [data_FX, data_FY]  # 定义辅助求解矩阵
            data_FX_FY_abs = [abs(data_FX), abs(data_FY)]
            data_FX_FY_FZ = [data_FX, data_FY, data_FZ]  # 定义辅助求解矩阵

            data_E1 = np.linalg.norm(data_FX_FY, ord=None, axis=0)  # ①FX与FY的合力
            data_E2 = np.linalg.norm(data_FX_FY_FZ, ord=None, axis=0)  # ②FX、FY及FZ的合力
            data_E3 = np.divide(data_FX, data_E1)
            data_E4 = np.divide(data_FY, data_E1)
            data_E5 = np.divide(data_FZ, data_E1)
            data_E6 = np.divide(data_FX, data_E2)
            data_E7 = np.divide(data_FY, data_E2)
            data_E8 = np.divide(data_FZ, data_E2)
            data_E9 = np.diff(data_FX_FY, axis=0)  # FX-FY
            data_E10 = np.diff(data_FX_FY_abs, axis=0)  # abs(FX)-abs(FY)
            data_E11 = np.divide(data_E9, data_E1)  # (FX-FY)/E1
            data_E12 = np.divide(data_E10, data_E1)  # (abs(FX)-abs(FY))/E1

            # 这个比值有可能是0，所以数据增强不用这种方法
            # data_E3=np.divide(data_FY, data_FX)#data_FX/data_FY
            # data_E4=np.divide(data_FX, data_FZ)#data_FX/data_FZ
            # data_E5=np.divide(data_FY, data_FZ)#data_FY/data_FZ

            # 数据存储,注意，此处如果增加数据（或理解为传感器），需要修改num_sensors的值
            self.data_processed_enhancement[:, k] = data_FX
            self.data_processed_enhancement[:, k + 1] = data_FY
            self.data_processed_enhancement[:, k + 2] = data_FZ
            self.data_processed_enhancement[:, k + 3] = data_E1
            self.data_processed_enhancement[:, k + 4] = data_E2
            self.data_processed_enhancement[:, k + 5] = data_E3
            self.data_processed_enhancement[:, k + 6] = data_E4
            self.data_processed_enhancement[:, k + 7] = data_E5
            self.data_processed_enhancement[:, k + 8] = data_E6
            self.data_processed_enhancement[:, k + 9] = data_E7
            self.data_processed_enhancement[:, k + 10] = data_E8
            self.data_processed_enhancement[:, k + 11] = data_E9
            self.data_processed_enhancement[:, k + 12] = data_E10
            self.data_processed_enhancement[:, k + 13] = data_E11
            self.data_processed_enhancement[:, k + 14] = data_E12
            print(i)  # 为了可视化监督求解过程

    def feature_extraction(self):
        """
        特征提取
        :return:
        """
        #####（5.1）确定用于提取特征的数据######
        # data_used_for_feature=data_original_used
        data_used_for_feature = pd.DataFrame(self.data_processed_enhancement)

        size_feature = data_used_for_feature.shape  # 求解表格中数据的个数,行数=size[0]，列数=size[1]
        size_feature_row = size_feature[0]
        size_feature_colunm = size_feature[1]

        ######（5.2）求解特征，数据存储在data_processed_faetures中，特征依次存储在对应列######
        for i in range(size_feature_colunm):  # 遍历数据的每列
            data_X1 = data_used_for_feature.iloc[:, i]  # 读取需要求解的列
            size_X1 = data_X1.size

            # 时域统计特征
            max = data_X1.nlargest(1).mean()  # 特征1:最大值。注：将峰值修改为最大的前200个数的均值,比直接取最大值好，data_X1.nlargest(200).mean()
            min = data_X1.nsmallest(1).mean()  # 特征2:最小值。这里取的样本应该是有讲究的，取多少个数值的最大值合适？
            max_min = max - min  # 特征3，峰峰值
            mean_value = np.sum(data_X1) / size_X1  # 特征4，平均值
            absolute_mean_value = np.sum(np.fabs(data_X1)) / size_X1  # 特征5，绝对均值，绝对值的平均值
            standard_deviation = np.sqrt(np.sum(np.square(data_X1 - mean_value)) / size_X1)  # 特征6，标准差
            root_mean_score = np.sqrt(np.sum(np.square(data_X1)) / size_X1)  # 特征7，均方根值（RMS，有效值）

            # 无量纲波形特征
            skewness_value = np.sum(np.power((data_X1 - mean_value) / standard_deviation, 3)) / size_X1  # 特征8，偏度
            Kurtosis_value = np.sum(np.power((data_X1 - mean_value) / standard_deviation, 4)) / size_X1  # 特征9，峭度
            shape_factor = root_mean_score / absolute_mean_value  # 特征10，波形因子=有效值/绝对平均幅值
            crest_factor = max / root_mean_score  # 特征11，峰值因子=峰值/有效值
            pulse_factor = max / absolute_mean_value  # 特征12，脉冲因子=峰值/绝对平均值
            Root_amplitude = np.square(np.sum(np.sqrt(np.fabs(data_X1))) / size_X1)  # 方根幅值
            clearance_factor = max / Root_amplitude  # 特征13，裕度因子=峰值/方根幅值

            # 对于PAPERA_T不同的试验有不同的转速
            '''
            求解原理：数据扩充后，每组试验15列，第21组（（21-1）*15）转速是5000，第22组转速是6000，第23组转速是7000，第24组转速是8000，第25组转速是9000
            '''
            ''' '''
            if i >= (21 - 1) * 15 and i < 21 * 15:
                nsp_input = 5000
            elif i >= (22 - 1) * 15 and i < 22 * 15:
                nsp_input = 6000
            elif i >= (23 - 1) * 15 and i < 23 * 15:
                nsp_input = 7000
            elif i >= (24 - 1) * 15 and i < 24 * 15:
                nsp_input = 8000
            elif i >= (25 - 1) * 15 and i < 25 * 15:
                nsp_input = 9000
            else:
                nsp_input = 7000

            # nsp_input=7000
            sampling_rate = self.num_data_per_cycle * nsp_input / 60

            # 频域特征
            data_fft = np.fft.fft(
                data_X1)  # 快速傅里叶变换,得到一系列的复数值，如a+bi；即实现用很多正玄波取表示这个信号,data_fft格式为numpy.ndarray，为1维数组，元素个数等于data中元素的个数,每个元素是1个复数，代表1个分解的信号信息
            Y = np.abs(data_fft)  # 对复数取绝对值，即求该复数的模
            Y_real_duichen = Y / (size_X1 / 2)  # 将fft结果的振幅特征转换为原始信号的振幅，即真实幅值
            freq_real_duichen = np.arange(size_X1) / size_X1 * sampling_rate  # 获得真实频率
            # freq = np.fft.fftfreq(size, 1 / sampling_rate)#获得频率，与上面公式类似

            # 考虑对称，可以得到一半的值,即真实的频率、幅值
            freq_real = freq_real_duichen[range(int(size_X1 / 2))]  # 频率
            Y_real = Y_real_duichen[range(int(size_X1 / 2))]  # 对应的幅值

            # 绘制频谱图（幅值-频率图），进行验证
            '''
            if i==1:
                plt.plot(freq_real, Y_real)
                plt.title('Original wave')
                plt.show()
            '''

            FC = np.dot(freq_real, Y_real) / np.sum(Y_real)  # 特征14， 重心频率
            MSF = np.dot(np.multiply(freq_real, freq_real), Y_real) / np.sum(Y_real)
            RMSF = np.sqrt(MSF)  # 特征15， 均方根频率。
            VF = np.dot(np.multiply(freq_real - FC, freq_real - FC), Y_real) / np.sum(Y_real);
            RVF = np.sqrt(VF)  # 特征16，频率标准差

            ps_real = (Y_real ** 2) / size_X1  # 获得功率谱，暂时没用

            # 时频域，特征18-25
            wp = pywt.WaveletPacket(data_X1, wavelet='db3', mode='symmetric', maxlevel=3)  # 小波包变换，他的原理要搞明白
            aaa = wp['aaa'].data
            aad = wp['aad'].data
            ada = wp['ada'].data
            add = wp['add'].data
            daa = wp['daa'].data
            dad = wp['dad'].data
            dda = wp['dda'].data
            ddd = wp['ddd'].data
            ret1 = np.linalg.norm(aaa, ord=None)  # 求向量、矩阵的范数；ord=None：默认情况下，是求整体的矩阵元素平方和，再开根号
            ret2 = np.linalg.norm(aad, ord=None)
            ret3 = np.linalg.norm(ada, ord=None)
            ret4 = np.linalg.norm(add, ord=None)
            ret5 = np.linalg.norm(daa, ord=None)
            ret6 = np.linalg.norm(dad, ord=None)
            ret7 = np.linalg.norm(dda, ord=None)
            ret8 = np.linalg.norm(ddd, ord=None)

            # 信息论特征指标（熵）
            # 近似熵
            # 样本熵
            # 模糊熵

            '''
            #补充说明
            # 标准差的另一种表示方法，可以分析是否有影响？ standard_deviation = np.sqrt(np.sum(np.square(np.fabs(data) - absolute_mean_value)) / size)
            # 特征，峰值，特征4-1,该特征不合理，应该修改为最大的前。。个取平均值，可以和采样频率、刀齿数等联系起来
            # max = np.max(data)
            '''

            # 数据汇总
            f = [max, min, max_min, mean_value, absolute_mean_value, standard_deviation, root_mean_score,  # 7个
                 skewness_value, Kurtosis_value, shape_factor, crest_factor, pulse_factor, clearance_factor,  # 6个
                 FC, RMSF, VF,  # 3个
                 ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8]  # 8个
            feature = np.array(f).reshape(-1, 1)  # -1表示任意行数，1表示1列

            if i > 0:
                feature_export = np.concatenate((feature_export, feature), axis=1)
            else:
                feature_export = feature

        print("特征提取结束")

        # 存储特征，注意，此处特征排列顺序为：每一列对应的特征，排在每列后面。
        path = os.path.join('E:/1研究生/9.实验/0.T00/6.数据集/', 'data_processed_features.xlsx')
        df1 = pd.DataFrame(feature_export)
        df1.to_excel(path, index=False)

        # # 存储特征，注意，此处特征排列顺序为：每一个样本所有特征排在一列中
        # feature_export_reshape_X1 = feature_export[:, :self.num_sample * self.num_sensors].T  # 选取有效的列，=样本数*传感器数
        # feature_export_reshape = feature_export_reshape_X1.reshape(self.num_sample,-1).T  # 数据变换，变换为每个样本一列,num_sample代表的是样本个数
        # path = os.path.join('E:/1研究生/9.实验/0.T00/6.数据集/', 'data_processed_features_reshape.xlsx')
        # df1 = pd.DataFrame(feature_export_reshape)
        # df1.to_excel(path, index=False)
        #
        # # 对特征进行归一化，每一行归一化。
        #
        # ''''''
        # feature_export_reshape_guiyi = ((feature_export_reshape.T - feature_export_reshape.T.min(axis=0)) / (
        #             feature_export_reshape.T.max(axis=0) - feature_export_reshape.T.min(axis=0))).T
        # path = os.path.join('E:/1研究生/9.实验/0.T00/6.数据集/', 'data_processed_features_reshape_guiyi.xlsx')
        # df1 = pd.DataFrame(feature_export_reshape_guiyi)
        # df1.to_excel(path, index=False)

        print("数据存储OK")