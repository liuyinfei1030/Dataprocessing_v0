#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import pywt
# 设置文件路径
# path = r'D:\DEWEsoft\Data\20220520实验数据\20220520-对称B2\Exports'


def get_all_file_preprocessing(path):        # 所有的文件路径，存在列表之中,再返回其路径列表
    files = os.listdir(path)
    real_path = []
    for file in files:
        if not os.path.isdir(path + file):
            tr = "\\"
            real_path.append(path + tr + file)
    return real_path
#a = get_all_file_preprocessing(path)
#print(a)


def get_filename_preprocessing(path):        # 读取所有的文件名，再返回其文件
    files = os.listdir(path)
    return files
#b = get_filename_preprocessing(path)
#print(b)


class DataProcess:
    def __init__(self, path):
        self.path = path

    def get_all_file(self):                  # 获取所有文件
        files = os.listdir(path)
        real_path = []
        for file in files:
            if not os.path.isdir(path + file):
                tr = "\\"
                real_path.append(path + tr + file)
        return real_path

    def get_data(self, i):                  # 获取所有数据
        list = self.get_all_file()
        #print(list)
        data = pd.read_csv(list[i], dtype=float)
        #print(data.head())
        return data

    def get_feature(self, i, column):       # 获取特征
        data_original = self.get_data(i=i)
        #print(data_original.head())        # 展示所有列前5行数据
        data_original = data_original.iloc[:, column]
        #print(data_original.head())        # 展示第2列前5行数据（AI1)
        data_v1 = np.array(data_original)
        #print(data_v1)                     # 转换列表形式
        #print(data_v1.shape)               # (1207296,)
        box_pts = 3
        box = np.ones(box_pts) / box_pts
        data_v2 = np.convolve(data_v1, box, mode='same')
        data_v2_pd = pd.Series(data_v2)
        #print(data_v2.shape)               # (1207296,)
        data = data_v2_pd.iloc[:]
        size = data.size
        #print(size)                        # 每列有1207296行数据

#c = DataProcess(path)
#data1 = c.get_all_file()
#data2 = c.get_data(0)
#data3 = c.get_feature(0, 1)
#print(c)


################################################################################################################
        # 24个特征公式：
        # time_domain:1-12
        # 绝对均值，特征1
        absolute_mean_value = np.sum(np.fabs(data)) / size
        # 峰值，特征2
        max = np.max(data)
        # 均方根值，特征3
        root_mean_score = np.sqrt(np.sum(np.square(data)) / size)
        # 方根幅值，特征4
        Root_amplitude = np.square(np.sum(np.sqrt(np.fabs(data))) / size)
        # 歪度值，特征5
        skewness = np.sum(np.power((np.fabs(data) - absolute_mean_value), 3)) / size
        # 峭度值，特征6
        Kurtosis_value = np.sum(np.power(data, 4)) / size
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
        data_fft = np.fft.fft(data)             # fft,得到一系列的复数值，如a+bi（实部和虚部）:指实现用很多正弦波取表示这个信号
        Y = np.abs(data_fft)                    # 对复数取绝对值=为该复数的模
        freq = np.fft.fftfreq(size, 1 / 50000)  # 获得频率；第一个参数是FFT的点数，一般取FFT之后的数据的长度（size）；
                                                # 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs，针对回转支承，采用频率为20KHZ
        ps = Y ** 2 / size                      # 频域特性：指频率值及其对应的幅值
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

        feature_list = [absolute_mean_value, max, root_mean_score, Root_amplitude, skewness, Kurtosis_value,
                        shape_factor, pulse_factor, skewness_factor, crest_factor, clearance_factor, Kurtosis_factor,
                        FC, MSF, RMSF, VF,
                        ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8]
        return feature_list

#######################################################################################################################
    def get_all_features(self, files_num):   # 获取所有特征
        features = np.empty([files_num, 1, 24])
        for i in range(files_num):                      # get_feature(self, i, column)
            features[i, 0, :] = self.get_feature(i, 6)  # 这里：0为时间，1为AI1(X),2为AI2(Y),3为AI3(z)
        return features

    def get_labels(self, filename=None):     # 获取所有标签
        data = pd.read_csv(filename)
        # print(data.head())
        labels = np.array(data['labels'])
        return labels


path = r'E:\DEWEsoft\数据集4\noise reduction-奇异值分解降噪\降噪重构工况1'
data_N = DataProcess(path)
data_N_AI1 = data_N.get_all_features(20)           # 文件夹中文件数
print(data_N_AI1)
print(data_N_AI1.shape)
data_N_AI1 = data_N_AI1.reshape(data_N_AI1.shape[0], data_N_AI1.shape[1] * data_N_AI1.shape[2])
df1 = pd.DataFrame(data_N_AI1)
df1.to_csv(r'E:\DEWEsoft\数据集4\noise reduction-奇异值分解降噪\noise reduction所有工况特征提取\1\1-AI6.csv', index=False)

