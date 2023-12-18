#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 14:00
# @Author  : 我的名字
# @File    : CMAPSSDataset.py.py
# @Description : 这个函数是用来balabalabala自己写

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/11 14:50
# @Author  : 我的名字
# @File    : NUAADataset.py.py
# @Description : 这个函数是用来balabalabala自己写

import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import re  # 导入正则表达式模块
import pywt              #Python中的小波分析库
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class NUAA(Dataset):
    def __init__(self, transform=None):
        data_path = './Data/Data collection of machining pocket/'
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.datasets = []
        self.size_row = 50000
        NUAA_Data = [
            # ("W1", self.read_data_from_folder(data_path + "W1")),
            # ("W2", self.read_data_from_folder(data_path + "W2")),
            # ("W3", self.read_data_from_folder(data_path + "W3")),
            # ("W4", self.read_data_from_folder(data_path + "W4")),
            # ("W5", self.read_data_from_folder(data_path + "W5")),
            # ("W6", self.read_data_from_folder(data_path + "W6")),
            # ("W7", self.read_data_from_folder(data_path + "W7")),
            # ("W8", self.read_data_from_folder(data_path + "W8")),
            ("W9", self.read_data_from_folder(data_path + "W9")),
        ]

        for folder, files in NUAA_Data:
            folder_path = os.path.join(self.data_path, folder)
            num_files = len(files)
            # processed_data.append(read_csv_data(folder_path, files))

            for idx, file in enumerate(files):
                file_path = os.path.join(folder_path, file)
                # label = int(file[-7:-4])
                label = (num_files - int(file[-7:-4])) / num_files  # 使用文件名作为标签
                print(folder, file, label)
                data = self.preprocess_data(pd.read_csv(file_path, header=None), num=int(file[-7:-4]))

                # self.datasets.append((data, label))

    def read_data_from_folder(self,folder_path):
        # 获取文件夹中的所有文件
        file_names = os.listdir(folder_path)
        # 过滤出CSV文件
        # 使用正则表达式提取数字部分并将其转换为整数作为排序键
        numeric_part = lambda x: int(re.findall(r'\d+', x)[0])
        csv_files = sorted([f for f in file_names if f.endswith('.csv')], key=numeric_part)
        return csv_files

    def z_score_normalization(self,data):
        # 使用 Z-Score 标准化
        standardized_data = zscore(data)

        # 线性变换将数据缩放到您希望的范围
        # min_val = -1  # 您可以将最小值设置为所需的值
        # max_val = 1  # 您可以将最大值设置为所需的值
        scaled_data = (standardized_data - standardized_data.min()) / (
                    standardized_data.max() - standardized_data.min()) * 2 - 1

        return scaled_data

    def signal_stats_all(self,data):
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
        freq = np.fft.fftfreq(len(data_fft), 1 / 300)  # 获得频率；第一个参数是FFT的点数，一般取FFT之后的数据的长度（size）；
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

        return [absolute_mean_value,
            max,
            root_mean_score,
            Root_amplitude,
            skewness,
            Kurtosis_value,
            shape_factor,
            pulse_factor,
            skewness_factor,
            crest_factor,
            clearance_factor,
            Kurtosis_factor,
            FC,
            MSF,
            RMSF,
            VF,
        ]

    def preprocess_data(self,data,num):
        mean = data.mean()
        std = data.std()

        drop_indices = []

        for index, row in data.iterrows():
            # print(index, row['age'], row['gender'])
            tmp = (row - mean).abs() > 3 * std
            if tmp.any():
                drop_indices.append(index)

        print(drop_indices)

        dst_data = data.drop(drop_indices)

        # Visualize resampled_data
        plt.figure(figsize=(12, 4))
        # plt.subplot(131)
        plt.title("Data")
        plt.plot(dst_data.index, dst_data.iloc[:, 5], label="Channel 1")
        # plt.plot(dst_data.index, dst_data.iloc[:, 2], label="Channel 2")
        plt.legend()
        plt.show()

        # sample_rate = 9
        start_index = int(input(f"Enter the start index for file {num}: "))
        # start_index = 1600
        resampled_data = dst_data.iloc[start_index:start_index + 5000, :8]

        # Visualize resampled_data
        # plt.figure(figsize=(12, 4))
        # # plt.subplot(131)
        # plt.title("resampled_data")
        # plt.plot(resampled_data.index, resampled_data.iloc[:, 1], label="Channel 1")
        # plt.plot(resampled_data.index, resampled_data.iloc[:, 2], label="Channel 2")
        # plt.legend()
        # plt.show()

        # standardized_data = stats.zscore(resampled_data)
        # 最小-最大归一化
        # min_max_scaler = MinMaxScaler()
        # standardized_data = pd.DataFrame(min_max_scaler.fit_transform(resampled_data))


        # Visualize resampled_data
        # plt.figure(figsize=(12, 4))
        # # plt.subplot(131)
        # plt.title("resampled_data")
        # plt.plot(standardized_data.index, standardized_data.iloc[:, 1], label="Channel 1")
        # plt.plot(standardized_data.index, standardized_data.iloc[:, 2], label="Channel 2")
        # plt.legend()
        # plt.show()

        # 最小-最大归一化
        # min_max_scaler = MinMaxScaler()
        # normalized_data_min_max = min_max_scaler.fit_transform(data)

        # normalized_data_zscore = z_score_normalization(data)

        # F_z = np.reshape(self.signal_stats_all(standardized_data.iloc[:, 0]), (1, -1))
        # # F_x = np.reshape(self.signal_stats_all(standardized_data.iloc[:, 1]), (1, -1))
        # # F_y = np.reshape(self.signal_stats_all(standardized_data.iloc[:, 2]), (1, -1))
        # # T_z = np.reshape(self.signal_stats_all(standardized_data.iloc[:, 3]), (1, -1))
        # V_1 = np.reshape(self.signal_stats_all(standardized_data.iloc[:, 1]), (1, -1))
        # V_2 = np.reshape(self.signal_stats_all(standardized_data.iloc[:, 2]), (1, -1))
        # C_w = np.reshape(self.signal_stats_all(standardized_data.iloc[:, 3]), (1, -1))
        # C_I = np.reshape(self.signal_stats_all(standardized_data.iloc[:, 4]), (1, -1))

        # F_z = np.reshape(standardized_data.iloc[:, 0], (-1, 1))
        # V_1 = np.reshape(standardized_data.iloc[:, 1], (-1, 1))
        # V_2 = np.reshape(standardized_data.iloc[:, 2], (-1, 1))
        # C_w = np.reshape(standardized_data.iloc[:, 3], (-1, 1))
        # C_I = np.reshape(standardized_data.iloc[:, 4], (-1, 1))
        #
        # img = np.stack([F_z, V_1, V_2, C_w, C_I], axis=0)

        output_excel_file = f"{num}.xlsx"
        resampled_data.drop(resampled_data.columns[1:4], inplace=True, axis = 1)
        resampled_data.to_excel(output_excel_file, index=False)
        # data.to_excel(output_excel_file, index=False)

        # img = torch.tensor(img, dtype=torch.float32)
        # return img

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        data, label = self.datasets[index]
        return data, label

class CMAPSSDataset(Dataset):
    def __init__(self):
        self.datasets=[]
        self.data_process()
        df = pd.read_excel('FD002.xlsx')
        for index in range(len(df)):
            self.datasets.append((df.iloc[index,0:3],df.iloc[index,3:-1],df.iloc[index,-1]))

    def data_process(self):
        # define filepath to read data
        dir_path = './Data/'

        # define column names for easy indexing
        index_names = ['unit_nr', 'time_cycles']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
        col_names = index_names + setting_names + sensor_names

        # read data
        train = pd.read_csv((dir_path + 'train_FD002.txt'), sep='\s+', header=None, names=col_names)

        # Extract the setting columns for clustering
        settings = train[setting_names]

        # Perform K-Means clustering on the settings
        num_clusters = 6  # You can choose the number of clusters you want
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(settings)
        train['setting_cluster'] = kmeans.labels_  # Add cluster labels to the DataFrame

        # Now, you can normalize the sensor data for each cluster using Z-score
        for cluster in range(num_clusters):
            cluster_data = train[train['setting_cluster'] == cluster]
            sensors = cluster_data[sensor_names]
            scaler = StandardScaler()
            normalized_sensors = scaler.fit_transform(sensors)
            train.loc[train['setting_cluster'] == cluster, sensor_names] = normalized_sensors

        # You can now access the sensor data with Z-score normalization for each cluster
        # For example, to access the normalized sensor data for the first cluster:
        # cluster_0_data = train[train['setting_cluster'] == 0]
        # normalized_sensors_cluster_0 = cluster_0_data[sensor_names]


        drop_labels = setting_names
        train.drop(labels=drop_labels, axis=1, inplace=True)

        # separate title information and sensor data
        title = train.iloc[:, 0:5]
        data = train.iloc[:, 5:-1]

        # data = data.rolling(window=15, center=True, min_periods=1).mean()

        # min-max normalization of the sensor data
        title_norm = (title - title.min()) / (title.max() - title.min())
        data_norm = zscore(data)
        # scaled_data = (data_norm - data_norm.min()) / (
        #         data_norm.max() - data_norm.min()) * 2 - 1
        train_norm = pd.concat([title_norm, data_norm], axis=1)

        # add piece-wise target remaining useful life
        train_norm = self.add_remaining_useful_life(train_norm)
        # train_norm['RUL'].clip(upper=125, inplace=True) # in the paper the MAX RUL is mentioned as 125

        # group the training set with unit
        group = train_norm.groupby(by="unit_nr")
        group = group.apply(lambda x: x.rolling(window=15, center=True, min_periods=1).mean())
        # group = group.groupby(by="unit_nr")

        # df = pd.DataFrame(group)
        # df = df.rolling(window=15, min_periods=1).mean()

        # Save the DataFrame to an Excel file
        output_file_path = "FD002.xlsx"
        train_norm.to_excel(output_file_path, index=False)
    def add_remaining_useful_life(self,df):
        # Get the total number of cycles for each unit
        grouped_by_unit = df.groupby(by="unit_nr")
        max_cycle = grouped_by_unit["time_cycles"].max()

        # Merge the max cycle back into the original frame
        result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

        # Calculate remaining useful life for each row (piece-wise Linear)
        remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
        result_frame["RUL"] = remaining_useful_life

        # drop max_cycle as it's no longer needed
        result_frame = result_frame.drop("max_cycle", axis=1)

        return result_frame

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        setting, data, label = self.datasets[index]
        return setting, data, label
