#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 16:06
# @Author  : Liu
# @File    : main.py
# @Description : 编写标准数据处理程序范例，实现从原始数据的裁剪合并并生成标准数据集，对数据集进行降噪、特征提取、数据存储、数据可视化等操作

from FeatureExtraction import *

if __name__ == '__main__':

    # 文件夹地址
    dirForce = 'E:/1研究生/9.实验/0.T00/1.力'
    dirVibration = 'E:/1研究生/9.实验/0.T00/2.振动'
    dirCurrent = 'E:/1研究生/9.实验/0.T00/3.电流'
    dirOut = 'E:/1研究生/9.实验/0.T00/6.数据集'

    # 文件名设置， 第num次切削数据
    num = 15

    VibrationFilename = f'T00-{num}.xlsx'
    ForceFilename = f'T00-{num}.txt'
    CurrentFilename = f'00{num}.mat'
    OutFilename = f'T00-{num:03}.xlsx'
    print(f"VibrationFilename:{VibrationFilename}\nForceFilename:{ForceFilename}\nCurrentFilename:{CurrentFilename}")

    window=50000    # 裁剪宽度

    '''
    初始化DataLoad，传入地址和文件名，设置裁剪宽度，是否显示图像，是否保存Excel
    此部分用于原始数据的裁剪合并并生成标准数据集，完成裁剪后可将此部分注释
    '''
    # test = DataLoad(dirForce,dirVibration,dirCurrent,ForceFilename,VibrationFilename,CurrentFilename,dirOut,OutFilename,window,pltshow = False,excelsave = False)
    # test.read_and_concat_data()

    ''''''
    # data_original = pd.read_excel(f'E:/1研究生/9.实验/0.T00/T00/T00_{num:03}.xlsx')
    OutProcessedFilename = f'T00-processed-bywindow.xlsx'
    # data_path = f'E:/1研究生/9.实验/0.T00/T00'

    test = FeatureExtraction(num,OutProcessedFilename)
    test.get_feature_bywindow()
    # 设置数据处理参数
    # sampling_rate = 50000  # （2-2）FFT设置，设置采样频率，需要和数据传感器的实际采样频率一致
    # num_for_FFT = 8192  # （2-3）FFT设置，设置用于FFT的选取点个数，为了便于进行FFT运算，这里进行频谱分析的数据num_for_FFT通常取2的整数次幂，如1024，2048，4096，8192，16384，32768.
    # num_for_periodic_aver = 3  # 移动平均，控制参数
    # num_sample = 1  # 样本个数，即多少次走刀，对于“T2 Jiaodapuer milling of TC4”取值11，对于“T5 LIUZHIGANG”取值15,对于“T6 ZHENGHAO-ZHdataA.xlsx”取值50
    # num_sensors = 14  # 每个样本的传感器个数，如果只是测力仪就是3，注意数据增强后的个数需要修改
    # num_data_per_cycle = 500  # 一个周期内采集多少个数据点，在时域同步平均时使用，当数据为仿真数据时，频域分析也会用到。对于“T2 Jiaodapuer milling of TC4”：刀具每旋转1周，采集获得7124HZ*60/1909=224个点，这个数值是不准确的。
    #
    # test = DataProcess(data_original, OutProcessedFilename, sampling_rate, num_for_FFT, num_for_periodic_aver, num_sample, num_sensors, num_data_per_cycle)

    #均值 均方差 方差  峭度  偏度  均方根 峰峰值 绝对平均值   峰值
    # 平均频  重心频率率   频率均方根频率标
    # 准差
    
    # test.time_synchronous_averaging()
    # test.de_trending_items()
    # test.move_average()
    # test.fft()
    # test.data_enhancement()
    # test.feature_extraction()











