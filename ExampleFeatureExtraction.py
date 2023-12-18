#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 13:51
# @Author  : 刘寅飞
# @File    : ExampleFeatureExtraction.py.py
# @Description : 这个例子是用来对经过裁剪降噪之后的传感器数据进行特征提取

from FeatureExtraction import *

if __name__ == '__main__':
    window_size=5000    # 数据长度
    id = 4              # 刀具序号
    start_file = 1     # 起始文件编号
    end_file = 40+1       # 结束文件编号
    OutProcessedFilename = f'./FeatureExtraction/T0{id}-FeatureExtraction.xlsx'

    # data_path = 'E:/1研究生/9.实验/0.T00/T00/T00_{start_file:03}.xlsx'
    # data_original = pd.read_excel(f'E:/1研究生/9.实验/0.T00/T00/T00_{start_file:03}.xlsx')

    '''初始化FeatureExtraction实例'''
    test = FeatureExtraction(id,start_file, end_file, OutProcessedFilename)
    '''调用实例中的方法'''
    # test.get_feature_bywindow()   # get_feature_bywindow()这个方法是针对一组信号分段进行特征提取，根据需要运行
    test.get_feature()
