#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 13:51
# @Author  : 我的名字
# @File    : ExampleFeatureExtraction.py.py
# @Description : 这个函数是用来balabalabala自己写

from FeatureExtraction import *

if __name__ == '__main__':
    OutProcessedFilename = f'T00-processed.xlsx'
    window_size=5000

    start_file = 42
    end_file = 283

    # data_path = 'E:/1研究生/9.实验/0.T00/T00/T00_{start_file:03}.xlsx'
    # data_original = pd.read_excel(f'E:/1研究生/9.实验/0.T00/T00/T00_{start_file:03}.xlsx')

    test = FeatureExtraction(start_file, end_file, OutProcessedFilename)
    test.get_feature_bywindow()
    test.get_feature()
