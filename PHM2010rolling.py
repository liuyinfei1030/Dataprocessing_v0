#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/7 14:37
# @Author  : 我的名字
# @File    : PHM2010rolling.py.py
# @Description : 这个函数是用来balabalabala自己写
from PHM2010Dataset import *
import torch
import matplotlib as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

df = pd.read_excel('./Data/PHMall.xlsx')

# 按id_num列分组
grouped = df.groupby('ID')

# 滑动滤波函数，这里使用的是滑动平均
def rolling_filter(group):
    return group.rolling(window=5, min_periods=1).mean()

# 对每个分组应用滑动滤波，并将结果添加到新的列中
df = grouped.apply(rolling_filter).reset_index(level=0, drop=True)

# 保存结果到新的Excel文件
output_file_path = './Data/phmallfilter.xlsx'
df.to_excel(output_file_path, index=False)

print(f"滤波后的数据已保存到 {output_file_path}")