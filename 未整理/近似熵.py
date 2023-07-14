# -*- coding: utf-8 -*-
"""
作者：郑浩
日期：2022年02月28日
"""

import numpy as np

import matplotlib.pyplot as plt


def Approximate_Entropy(x, m, r=0.15):
    # 将x转化为数组
    x = np.array(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")

    if len(x) < m + 1:
        raise ValueError("len(x)小于m+1")  # 将x以m为窗口进行划分
    entropy = 0  # 近似嫡
    for temp in range(2):
        X = []
        for i in range(len(x) - m + 1 - temp):
            X.append(x[i:i + m + temp])
        X = np.array(X)
        # 计算X任意一行数据与所有行数据对应索引数据的差值绝对值的最大值
        D_value = [] #存储差值
        for i in X:
            sub = []
            for j in X:
                sub.append(max(np.abs(i - j)))
            D_value.append(sub)
        # 计算阈值
        F = r * np.std(x, ddof=1)
        # 判断D_vaLue中的每一行中的值比阈值小的个数除以Len(x ) -m+1的比例
        num = np.sum(D_value < F, axis=1) / (len(x) - m + 1 - temp)
        # 计算num的对数平均值
        Lm = np.average(np.log(num))
        entropy = abs(entropy) - Lm

    return entropy
