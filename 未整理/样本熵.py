# -*- coding: utf-8 -*-
"""
作者：郑浩
日期：2022年02月28日
"""

import numpy as np
import matplotlib.pyplot as plt


def Sample_Entropy(x, m, r=0.15):
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")

    if len(x) < m + 1:
        raise ValueError("len(x)小于m+1")  # 将x以m为窗口进行划分
    entropy = 0
    for temp in range(2):
        X=[]
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        D_value = []
        for index1,i in enumerate(X):
            sub=[]
            for index2,j in enumerate(X):
                if index1 != index2:
                    sub.append(max(np.abs(i-j)))
            D_value.append(sub)
        F = r * np.std(x,ddof=1)
        num = np.sum(D_value<F,axis=1)/(len(X)-m+1-temp)
        Lm = np.average(np.log(num))
        entropy = abs(entropy) - Lm

    return entropy