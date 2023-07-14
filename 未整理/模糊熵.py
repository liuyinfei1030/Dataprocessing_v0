# -*- coding: utf-8 -*-
"""
作者：郑浩
日期：2022年02月28日
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

def Fuzzy_Entropy(x, m, r=0.15, n=2):
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    if len(x) < m + 1:
        raise ValueError("len(x)小于m+1")  # 将x以m为窗口进行划分
    entropy = 0
    for temp in range(2):
        X = []
        for i in range(len(x) - m + 1 - temp):
            X.append(x[i:i + m + temp])
        X = np.array(X)
        D_value = []
        # 计算绝对距离
        for index1, i in enumerate(X):
            sub = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    sub.append(max(np.abs(i - j)))
            D_value.append(sub)
        # 计算模糊隶属度D
        D = np.exp(-np.power(D_value, n) / r)
        Lm = np.average(D.ravel())
        entropy = abs(entropy) - Lm

    return entropy


if __name__ == '__main__':
    # x = [2,1,4,5,6,3,2,1,4,5,6,3,2]
    x = [2, 1, 4, 5, 6, 3, 2, 3, 5, 6, 9, 2, 6]
    # x = pd.read_csv("E:\File\Python\DATAS\_NASA\C1\C1\C_1_004.csv")
    # x = x.iloc[:, 0]
    fig = plt.figure()
    plt.plot(x,'b', label='原始数据')
    plt.show()
    # print(x)
    # print(len(x))
    print(Fuzzy_Entropy(x, 2))
