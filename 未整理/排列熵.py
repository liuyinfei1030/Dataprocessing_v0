# -*- coding: utf-8 -*-
"""
作者：郑浩
日期：2022年02月28日
"""
import numpy as np
from math import factorial
def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    x = np.array(time_series)  #x = [4, 7, 9, 10, 6, 11, 3]
    hashmult = np.power(order, np.arange(order))  #[1 3 9]


    #_embed的作用是生成上图中重构后的矩阵
    #argsort的作用是对下标排序，排序的标准是值的大小
    #比如第3行[9,10,6] 9的下标是0 6是2....， 6最小
    #所以排序后向量的第一个元素是6的下标2... 排完[201]
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')

    #np.multiply 对应位置相乘  hashmult是1 3 9  sum是求每一行的和
    #hashmult一定要保证三个一样的值顺序不同 按位乘起来后 每一行加起来 大小不同 类似赋一个权重
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)  #[21 21 11 19 11]

    # Return the counts
    _, c = np.unique(hashval, return_counts=True)  #重小到大 每个数字出现的次数  #c是[2 1 2]  最小的11出现了2次 19 1次

    p = np.true_divide(c, c.sum())#[0.4 0.2 0.4]  2/5=0.4

    pe = -np.multiply(p, np.log2(p)).sum()  #根据公式
    if normalize:#如果需要归一化
        pe /= np.log2(factorial(order))
    return pe


#将一维时间序列，生成矩阵
def _embed(x, order=3, delay=1):
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T


if __name__ == '__main__':
    x = [4, 7, 9, 10, 6, 11, 3]
    print(permutation_entropy(x, order=3, normalize=True))