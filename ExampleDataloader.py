#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 13:39
# @Author  : 我的名字
# @File    : ExampleDataloader.py.py
# @Description : 这个函数是用来balabalabala自己写

from dataloader import *


if __name__ == '__main__':

    '''文件夹地址'''
    dirForce = 'E:/1研究生/9.实验/0.T00/1.力'
    dirVibration = 'E:/1研究生/9.实验/0.T00/2.振动'
    dirCurrent = 'E:/1研究生/9.实验/0.T00/3.电流'
    dirOut = 'E:/1研究生/9.实验/0.T00/6.数据集'


    # 如果需要批量处理程序可通过for循环来实现
    # for num in range

    num = 15    # 文件名设置， 第num次切削数据

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
    test = DataLoader(dirForce,dirVibration,dirCurrent,ForceFilename,VibrationFilename,CurrentFilename,dirOut,OutFilename,window,pltshow = False,excelsave = False)
    test.read_and_concat_data()