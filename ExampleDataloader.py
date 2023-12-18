#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 13:39
# @Author  : liu
# @File    : ExampleDataloader.py.py
# @Description : 力、振动、电流三类文件的读取和裁剪

from dataloader import *

if __name__ == '__main__':

    '''读取文件夹地址,写入自己地址'''
    dirForce = 'E:/1研究生/9.实验/5.T05/1.力'
    dirVibration = 'E:/1研究生/9.实验/5.T05/2.振动'
    dirCurrent = 'E:/1研究生/9.实验/5.T05/3.电流'
    '''输出文件夹地址,写入自己地址'''
    dirOut = 'E:/1研究生/9.实验/5.T05/6.数据集'

    # 如果需要批量处理程序可通过for循环来实现
    for num in range(26,36):

    # num = 8    # 文件名设置， 第num次切削数据

        VibrationFilename = f'T05-{num:02}.xlsx'
        ForceFilename = f'T05-{num:02}.txt'
        CurrentFilename = f'{num:03}.mat'
        OutFilename = f'T05-{num:03}.xlsx'
        print(f"VibrationFilename:{VibrationFilename}\nForceFilename:{ForceFilename}\nCurrentFilename:{CurrentFilename}")

        window=50000    # 裁剪宽度

        '''
        初始化DataLoad，传入地址和文件名，设置裁剪宽度，是否显示图像，是否保存Excel
        此部分用于原始数据的裁剪合并并生成标准数据集，完成裁剪后可将此部分注释
        '''
        test = DataLoader(dirForce, dirVibration, dirCurrent, # 地址
                          ForceFilename, VibrationFilename, CurrentFilename,  # 文件名
                          dirOut, OutFilename,   # 输出地址文件名
                          window, pltshow = True, excelsave = True)   # 是否保存和显示图片

        test.read_and_concat_data()