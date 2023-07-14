#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 16:20
# @Author  : Liu
# @File    : dataloader.py
# @Description : 这个函数是用来balabalabala自己写

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import glob

class DataLoader:
    """

    """
    def __init__(self, dirForce, dirVibration, dirCurrent, ForceFilename, VibrationFilename, CurrentFilename, dirOut, OutFilename, window, pltshow, excelsave):
        self.dirForce = dirForce
        self.dirVibration = dirVibration
        self.dirCurrent = dirCurrent

        self.ForceFilename = ForceFilename
        self.VibrationFilename = VibrationFilename
        self.CurrentFilename = CurrentFilename

        self.dirOut = dirOut
        self.OutFilename = OutFilename

        '''窗口设置和裁剪区间'''
        self.window = window
        self.selected_start_F = 250000
        self.selected_end_F = 250000 + self.window
        self.selected_start_V = 400000
        self.selected_end_V = 400000 + self.window
        self.selected_start_C = 250000
        self.selected_end_C = 250000 + self.window



        self.pltshow = pltshow
        self.excelsave = excelsave

    def read_and_concat_data(self):
        # 读取三向力文件
        self.dfForce = pd.read_csv(os.path.join(self.dirForce, self.ForceFilename), sep='\t', header=None, encoding='ANSI',
                            usecols=[1, 2, 3], skiprows=20, low_memory=False)

        # 截取所选的行
        self.dfForce_new = self.dfForce.iloc[self.selected_start_F:self.selected_end_F].values
        self.dfForce_new = pd.DataFrame(self.dfForce_new)

        # 读取振动文件
        self.dfVibration = pd.read_excel(os.path.join(self.dirVibration, self.VibrationFilename), sheet_name='Data1',
                                    usecols=[1, 2, 3, 4, 5, 6, 7, 8], skiprows=2, header=None)
        self.dfVibration_new = self.dfVibration.iloc[self.selected_start_V:self.selected_end_V].values
        self.dfVibration_new = pd.DataFrame(self.dfVibration_new)

        # 读取电流文件
        Current_data = scipy.io.loadmat(os.path.join(self.dirCurrent, self.CurrentFilename))
        self.dfCurrent = pd.DataFrame({'CH03': Current_data['CH03'].squeeze(),
                                        'CH04': Current_data['CH04'].squeeze(),
                                        'CH05': Current_data['CH05'].squeeze()})

        self.dfCurrent_new = self.dfCurrent.iloc[self.selected_start_C:self.selected_end_C].values
        self.dfCurrent_new = pd.DataFrame(self.dfCurrent_new)

        if self.excelsave == True:
            # 拼接数据
            self.result = pd.concat([self.dfForce_new, self.dfVibration_new, self.dfCurrent_new], ignore_index=True, axis=1)
            self.result.columns = ['Fx', 'Fy', 'Fz', 'Vs1', 'Vs2', 'Vs3', 'Vt1', 'Vt2', 'Vt3', 'AE', 'SN', 'ACx', 'ACy', 'ACz']

            # 写入新的excel文件
            self.result.to_excel(os.path.join(self.dirOut, self.OutFilename), index=False, header=True)
            print(f'{self.OutFilename}生成完毕')

        if self.pltshow==True:

            # 可视化截取的数据
            plt.figure()
            plt.plot(self.dfForce.iloc[:800000])
            plt.plot(self.dfForce_new)

            # 可视化截取的数据
            plt.figure()
            plt.plot(self.dfVibration)
            plt.plot(self.dfVibration_new)

            plt.figure()
            plt.plot(self.dfCurrent)
            plt.plot(self.dfCurrent_new)

            plt.show()
