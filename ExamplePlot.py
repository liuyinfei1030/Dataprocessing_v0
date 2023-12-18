#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 18:25
# @Author  : liu
# @File    : ExamplePlot.py.py
# @Description : JUST铣削试验出图
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

if __name__ == '__main__':
    # 读取Excel文件
    file_names = ["./FeatureExtraction/T03-FeatureExtraction.xlsx",
                  "./FeatureExtraction/T04-FeatureExtraction.xlsx",
                  "./FeatureExtraction/T05-FeatureExtraction.xlsx"]

    dfs = []  # 用于存储所有的DataFrame
    for file_name in file_names:
        xls = pd.ExcelFile(file_name)
        df_list = [xls.parse(sheet_name) for sheet_name in xls.sheet_names]
        dfs.extend(df_list)



    # 绘制多个图形
    num_files = len(file_names)
    num_sheets = len(dfs)

    featurename = ['file',
    'signal',
    'absolute_mean_value',
    'max',
    'root_mean_score',
    'root_amplitude',
    'skewness',
    'Kurtosis_value',
    'shape_factor',
    'pulse_factor',
    'skewness_factor',
    'crest_factor',
    'clearance_factor',
    'Kurtosis_factor',
    'FC',
    'MSF',
    'RMSF',
    'VF',
    'ret1',
    'ret2',
    'ret3',
    'ret4',
    'ret5',
    'ret6',
    'ret7',
    'ret8']
    sensor = ['ACx', 'ACy', 'AE','ACz','Fx', 'Fy', 'Fz', 'SN', 'Vs1', 'Vs2', 'Vs3', 'Vt1', 'Vt2', 'Vt3']
    # 设置图形的行和列
    num_rows = 4
    num_cols = 6


    for sensor_num in range(0,14):
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        for row in range(num_rows):
            for col in range(num_cols):
                ax = axs[row, col]
                # 在每个坐标系中绘制一个特征信号
                signal_index = row * num_cols + col + 2
                feature_column = featurename[signal_index]
                for i in range(num_files):
                    df = dfs[i*14 + sensor_num]
                    ax.plot(df[feature_column],label=f"T0{i+1}")

                ax.set_title(f"{feature_column}")
                # ax.legend()

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels,loc="upper right", bbox_to_anchor=(1.0, 0.9))

        # 调整布局
        # plt.tight_layout()
        plt.suptitle(f'{sensor[sensor_num]}')
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(f'./figs/{sensor[sensor_num]}.png',dpi=750, bbox_inches = 'tight')
        # 显示图形
        # plt.show()

