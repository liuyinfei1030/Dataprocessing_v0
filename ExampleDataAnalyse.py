#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/18 15:03
# @Author  : liu
# @File    : ExampleDataAnalyse.py.py
# @Description : 这个文件是用来做数据分析：相关性分析
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

if __name__ == '__main__':
    dirfile = "./FeatureExtraction/"
    file_names = ["T03-FeatureExtraction.xlsx","T04-FeatureExtraction.xlsx","T05-FeatureExtraction.xlsx"]
    os.path.join(dirfile, file_names)

    dfs = []  # 用于存储所有的DataFrame
    for file_name in file_names:
        xls = pd.ExcelFile(file_name)
        df_list = [xls.parse(sheet_name) for sheet_name in xls.sheet_names]
        dfs.extend(df_list)

    # all_features = pd.concat(dfs, axis=0, ignore_index=True).reset_index(drop=True)
        #
        # # 假设 'file' 列是目标变量
        # # file = all_features['file']
        # # 按 'sheet_name' 列进行分组并合并相同sheetname的数据
        # merged_features = all_features.groupby('signal').apply(lambda x: x.reset_index(drop=True))
        # sensor = ['ACx', 'ACy', 'AE', 'ACz', 'Fx', 'Fy', 'Fz', 'SN', 'Vs1', 'Vs2', 'Vs3', 'Vt1', 'Vt2', 'Vt3']
        # dffeature=[]
        # for sensor_name in sensor:
        #     subset = merged_features[merged_features['signal'] == sensor_name].reset_index(drop=True)
        #     dffeature.extend(df_list)
        # features = pd.concat(dffeature,axis=1, ignore_index=False)
        # num_features_to_select = 20
        # selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
        #
        # # 使用 fit_transform 选择特征
        # selected_features = selector.fit_transform(all_features.drop('file', axis=1), file)
        #
        # # 获取所选特征的索引
        # selected_indices = selector.get_support(indices=True)
        #
        # # 打印所选特征的索引
        # print("Selected feature indices:")
        # print(selected_indices)

        # selected_features = selector.fit_transform(all_features.view(all_features.shape[0], -1), file)
        # # Get the selected feature indices
        # selected_indices = selector.get_support(indices=True)
        #
        # # Get the names of the selected features
        # selected_feature_names = [f"Feature_{i + 1}" for i in selected_indices]
        # print("Selected Feature Names:", selected_feature_names)