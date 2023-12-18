#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/11 14:49
# @Author  : 我的名字
# @File    : NUAA.py.py
# @Description : 这个函数是用来balabalabala自己写
import numpy as np
import pandas as pd

from NUAADataset import *
import torch
import matplotlib as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

device = torch.device("cuda")

# NUAA()

# train_loader = torch.utils.data.DataLoader(dataset,
#                                             batch_size=1,
#                                             shuffle=False,
#                                             num_workers=1,
#                                             pin_memory=True)
#
'''
if __name__ == '__main__':
    dataset = NUAADataset()
    # for step, (X, y) in enumerate(train_loader):
    #     X = X.to(device=device, dtype=torch.float32)
    #     y = y.unsqueeze(1).to(device=device, dtype=torch.float32)
    #     print(X,y)
    # Step 1: Extract features and labels
    # data_loader = NUAADataset()
    all_features = []
    all_labels = []
    for data, label in dataset:
        feature = np.array(data).reshape(-1)
        # np.reshape(feature,(1,16*8))
        all_features.append(feature)
        all_labels.append(label)

    # all_features = torch.stack(all_features)  # Convert the list of tensors to a single tensor
    # all_labels = torch.tensor(all_labels)

    df = pd.DataFrame(all_features)
    # df = df.rolling(window=15, min_periods=1).mean()

    # Save the DataFrame to an Excel file
    output_file_path = "./NUAA/nuaa_test.xlsx"
    df.to_excel(output_file_path, index=False)

    print("Selected features dataset has been saved to:", output_file_path)

    # # Step 2: Feature selection using SelectKBest
    num_features_to_select = 10
    selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
    selected_features = selector.fit_transform(df.copy().fillna(-1).reset_index(drop=True), all_labels)
    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)

    W = ['FZ','V1','V2','C','CW']
    Feature = ['absolute_mean_value',
            'max',
            'root_mean_score',
            'Root_amplitude',
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
            'VF']
    selected_feature_names = []
    # Get the names of the selected features
    for i in selected_indices:
        print(i//16,i%16)
        selected_feature_names.append(f"{W[i//16],Feature[i%16]}")
    print("Selected Feature Names:", selected_feature_names)

    # # Get the scores of each feature based on their relationship with the labels
    # feature_scores = selector.scores_
    #
    # # Print the feature scores
    # print("Feature Scores:")
    # for i, score in enumerate(feature_scores):
    #     print(f"Feature_{i + 1}: {score}")
    # Calculate Pearson correlation coefficients between features and labels
    # correlation_matrix = np.corrcoef(all_features.view(all_features.shape[0], -1).T, all_labels)
    #
    # # Extract the correlations between features and the last column (labels)
    # correlation_with_labels = correlation_matrix[:-1, -1]
    #
    # # Print the Pearson correlation coefficients
    # print("Pearson Correlation Coefficients:")
    # for i, correlation in enumerate(correlation_with_labels):
    #     print(f"Feature_{i + 1}: {correlation}")

    # Step 3: Create the new dataset with selected features

    # # Convert the selected features back to tensors
    # selected_features = torch.tensor(selected_features, dtype=torch.float32)
    #
    # # Create the new dataset
    # selected_features_dataset = PHM2010_Selected_Features_Dataset(selected_features, all_labels)

    # Convert selected_features to a pandas DataFrame

    df_selected_features = pd.DataFrame(selected_features,
                                        columns=[f"{W[i//16],Feature[i%16]}" for i in selected_indices])

    # Apply sliding window filter to each feature using pandas rolling mean
    # df_filtered_features = df_selected_features.rolling(window=5, min_periods=1).mean()

    # pca = PCA(n_components=10)  # 加载PCA算法，设置降维后主成分数目为2
    # reduced_x = pca.fit_transform(df_selected_features)  # 对样本进行降维

    # Normalize the selected features
    scaler = StandardScaler()
    normalized_selected_features = scaler.fit_transform(df_selected_features)

    # scaler = MinMaxScaler()
    # minmax_selected_features = scaler.fit_transform(normalized_selected_features)


    # Convert the normalized features back to tensors
    minmax_selected_features = torch.tensor(normalized_selected_features, dtype=torch.float32)

    # Create the new dataset with normalized features
    # selected_features_dataset = PHM2010_Selected_Features_Dataset(normalized_selected_features, all_labels)

    data_dict = {
        f"Feature_{i + 1}": minmax_selected_features[:, i] for i in range(10)
    }
    data_dict["Label"] = all_labels

    df = pd.DataFrame(data_dict)

    # Save the DataFrame to an Excel file
    output_file_path = "./NUAA/NUAA_Feature_ALL1.xlsx"
    df.to_excel(output_file_path, index=False)

    print("Selected features dataset has been saved to:", output_file_path)

    # X = df.iloc[:, :20]  # 特征列
    # y = df.iloc[:, 20]  # 标签列
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # model = MLPRegressor(hidden_layer_sizes=(20,10,8,2),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    # learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=50000, shuffle=True)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X)
    # # print(y_pred)
    # # 将y_pred转换为DataFrame并指定列名
    # y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_Label'])
    #
    # # 将原始DataFrame df 与 y_pred_df 连接（沿列方向连接）
    # df = pd.concat([df, y_pred_df], axis=1)
    # output_file_path = "./NUAA/NUAA_Feature_pred.xlsx"
    # df.to_excel(output_file_path, index=False)
'''

df = pd.read_excel('./NUAA/NUAA_Feature_ALL1.xlsx')
scaler = MinMaxScaler()
minmax_selected_features = scaler.fit_transform(df)
df = pd.DataFrame(minmax_selected_features)
output_file_path = "./NUAA/NUAA_Feature_ALL_minmax.xlsx"
df.to_excel(output_file_path, index=False)

