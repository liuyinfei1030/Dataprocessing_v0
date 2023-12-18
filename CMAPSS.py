#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 14:32
# @Author  : 我的名字
# @File    : CMAPSS.py.py
# @Description : 这个函数是用来balabalabala自己写
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/11 14:49
# @Author  : 我的名字
# @File    : NUAA.py.py
# @Description : 这个函数是用来balabalabala自己写
import numpy as np

from CMAPSSDataset import *
import torch
import matplotlib as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

device = torch.device("cuda")

if __name__ == '__main__':
    dataset = CMAPSSDataset()

    all_settings = []
    all_features = []
    all_labels = []
    for setting, data, label in dataset:
        set = np.array(setting).reshape(-1)
        feature = np.array(data).reshape(-1)
        # np.reshape(feature,(1,16*8))
        all_settings.append(set)
        all_features.append(feature)
        all_labels.append(label)

    set_df = pd.DataFrame(all_settings)
    df = pd.DataFrame(all_features)
    lab_df = pd.DataFrame(all_features)
    # df = df.rolling(window=15, min_periods=1).mean()

    # Save the DataFrame to an Excel file
    output_file_path = "CMAPSS_test.xlsx"
    df.to_excel(output_file_path, index=False)

    print("Selected features dataset has been saved to:", output_file_path)

    # # Step 2: Feature selection using SelectKBest
    num_features_to_select = 10
    selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
    selected_features = selector.fit_transform(df.copy().fillna(-1).reset_index(drop=True), all_labels)
    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)



    df_selected_features = pd.DataFrame(selected_features,
                                        columns=[i for i in selected_indices])

    # Apply sliding window filter to each feature using pandas rolling mean
    # df_filtered_features = df_selected_features.rolling(window=5, min_periods=1).mean()

    # Normalize the selected features
    scaler = MinMaxScaler()
    normalized_selected_features = scaler.fit_transform(df_selected_features)

    data_dict = pd.merge(set_df, normalized_selected_features, how='left', on='alpha')

    # Convert the normalized features back to tensors
    normalized_selected_features = torch.tensor(normalized_selected_features, dtype=torch.float32)
    data_dict = pd.merge(normalized_selected_features, lab_df, how='left', on='alpha')
    # data_dict = {
    #     f"Feature_{i + 1}": normalized_selected_features[:, i] for i in range(num_features_to_select)
    # }
    # data_dict["Label"] = all_labels
    #

    # df = pd.DataFrame(data_dict)

    # Save the DataFrame to an Excel file
    output_file_path = "CMAPSS_Feature_ALL.xlsx"
    data_dict.to_excel(output_file_path, index=False)

    print("Selected features dataset has been saved to:", output_file_path)





