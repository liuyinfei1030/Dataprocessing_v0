#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/22 14:09
# @Author  : 我的名字
# @File    : PHM2010.py.py
# @Description : PHM2010

from PHM2010Dataset import *
import torch
import matplotlib as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

device = torch.device("cuda")

dataset = PHM2010_Dataset()

train_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            pin_memory=True)

if __name__ == '__main__':
    # Step 1: Extract features and labels
    data_loader = PHM2010_Dataset()
    all_features = []
    all_labels = []
    for data, label in data_loader:
        all_features.append(data)
        all_labels.append(label)

    all_features = torch.stack(all_features)  # Convert the list of tensors to a single tensor
    all_labels = torch.tensor(all_labels)

    # Step 2: Feature selection using SelectKBest
    num_features_to_select = 20
    selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
    selected_features = selector.fit_transform(all_features.view(all_features.shape[0], -1), all_labels)
    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get the names of the selected features
    selected_feature_names = [f"Feature_{i + 1}" for i in selected_indices]
    print("Selected Feature Names:", selected_feature_names)

    # Step 3: Create the new dataset with selected features
    class PHM2010_Selected_Features_Dataset(Dataset):
        def __init__(self, features, labels, transform=None):
            self.features = features
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.features)

        def __getitem__(self, index):
            data = self.features[index]
            label = self.labels[index]
            return data, label

    # Convert selected_features to a pandas DataFrame
    df_selected_features = pd.DataFrame(selected_features,
                                        columns=[f"Feature_{i + 1}" for i in range(num_features_to_select)])

    # Apply sliding window filter to each feature using pandas rolling mean
    # df_filtered_features = df_selected_features.rolling(window=5, min_periods=1).mean()


    # Normalize the selected features
    scaler = MinMaxScaler()
    normalized_selected_features = scaler.fit_transform(df_selected_features)

    # Convert the normalized features back to tensors
    normalized_selected_features = torch.tensor(normalized_selected_features, dtype=torch.float32)



    # Create the new dataset with normalized features
    selected_features_dataset = PHM2010_Selected_Features_Dataset(normalized_selected_features, all_labels)

    data_dict = {
        f"Feature_{i + 1}": normalized_selected_features[:, i] for i in range(num_features_to_select)
    }
    data_dict["Label"] = all_labels

    df = pd.DataFrame(data_dict)

    # Save the DataFrame to an Excel file
    output_file_path = "./Data/PHMall.xlsx"
    df.to_excel(output_file_path, index=False)

    print("Selected features dataset has been saved to:", output_file_path)

