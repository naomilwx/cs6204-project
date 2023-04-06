import numpy as np
import pandas as pd
import pickle

def select_query_set(label_info, classes, num_per_class):
    selected = set()
    query_set = {}
    for c in classes:
        img_ids = label_info[(label_info[c] > 0) & ~(label_info['image_id'].isin(selected))]['image_id']
        selected_ids = np.random.choice(img_ids, num_per_class, replace=False)
        selected.update(selected_ids)
        query_set[c] = selected_ids
    return query_set, selected

def count_classes(info, classes):
    classes_count = {}
    for i in range(len(info)):
        df = info.iloc[i]
        for c in classes:
            if c not in classes_count:
                classes_count[c] = 0
            if df[c] > 0:
                classes_count[c] += 1
    return classes_count

def get_query_and_support_ids(img_info, split_file, split='train'):
    with open(split_file, 'rb') as fp:
        query_split = pickle.load(fp)
    query_image_ids = []
    for ids in query_split.values():
        query_image_ids.extend(ids)
    support_image_ids = img_info[(img_info['meta_split'] == split) & ~img_info['image_id'].isin(query_image_ids)]['image_id'].to_list()
    return query_image_ids, support_image_ids

class DatasetConfig:
    def __init__(self, img_path, info_path, training_split_path, label_names_map, classes_split_map, mean_std=None):
        self.img_info = pd.read_pickle(info_path)
        self.img_path = img_path
        self.training_split_path = training_split_path
        self.label_names_map = label_names_map
        self.classes_split_map = classes_split_map
        self.mean_std = mean_std

    def set_test_split_path(self, path):
        self.test_split_path = path