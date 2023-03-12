import numpy as np

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