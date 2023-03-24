# https://github.com/issamemari/pytorch-multilabel-balanced-sampler
import random
import numpy as np
import torch
import copy

from torch.utils.data.sampler import Sampler, WeightedRandomSampler

def create_single_class_sampler(labels):
    class_counts = 1 / np.bincount(labels)
    samples_weight = np.array([class_counts[l] for l in labels])
    return WeightedRandomSampler(samples_weight, len(samples_weight))

class ClassCycler:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.classes = []
        self._update_class_list()
    
    def _update_class_list(self):
        self.classes.extend(np.random.choice(self.num_classes, self.num_classes, replace=False))

    def next_n(self, n):
        if n > len(self.classes):
            self._update_class_list()
        out = self.classes[:n]
        self.classes = self.classes[n:]
        return out

class FewShotBatchSampler(Sampler):
    def __init__(self, labels, k_shot, n_ways=None, include_query=False):
        self.labels = labels
        self.num_classes = self.labels.shape[1]
        self.n_ways = n_ways
        if self.n_ways is None:
            self.n_ways = self.num_classes
        self.shots = k_shot
        inds, class_indices = np.nonzero(labels)
        self.class_indices = {}
        total_batches = 0
        for c in range(self.num_classes):
            indices = inds[class_indices == c]
            self.class_indices[c] = indices
            total_batches += int(np.ceil(len(indices) / k_shot))
        self.iterations = total_batches // self.n_ways # Average batches per class
        self.include_query = include_query
        self.cycler = None
        self.n_query = self.shots

    def set_query_size(self, size):
        self.n_query = size

    def get_iteration_classes(self, iter):
        if self.num_classes == self.n_ways:
            classes = np.arange(self.num_classes)
        else:
            if self.cycler is None:
                self.cycler = ClassCycler(self.num_classes)
            classes = self.cycler.next_n(self.n_ways)
        return classes
    
    def generate_batches(self, size):
        class_indices = copy.copy(self.class_indices)
        for c in range(self.num_classes):
            np.random.shuffle(class_indices[c])

        if self.include_query:
            all_indices = set([ind for l in class_indices.values() for ind in l])
    
        for it in range(self.iterations):
            batch = set()

            if self.include_query:
                query_list = []
                support_list = []

            self.curr_classes = self.get_iteration_classes(it)
            curr_classes = sorted(self.curr_classes , key=lambda c: len(self.class_indices[c]))
            for c in curr_classes:
                indices = class_indices[c]
                diff = size
                selected = set()
                if len(indices) > 0:
                    selected = set(indices[:size]) - batch
                    class_indices[c] = indices[size:]
                    diff -= len(selected)
                
                rem = set(self.class_indices[c]) - batch - selected
                diff = min(len(rem), diff)
                if diff > 0:
                    selected.update(np.random.choice(list(rem), diff, replace=False))

                batch.update(selected)
                if self.include_query:
                    support_list.extend([selected.pop() for _ in range(self.shots)])
                    query_list.extend(selected)
            
            if self.include_query:
                size_support = self.n_ways * self.shots
                if len(support_list) < size_support:
                    selected = np.random.choice(list(all_indices - batch), size_support - len(support_list), replace=False)
                    support_list.extend(selected)
                    batch.update(selected)

                size_query  =  self.n_ways * self.n_query
                if len(query_list) < size_query:
                    selected = np.random.choice(list(all_indices - batch), size_query - len(query_list), replace=False)
                    support_list.extend(selected)
                    batch.update(selected)
                yield(support_list+query_list)
            else:
                yield list(batch)
            
    def __iter__(self):
        size = self.shots 
        if self.include_query:
            size += self.n_query
        return self.generate_batches(size)
    
    def __len__(self):
        return self.iterations
    
class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)