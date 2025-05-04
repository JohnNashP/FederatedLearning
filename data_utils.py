from collections import defaultdict
import torch
from torch.utils.data import Subset, DataLoader
import numpy as np

def create_iid_partitions(dataset, num_clients):
    data_per_client = len(dataset) // num_clients
    indices = torch.randperm(len(dataset))
    return [Subset(dataset, indices[i * data_per_client:(i + 1) * data_per_client]) for i in range(num_clients)]

def create_non_iid_partitions(dataset, num_clients, classes_per_client=2):
    targets = np.array(dataset.targets)
    class_indices = {label: np.where(targets == label)[0] for label in np.unique(targets)}
    client_indices = [[] for _ in range(num_clients)]
    all_labels = list(class_indices.keys())
    for i in range(num_clients):
        selected_labels = np.random.choice(all_labels, classes_per_client, replace=False)
        for label in selected_labels:
            chosen = np.random.choice(class_indices[label], len(class_indices[label]) // num_clients, replace=False)
            client_indices[i].extend(chosen)
            class_indices[label] = list(set(class_indices[label]) - set(chosen))
    return [Subset(dataset, inds) for inds in client_indices]