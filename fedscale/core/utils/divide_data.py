# -*- coding: utf-8 -*-
from cProfile import label
from random import Random
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import random, csv
from collections import defaultdict
#from argParser import args


class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, args, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets
        self.args = args
        self.isTest = isTest
        np.random.seed(seed)

        self.data_len = len(self.data)
        self.task = args.task
        self.numOfLabels = numOfClass
        self.client_label_cnt = defaultdict(set)

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    def getClientLen(self):
        return len(self.partitions)

    def getClientLabel(self):
        return [len(self.client_label_cnt[i]) for i in range( self.getClientLen())]

    def trace_partition(self, data_map_file):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")

        clientId_maps = {}
        unique_clientIds = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_clientIds:
                        unique_clientIds[client_id] = len(unique_clientIds)

                    clientId_maps[sample_id] = unique_clientIds[client_id]
                    self.client_label_cnt[unique_clientIds[client_id]].add(row[-1])
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]

        for idx in range(sample_id):
            self.partitions[clientId_maps[idx]].append(idx)


    def partition_data_helper(self, num_clients, iid: bool=True, balanced: bool = False, num_part_label: int = -1):
        # partition data according to modes
        if self.isTest:
            logging.info(f"Random Partition for testing")
            self.uniform_partition(num_clients=num_clients)
        if iid is True:
            logging.info(f"IID partition data")
            self.uniform_partition(num_clients=num_clients)
        elif balanced is False:
            logging.info(f"NON-IID partition data: each clients has unbalanced data of all classes ")
            self.unbalanced_whole_label_partition()(num_clients)
        elif num_part_label != -1:
            logging.info(f"NON-IID partition data: each clients has balanced data of {num_part_label} classes")
            self.balanced_skew_label_partition(num_clients, num_part_label)
        else:
            self.uniform_partition(num_clients)

    def uniform_partition(self, num_clients):
        # random partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1./num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
    
    def balanced_skew_label_partition(self, num_clients, num_part_labels):
        for _ in range(num_clients):
            part_label_len = int(1. / num_clients / num_part_labels * self.getDataLen())
            local_label_idx = self.rng.sample(list(range(self.numOfLabels)), k=num_part_labels)
            logging.info(f"{local_label_idx}")
            selected_label_idx = []
            for label in local_label_idx:
                l = [i for i in range(len(self.labels)) if self.labels[i] == label]
                self.rng.shuffle(l)
                l = l[0:part_label_len]
                selected_label_idx += l
            self.partitions.append(selected_label_idx)

    def unbalanced_whole_label_partition(self, num_clients):
        for _ in num_clients:
            part_label_len = int(1. / num_clients * self.getDataLen())
            label_prop = self.rng.choices(list(range(10)), k=self.numOfLabels)
            label_prop_normed = [label_p / sum(label_prop) for label_p in label_prop]
            label_num = [part_label_len * label_p_n for label_p_n in label_prop_normed]
            selected_label_idx = []
            for i in range(self.numOfLabels):
                l = [i for i in range(len(self.labels)) if self.labels[i] == label]
                self.rng.shuffle(l)
                l = l[0:label_num[i]]
                selected_label_idx += l
            self.partitions.append(selected_label_idx)

    def use(self, partition, istest):
        resultIndex = self.partitions[partition]

        exeuteLength = len(resultIndex) if not istest else int(len(resultIndex) * self.args.test_ratio)
        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)


    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}


def select_dataset(rank, partition, batch_size, args, isTest=False, collate_fn=None):
    """Load data given client Id"""
    partition = partition.use(rank - 1, isTest)
    dropLast = False if isTest else True
    num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)
    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)


