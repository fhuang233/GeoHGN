# Based on the code from: https://github.com/guaguabujianle/GIGN/blob/main/GIGN/dataset_GIGN.py
# %%
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from torch_geometric.data import Batch, Data


#def collate_fn(batch):
    #return Batch.from_data_list(batch)


class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)
        

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir=None, data_df=None, model_name='GeoHGN', dis_threshold=5, num_process=8):
        self.data_dir = data_dir
        self.data_df = data_df
        self.name = model_name
        self.dis_threshold = dis_threshold
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        if self.data_df is not None:
            self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'].strip(), float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"Graph_{self.name}_{cid}_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        data = torch.load(self.graph_paths[idx])
        data.idx = idx
        return data

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.graph_paths)
    
      
    def __add__(self, other):
        
        new_dataset = GraphDataset()
        
        new_dataset.graph_paths = self.graph_paths + other.graph_paths
        
        return new_dataset
        


class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, max_n_vertex_per_batch=25000) -> None:
        super().__init__()
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset))]
        self.max_n_vertex_per_batch = max_n_vertex_per_batch
        self.total_size = None
        self.batch_indexes = []
        self._form_batch()

    ########## overload with your criterion ##########
    def _form_batch(self):

        np.random.shuffle(self.indexes)
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        cur_vertex_cnt = 0
        batch = []

        for i in self.indexes:
            data = self.dataset[i]
            item_len = data.num_nodes
            if item_len > self.max_n_vertex_per_batch:
                continue
            cur_vertex_cnt += item_len
            if cur_vertex_cnt > self.max_n_vertex_per_batch:
                self.batch_indexes.append(batch)
                batch = []
                cur_vertex_cnt = item_len
            batch.append(i)
        self.batch_indexes.append(batch)

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        else:
            # control the lengths of the dataset, otherwise the dataloader will raise error
            if len(self.batch_indexes) < self.total_size:
                num_add = self.total_size - len(self.batch_indexes)
                self.batch_indexes = self.batch_indexes + last_batch_indexes[:num_add]
            else:
                self.batch_indexes = self.batch_indexes[:self.total_size]

    def __len__(self):
        return len(self.batch_indexes)
    
    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]
    
    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)
    
    
    
    

