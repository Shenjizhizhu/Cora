import torch
import numpy as np
import scipy.sparse as sp
import os
from torch.utils.data import Dataset

class CORADataset(Dataset):
    def __init__(self, root='./'):
        content = np.genfromtxt(os.path.join(root, 'Cora.content'), dtype=str)
        content = content[np.argsort(content[:,0].astype(np.int32))]
        cites = np.genfromtxt(os.path.join(root, 'Cora.cites'), dtype=np.int32)
        
        self.features = content[:,1:-1].astype(np.float32) / np.maximum(content[:,1:-1].astype(np.float32).sum(1, keepdims=True), 1)

        self.labels = np.unique(content[:, -1], return_inverse=True)[1].astype(np.int64)
        
        edges = np.searchsorted(content[:,0].astype(np.int32), cites) 
        edges_bi = np.vstack([edges, edges[:,::-1]])
        self.adj = sp.coo_matrix((np.ones(np.unique(edges_bi, axis=0).shape[0]), 
                                  tuple(np.unique(edges_bi, axis=0).T)),
                                 shape=(len(self.labels), len(self.labels)))

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

    def __len__(self):
        return len(self.labels)