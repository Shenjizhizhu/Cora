import torch, numpy as np, scipy.sparse as sp, os
from torch.utils.data import Dataset

class CORADataset(Dataset):
    def __init__(self, root='./'):
        content = np.genfromtxt(os.path.join(root, 'Cora.content'), dtype=str)
        content = content[np.argsort(content[:,0].astype(np.int32))]
        cites = np.genfromtxt(os.path.join(root, 'Cora.cites'), dtype=np.int32)
        
        feats = content[:,1:-1].astype(np.float32)
        self.features = feats / np.maximum(feats.sum(1, keepdims=True), 1)
        
        self.labels = np.unique(content[:,-1], return_inverse=True)[1].astype(np.int64)
        
        edges = np.searchsorted(content[:,0].astype(np.int32), cites)
        u_edges = np.unique(np.vstack([edges, edges[:,::-1]]), axis=0)
        self.adj = sp.coo_matrix((np.ones(len(u_edges)), (u_edges[:,0], u_edges[:,1])), shape=(len(self.labels),)*2)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], torch.float), torch.tensor(self.labels[idx], torch.long)

    def __len__(self):
        return len(self.labels)