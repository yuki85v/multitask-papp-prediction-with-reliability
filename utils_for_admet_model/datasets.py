import numpy as np
import torch
from torch.utils.data import Dataset
import dgl


class ADMETDataset(Dataset):
    def __init__(
            self, smiles_to_graph, merge_df, target_col, 
            id_col='COMPID', smiles_col='SMILES'
        ):
        self.compids = []
        self.graphs = []
        self.labels = []
        for _, row in merge_df.iterrows():
            self.compids.append(row[id_col])
            self.graphs.append(smiles_to_graph(row[smiles_col]))
            
            if target_col is not None:
                if isinstance(target_col, (list, tuple)):
                    self.labels.append(np.array(row[target_col], dtype=np.float32))
                else:
                    self.labels.append(float(row[target_col]))
            else:
                self.labels.append(np.nan)
    
    def __len__(self):
        return len(self.compids)
    
    def __getitem__(self, idx):
        compid = self.compids[idx]
        graph = self.graphs[idx]
        label = self.labels[idx]
        return compid, graph, label
    

def admet_collate_fn(batch):
    compids, graphs, labels = zip(*batch)
    graphs = dgl.batch(graphs)
    labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    
    if labels.ndim == 1:
        labels = labels.view(-1, 1)
    return compids, graphs, labels
