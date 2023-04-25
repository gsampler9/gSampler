import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import tqdm


class GraphConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats, n_classes)

    def forward(self, g, x):
        with g.local_scope():
            g.srcdata['x'] = x
            g.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'y'))
            return self.W(g.dstdata['y'])


class SAGEModel(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(GraphConv(in_size, hid_size))
        for i in range(num_layers - 2):
            self.layers.append(GraphConv(in_size, hid_size))
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

        self.sample_W = nn.Parameter(torch.zeros(
            size=(in_size, hid_size)), requires_grad=True)
        nn.init.xavier_uniform_(self.sample_W.data, gain=1.414)
        self.sample_W2 = nn.Parameter(torch.zeros(
            size=(in_size, hid_size)), requires_grad=True)
        nn.init.xavier_uniform_(self.sample_W2.data, gain=1.414)
        self.sample_a = nn.Parameter(torch.FloatTensor(
            [[10e-3], [10e-3], [10e-1]]), requires_grad=True)

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l == 0:
                with block.local_scope():
                    block.srcdata['x'] = h
                    block.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'y'))
                    self.X1 = nn.Parameter(block.dstdata['y'])
                h = layer.W(self.X1)
            else:
                h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, feat, nid):
        """Conduct layer-wise inference to get all the node embeddings."""
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(g, nid, sampler, device=device,
                                batch_size=batch_size, shuffle=False, drop_last=False,
                                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(nid.numel(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                            device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for it, (input_nodes, output_nodes, blocks) in enumerate(tqdm.tqdm(dataloader)):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[it * batch_size: min((it + 1) * batch_size,
                                       nid.numel())] = h.to(buffer_device)
            feat = y
        return y
