import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'u'))
        g.ndata['u'] = g_rev.ndata['u']
        g.apply_edges(lambda edges: {
                      'w': edges.data[weight] / torch.sqrt(edges.src['u'] * edges.dst['v'])})
        return g.edata['w']


class GraphConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = self.W(x)
            g.edata['w'] = w
            g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            return g.dstdata['y'], 0


class GraphConvReg(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.out_dim = n_classes
        self.W = nn.Linear(in_feats, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = self.W(x)
            g.edata['w'] = w
            g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))

            mean_x = torch.mean(g.dstdata['y'], dim=0)
            mean_u = g.srcdata['u_sum'] / g.number_of_dst_nodes()
            diff = mean_u.view(-1, 1) * g.srcdata['x'] - mean_x
            reg_loss = torch.sum(
                diff * diff) / (g.number_of_src_nodes() * self.out_dim)
            return g.dstdata['y'], reg_loss


class SAGEConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = x
            g.dstdata['x'] = x[:g.number_of_dst_nodes()]
            g.edata['w'] = w
            g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'y'))
            h = torch.cat([g.dstdata['x'], g.dstdata['y']], 1)
            return self.W(h), 0


class SAGEConvReg(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.out_dim = n_classes
        self.W = nn.Linear(in_feats * 2, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = x
            g.dstdata['x'] = x[:g.number_of_dst_nodes()]
            g.edata['w'] = w
            g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'y'))
            h = self.W(torch.cat([g.dstdata['x'], g.dstdata['y']], 1))

            mean_x = torch.mean(g.dstdata['y'], dim=0)
            mean_u = g.srcdata['u_sum'] / g.number_of_dst_nodes()
            diff = mean_u.view(-1, 1) * g.srcdata['x'] - mean_x
            reg_loss = torch.sum(
                diff * diff) / (g.number_of_src_nodes() * self.out_dim)
            return h, reg_loss


class Model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_feats, n_hidden))
        for i in range(n_layers - 2):
            self.convs.append(GraphConv(n_hidden, n_hidden))
        self.convs.append(GraphConvReg(n_hidden, n_classes))
        self.W_g = torch.nn.parameter.Parameter(
            nn.init.xavier_normal_(torch.Tensor(in_feats, 2)))

    def forward(self, blocks, x):
        reg_loss = 0
        for i, (conv, block) in enumerate(zip(self.convs, blocks)):
            x, reg = conv(block, x, block.edata['w'])
            reg_loss += reg
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x, reg_loss
