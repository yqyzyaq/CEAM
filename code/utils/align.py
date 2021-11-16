import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class HingeLoss(nn.Module):
    def __init__(self, margin=1):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self,h):
        # TODO: add mask
        return torch.sum(F.relu(h))

class HingeDistLoss(nn.Module):
    def __init__(self, margin=10):
        super(HingeDistLoss, self).__init__()
        self.margin = margin

    def forward(self, h,lid,rid,labels):
        dist = torch.norm(h[lid]-h[rid], dim=1, p=2)/torch.sqrt(torch.norm(h[lid],p=2)*torch.norm(h[rid],p=2))
        # print(labels)
        # print(dist.shape)
        w = torch.ones(dist.shape[0])

        w[labels==0] = - 1
        print("distance: ",labels, dist, torch.sum(w * dist))
        return F.relu(torch.sum(w * dist) + self.margin)


class EmbeddingLayer(nn.Module):
    def __init__(self, kg, dim):
        super(EmbeddingLayer, self).__init__()
        self.layer = nn.Embedding(kg.vocab_size, dim)

    def forward(self,h):
        # print(h[0],h[1])
        h = self.layer(h)
        h = torch.flatten(h, start_dim=1, end_dim=2)
        return h

class DecisionLayer(nn.Module):
    def __init__(self, in_dim, h_dim, nn_type, mode):
        super(DecisionLayer, self).__init__()
        self.nn_type = nn_type
        self.mode = mode
        if mode == 'concat':
            self.in_dim = 2 * in_dim
        elif mode == 'diff':
            self.in_dim = in_dim
        elif mode == 'multi':
            self.in_dim = 3 * in_dim
        if nn_type == 'nn':
            # self.input = nn.Linear(self.in_dim, in_dim)
            self.h1 = nn.Linear(self.in_dim, h_dim)
            self.out = nn.Linear(h_dim, 1)

        elif nn_type == 'cnn':
            self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=5)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5)
            self.pool2 = nn.MaxPool1d(2)
            self.out = nn.Linear(144, 1)

    def forward(self, h,lid,rid):
        h_lr =torch.cat([h[lid], h[rid]], dim=1)
        h_diff = h[lid] - h[rid]
        if self.nn_type == 'nn':
            if self.mode == 'concat':
                h = h_lr
            elif self.mode == 'diff':
                h = h_diff
            elif self.mode == 'multi':
                h = torch.cat([h_lr, h_diff], dim=1)
            #h = F.relu(self.input(h))
            h =self.h1(h)

        elif self.nn_type == 'cnn':
            h = torch.unsqueeze(h, 1)
            # print("0",h.shape)
            if self.mode == 'multi':
                h_diff = h[lid] - h[rid]
                h_lr = torch.cat([h[lid], h[rid]], dim=1)
                h = torch.cat([h_lr, h_diff], dim=1)
            h = self.conv1(h)
            h = self.pool1(F.relu(h))
            h = self.pool2(F.relu(self.conv2(h)))
            h = h.view(h.size(0),-1)
        return self.out(F.relu(h))

class AlignNet(nn.Module):
    def __init__(self, g, kg, in_dim, h_dim, mode, gnn = None, nn_type = 'nn', emb_dim = 0):
        super(AlignNet, self).__init__()
        self.g = g
        self.emb_dim = emb_dim
        if emb_dim:
            print("embedding layer..")
            self.emblayer = EmbeddingLayer(kg.vocab_size, emb_dim)

        self.gnn = gnn

        if not self.gnn:
            self.layer = nn.Linear(2*in_dim, in_dim)
        self.out = DecisionLayer(in_dim, h_dim, nn_type, mode)



    def forward(self, g, lid, rid):
        # homo graph
        if len(g.ntypes) == 1:
            h = g.ndata['x']
            # Embedding layer (default not used)
            if self.emb_dim:
                h = self.emblayer(h)
            # GNN layers for node embedding
            if self.gnn:
                # if self.gnn.name == 'gat':
                #     h =  h
                h = self.gnn(h)

                # if self.gnn.name == 'gat':
                #     h = 1 * h
            else:
                h = F.relu(self.layer(h))
                # print("Error: GNN not found")
            # Output layer

            h = self.out(h, lid,rid)
            return h

        # hetero graph
        elif len(g.ntypes)>1:
            h = self.gnn(g)
            h  = self.out(h, lid, rid)

            return h
        # return self.fc(h)
