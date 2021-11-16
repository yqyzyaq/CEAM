import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh





# class GAT(nn.Module):
#
#     def __init__(self, g, kg, in_dim, hidden_dim, out_dim,num_layers=8):
#         super( GAT, self).__init__()
#
#         self.layers = self.heads = nn.ModuleList()
#         self.layers.append(GATLayer(g, in_dim, hidden_dim))
#
#         for i in range(num_layers-2):
#             self.layers.append((GATLayer(g, hidden_dim, hidden_dim)))
#         self.layers.append(GATLayer(g, hidden_dim, out_dim))
#
#
#         self.kg = kg
#
#     def forward(self, h):
#
#         for layer in self.layers:
#            h = layer(h)
#            h = F.elu(h)
#         return h

''' --------------- GCN ---------------- '''
gcn_msg = fn.copy_src(src='h', out='m')

# gcn_reduce = fn.sum(msg='m', out='h')
gcn_reduce =  fn.mean(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.g = g
        self.linear = nn.Linear(in_feats, out_feats)


    def forward(self,  feature):
        g = self.g
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h =  g.ndata['h']
            return self.linear(h)

    def get_weight(self):
        return self.linear.weight[0][:10]


class GCN(nn.Module):
    def __init__(self, g,g_mirror,in_dim, hidden_dim, out_dim,mask,num_layers=2):
        super(GCN, self).__init__()
        self.name = 'gcn'
        # self.g =  g
        self.layers = nn.ModuleList()
        if mask:
            self.layers.append(MaskedLayer(g, g_mirror, in_dim, hidden_dim))
        else:
            self.layers.append(GCNLayer(g,in_dim, hidden_dim))
        # self.layer1 = GCNLayer(in_dim, hidden_dim)
        for i in range(num_layers -2):
            self.layers.append(GCNLayer(g,hidden_dim, hidden_dim))
        self.layers.append(GCNLayer(g,hidden_dim,out_dim))
        # self.layer2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, h):
        i  = 0

        for layer in self.layers:
            h = F.elu(layer(h))
            # print(i, "th GNN layer", h[0][:5], h[1][:5], h[2][:5])
            i += 1
        return h

    def get_weight(self):
        i = 0
        for layer in self.layers:
            print("weight of ",i, "th GNN layer",  layer.get_weight())
            i+=1

    # def forward(self, g, features):
    #     x = F.relu(self.layer1(g, features))
    #     x = self.layer2(g, x)
    #     return x


''' --------------- GAT ---------------- '''

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        # self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # TODO：
        self.fc = nn.Linear(in_dim, out_dim)
        # equation mix

        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)

        # return {'e': a}

        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        # print("before softmax",nodes.mailbox['e'].detach().numpy()[:3])
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # print('attention shape:', alpha.shape)
        # print("after softmax", alpha[:3])
        # alpha = 10 * nodes.mailbox['e']
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

    def get_weight(self):
        return self.attn_fc.weight[0][:10].detach()

# class GATLayer(nn.Module):
#     def __init__(self, g, in_dim, out_dim):
#         super(GATLayer, self).__init__()
#         self.g = g
#         # 公式 (1)
#         self.fc = nn.Linear(in_dim, out_dim, bias=False)
#         # self.self_fc =  nn.Linear(in_dim, out_dim, bias=True)
#         # 公式 (2)
#         self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
#
#     # message function 消息函数 - 将 attention 以边特征的形式存在edges.data['e'] (for the update function is apply_edges)
#     def edge_attention(self, edges):
#         # 公式 (2) 所需，边上的用户定义函数
#         z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
#         a = self.attn_fc(z2)
#         return {'e' : F.leaky_relu(a)}
#
#     # send node representation and attention weights to node.mailbox (for the update function is update_all)
#     def message_func(self, edges):
#         return {'z' : edges.src['z'], 'e' : edges.data['e']}
#
#     def reduce_func(self, nodes):
#         # normalized attention weights
#         alpha = F.softmax(nodes.mailbox['e'], dim=1)
#         # weighted sum
#         h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
#         #h = torch.sigmoid((torch.self.self_fc(self.g.ndata['z']))
#         return {'h' : h}
#
#     def forward(self, h):
#         # 公式 (1)
#         # print('h shape',h.shape)
#         #print('GATLAyer initialize,',h)
#         # with self.g.local_scope():
#         z = self.fc(h)
#         self.g.ndata['z'] = z
#         # load attention weights as edge features
#         self.g.apply_edges(self.edge_attention)
#         # calculates weighted sum
#         self.g.update_all(self.message_func, self.reduce_func)
#
#         return  self.g.ndata['h']

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))
#

class MaskedLayer(GATLayer):
    def __init__(self,g,g_mirror, in_dim, out_dim, learnable=1):
        super(MaskedLayer,self).__init__(g, in_dim, out_dim)
        self.g_mirror  = g_mirror
        self.learnable = learnable

    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e' : F.leaky_relu(a)}

    def edge_mask(self):
        feat_dim = self.g.ndata['z'].shape[1]
        if self.learnable:
            self.sigma = torch.nn.Parameter(torch.ones(feat_dim), requires_grad=True)
        else:
            self.sigma = torch.ones(feat_dim)
        mask = torch.ones(self.g_mirror.shape[0],feat_dim)
        for i,m_srcs in enumerate(self.g_mirror):
            mask_i = torch.zeros(feat_dim)
            for m_src in m_srcs:
                mask_i = mask_i + (self.g.ndata['z'][self.g.edges()[0][i]] - self.g.ndata['z'][m_src])**2
            # mask[i] = torch.exp(torch.tensor(mask_i))

            mask[i] = torch.exp(-mask_i/(self.sigma**2))
            # mask[i] = torch.exp(-mask_i)

        self.g.edata['m'] = mask



    # send node representation and attention weights to node.mailbox (for the update function is update_all)
    def message_func(self, edges):
        #mask
        return {'z': edges.src['z'], 'e': edges.data['e'],'m':edges.data['m']}

    def reduce_func(self, nodes):
        # normalized attention weights
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # weighted sum
        # print("mask shape",nodes.mailbox['m'].shape, "node shape",nodes.mailbox['z'].shape)
        h = torch.sum(alpha *nodes.mailbox['m']* nodes.mailbox['z'], dim=1)
        # h_ = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': h}

    def forward(self, h):
        # 公式 (1)
        z = self.fc(h)
        self.g.ndata['z'] = z

        self.edge_mask()
        # print('mask of first entity',self.g.edata['m'][0])
        # load attention weights as edge features
        # print('check mask',self.g.edata['m'][363])
        self.g.apply_edges(self.edge_attention)
        # calculates weighted sum
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata['h']

class GNN(nn.Module):
    def __init__(self, g, kg,g_mirror,gnn_type,in_dim, hidden_dim, out_dim, multihead = 1,num_heads=2,mask = 1,learnable = 1,num_layers = 2):
        super( GNN, self).__init__()
        # self.name = 'mixed'
        # print("mirror:",type(g_mirror))
        print("GNN dimension:",in_dim, out_dim)
        self.layers = nn.ModuleList()
        if gnn_type == 'gat':
            self.layers.append(GATLayer(g, in_dim, hidden_dim))
            self.layers.append(GATLayer(g, hidden_dim, out_dim))
        if gnn_type =='gcn':
            self.layers.append(GCNLayer(g, in_dim, hidden_dim))
            self.layers.append(GCNLayer(g, hidden_dim, out_dim))

        # self.layers.append(MaskedLayer(g, g_mirror, in_dim, hidden_dim, learnable))
        # self.layers.append(GCNLayer(g, hidden_dim, out_dim))
        # self.layers.append(GATLayer(g, in_dim, out_dim))
        # self.layers.append(GCNLayer(g, in_dim, hidden_dim))
        # self.layers.append(GCNLayer(g, hidden_dim, hidden_dim))
        # self.layers.append(MaskedLayer(g, g_mirror, hidden_dim, hidden_dim, learnable))
        # self.layers.append(GCNLayer(g, hidden_dim, hidden_dim))
        # self.layers.append(GATLayer(g, hidden_dim, hidden_dim))
        # self.layers.append(MaskedLayer(g, g_mirror, hidden_dim, out_dim, learnable))
        # self.layers.append(GATLayer(g, hidden_dim, out_dim))
        # self.layers.append(GATLayer(g, hidden_dim, hidden_dim))
        # self.layers.append(GCNLayer(g, hidden_dim, out_dim))
        # self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        # self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        i = 0

        for layer in self.layers:
            h = F.elu(layer(h))
            # print(i, "th GNN layer", h[0][:5], h[1][:5], h[2][:5])
            # print("weight:", layer.get_weight())
            i += 1
        return h

    def get_weight(self):
        i = 0
        for layer in self.layers:
            print("weight of ",i, "th GNN layer",  layer.get_weight())
            i+=1

class GAT(nn.Module):
    def __init__(self,g, kg,g_mirror,in_dim, hidden_dim, out_dim, multihead = 1,num_heads=2,mask = 1,learnable = 1,num_layers = 2):
        super(GAT, self).__init__()
        self.name = 'gat'
        # learnable parameter for attribute mask
        # self.learnable = learnable
        self.hidden_dim = hidden_dim
        # self.layers = self.heads = nn.ModuleList()
        self.layers  = nn.ModuleList()
        print("multihead:{}, mask:{}".format(multihead, mask))
        # first layer
        if mask:
            self.layers.append(MaskedLayer(g,g_mirror, in_dim, hidden_dim,learnable))
            if multihead:
                self.layers.append(MultiHeadGATLayer(g, hidden_dim, hidden_dim, num_heads))
                for i in range(num_layers - 3):
                    self.layers.append((MultiHeadGATLayer(g, hidden_dim * num_heads, hidden_dim, num_heads)))
                self.layers.append(MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1))
            else:
                for i in range(num_layers-2):
                    self.layers.append(GATLayer(g, hidden_dim, hidden_dim))
                self.layers.append(GATLayer(g, hidden_dim, out_dim))
        else: # without mask
            if multihead:
                self.layers.append(MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads))
                for i in range(num_layers - 2):
                    self.layers.append((MultiHeadGATLayer(g, hidden_dim * num_heads, hidden_dim, num_heads)))
                self.layers.append(MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1))
            else:
                self.layers.append(GATLayer(g, in_dim, hidden_dim))
                for i in range(num_layers - 2):
                    self.layers.append(GATLayer(g, hidden_dim, hidden_dim))
                self.layers.append(GATLayer(g, hidden_dim, out_dim))

        self.g = g
        self.kg = kg

    def check_mask_status(self):
        return self.g.edata['m']

    def forward(self, h):
        for i, layer in enumerate(self.layers):
           # print('before:', i, h[0], h[0].shape)
           h = layer(h)

           h = F.elu(h)
           print(i, "th GNN layer", h[0][:5], h[1][:5],h[2][:5])
           # print("after elu, node 0 representation:", h[0][:10])
        return h
