import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class RGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(RGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'mean')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        # print(in_size, out_size)
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class FirstLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(FirstLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            if 'of' in etype:

                # Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            #     G.nodes[srctype].data['Wh_%s' % etype] = Wh
            #     G.update_all(fn.copy_u('x' % etype, 'm'), fn.mean('m', 'h'),etype = etype)
                funcs[etype] = (fn.copy_u('x', 'm'), fn.mean('m', 'h'))
        # print("xx", G.nodes['vul'].data['x'].shape)
        G.multi_update_all(funcs, 'stack')
        # print("xxx", G.nodes['vul'].data['h'].shape)
        for nodetype in G.ntypes:
            if nodetype == 'vul':
                G.apply_nodes(lambda nodes: {'h': torch.mean(nodes.data['h'],dim=1 )}, ntype=nodetype)
            else:
                G.apply_nodes(lambda nodes:{'h':nodes.data['x']},ntype = nodetype)
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        # G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class AggregateLayer(nn.Module):
    def __init__(self, g, kg, use_gpu):
        super(AggregateLayer, self).__init__()
        # W_r for each relation
        self.g = g; self.kg = kg
        self.attr_etypes = [t for t in self.g.etypes if 'of' in t]
        # 2-d list: i-th sublist --  the candidate entity ids of entity i
        self.candidates = kg.candidates
        self.use_gpu = use_gpu

    def edge_attention(self,edges):
        return {'rd1': -edges.src['d1'] / -edges.src['d'], 'rd2': -edges.src['d2'] / edges.src['d']}

    def message_func(self, edges):
        return {'rd1': edges.data['rd1'], 'rd2': edges.data['rd2'],'h':edges.src['x']}
        # def intype_attention(self,edges):

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['rd1'], dim=1)

        alpha_ = F.softmax(nodes.mailbox['rd2'], dim=1)
        # for nodes on G
        h = torch.sum(alpha * nodes.mailbox['h'], dim = 1)
        # for nodes on G'
        h_ = torch.sum(alpha_ * nodes.mailbox['h'], dim = 1)
        # print("Aggregate alpha", alpha.shape, 'aggregate h', nodes.mailbox['h'].shape)
        return {'h':torch.vstack((h[:self.kg.splitvulid], h_[self.kg.splitvulid:]))}
        # def message_func(self):
        # transit edge information to nodes.mailbox
    def relation_aggregate(self):
        funcs = {}
        for t in self.attr_etypes:
            self.g.apply_edges(self.edge_attention, etype = t )
            # print("Aggregate edge attention",self.g.edges[t].data['rd1'].shape)
            funcs[t] = (self.message_func, self.reduce_func)
        self.g.multi_update_all(funcs,'stack')

    def edge_mask(self,H,entype):
        # print("EEEEE")
        for i, cans in enumerate(self.candidates):
            att = []

            mask_i = torch.zeros(H[i].shape)
            diff = torch.zeros((len(cans), H[i].shape[0],H[i].shape[1]))
            if self.use_gpu:
                mask_i = mask_i.cuda()
                diff =diff.cuda()

            for j,c in enumerate(cans):
                diff[j] = H[i]-H[c]
                att.append(-torch.norm(diff[j]))
    

            att = F.softmax(torch.tensor(att), dim=0)

            for j,c in enumerate(cans):
                mask_i += att[j] * diff[j] * diff[j]
            mask_i = torch.exp(-mask_i)

            # print("mmm:",mask_i.shape, H[i].shape)
            # if i ==0:print(mask_i)
            self.g.nodes[entype].data['h'][i] = H[i] * mask_i

    def forward(self, G, feature_dict,entype='vul'):
        # The input is a dictionary of node features for each type

        self.relation_aggregate()
        #
        for nodetype in G.ntypes:
            # print("nodetype:", [nodetype,entype], nodetype==entype)
            if nodetype == entype:

                # print('mask')
                self.edge_mask(G.nodes[nodetype].data['h'], nodetype)

                G.apply_nodes(lambda nodes: {'h': torch.sum(nodes.data['h'],dim=1 )}, ntype=nodetype)
            else:
                G.apply_nodes(lambda nodes:{'h':nodes.data['x']},ntype = nodetype)

        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HGnnLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes,ntypes):
        super(HGnnLayer, self).__init__()
        # W_r for each relation
        self.eweight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })
        self.nweight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in ntypes
        })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.eweight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'z'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'mean')
        for ntype in G.ntypes:
            self_h = self.nweight[ntype](feat_dict[ntype])
            G.nodes[ntype].data['h'] = torch.hstack([self_h, G.nodes[ntype].data['z']])
            # G.apply_nodes(lambda nodes: {'h':nodes.data[]})

        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class PHGATLayer(nn.Module):
    def __init__(self,  in_dim, out_dim, etypes,ntypes,entype='vul'):
        super(PHGATLayer, self).__init__()
        # self.g = g
        print("indim",in_dim,"outdim",out_dim)
        self.ptypes =  ['weakness_name','vendor','product_name','impact']
        self.prate = 0.6#(len(self.ptypes)+1)/(len(g.ntypes)-1)
        # self.gc = gc
        # self.kg = kg
        self.eweight = nn.ModuleDict({
            etype: nn.Linear(in_dim, out_dim, bias=False) for etype in etypes
        })
        self.nweight = nn.ModuleDict({
            name: nn.Linear(in_dim, out_dim) for name in ntypes
        })
        # self.intype_attn = nn.Linear(2*out_dim, 1, bias = False)
        # self.attn_fcs = nn.ModuleDict({etype: nn.Linear(2 * out_dim, 1, bias=False) for etype in etypes})
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for etype, fc in self.eweight.items():
            nn.init.xavier_normal_(fc.weight, gain=gain)
        for ntype, fc in self.nweight.items():
            nn.init.xavier_normal_(fc.weight, gain=gain)
        # for etype, fc in self.attn_fc.items():
        #     nn.init.xavier_normal_(fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.intype_attn.weight, gain=gain)
        # nn.init.xavier_normal_(self.type_attn.weight, gain=gain)

    def edge_attention(self, edges):
        s = torch.cosine_similarity(edges.src['hr'], edges.dst['ht']).reshape(-1,1)
        return {'s': s}

    def message(self, edges):
        return {'a': edges.data['s'], 'h': edges.src['hr']}

    def p_message(self,edges):
        return {'pa':edges.data['s'], 'h':edges.src['hr']}

    def n_message(self,edges):
        return {'na':edges.data['s'],'h':edges.src['hr']}

    def reduce(self,nodes,rt = None):

        # alpha = nodes.mailbox['a'].squeeze(2) / torch.sum(nodes.mailbox['a'], dim=1)
        # alpha = alpha.unsqueeze(2)
        # alpha = alpha.reshape(-1,-1,1)
        # alpha = F.softmax(nodes.mailbox['a'], dim=1)
        # m_alpha = torch.mean(alpha, dim=1)
        # print("Alpha:", alpha.shape, "neighbor:", nodes.mailbox['h'].shape)
        return {'h': torch.sum(nodes.mailbox['a'] * nodes.mailbox['h'], dim = 1),
                'ma':torch.mean(nodes.mailbox['a'], dim=1)}

    def p_reduce(self,nodes):
        # alpha = nodes.mailbox['pa']/torch.sum(nodes.mailbox['pa'], dim=1)
        # alpha = F.softmax(nodes.mailbox['pa'], dim=1) #* self.prate
        # print("Alpha:",alpha.shape, "neighbor:",nodes.mailbox['h'].shape,torch.sum(alpha * nodes.mailbox['h'],dim=1).shape)
        return {'ph': torch.sum(nodes.mailbox['pa'] * nodes.mailbox['h'], dim = 1), 'mpa':torch.mean(nodes.mailbox['pa'], dim=1)}

    def n_reduce(self, nodes):
        # alpha = nodes.mailbox['na'] / torch.sum(nodes.mailbox['na'], dim=1)
        # alpha = F.softmax(nodes.mailbox['na'], dim=1) #*(1- self.prate)
        # print("Alpha:", alpha.shape, "neighbor:", nodes.mailbox['h'].shape)
        return {'nh': torch.sum(nodes.mailbox['na'] * nodes.mailbox['h'], dim=1), 'mna':torch.mean(nodes.mailbox['na'], dim=1)}

    def tar_rel_aggregate(self, nodes):
        pa = F.softmax(nodes.data['mpa'], dim=1) * 0.6
        na = F.softmax(nodes.data['mna'], dim=1) * 0.4
        return {'h': torch.sum(pa * nodes.data['ph'], dim=1)+torch.sum(na * nodes.data['nh'], dim=1)}

    def forward(self, G, feat_dict, entype='vul'):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # phi(j,r) = W_r * h_j
            # print(etype)
            G.nodes[srctype].data['hr'] = self.eweight[etype](feat_dict[srctype])

            G.nodes[dsttype].data['ht'] = self.nweight[dsttype](feat_dict[dsttype])
            # print(dsttype, G.nodes[dsttype].data['Wh'].shape)
            G.apply_edges(self.edge_attention, etype=etype)
            # print("edge attention:", G.edges[etype].data['e'].shape)
            # G.apply_edges(lambda edges:{'sh':edges.src['Wh'],'dh':edges.dst['Wh']})
            if srctype == entype:
                funcs[etype] = (self.message, self.reduce)

            elif srctype in self.ptypes:
                # print('p')
                funcs[etype] = (self.p_message, self.p_reduce)
            # G.apply_edges(self.p_attention, etype)
            else:
                funcs[etype] = (self.n_message, self.n_reduce)

        # print("!!!",G.nodes['vul'].data['h'].shape)
        G.multi_update_all(funcs, 'stack')
        for nodetype in G.ntypes:
            if nodetype == entype:
                G.apply_nodes(self.tar_rel_aggregate,ntype=nodetype)
            else:
                G.apply_nodes(lambda nodes: {'h': torch.sum(F.softmax(nodes.data['ma'],dim=1)*nodes.data['h'],dim=1)},ntype=nodetype)
            G.nodes[nodetype].data['h'] = torch.hstack([G.nodes[nodetype].data['ht'], G.nodes[nodetype].data['h']])
        # print("???",G.nodes['vul'].data['h'].shape)
        # G.multi_update_all(funcs,'stack')
        # for nodetype in G.ntypes:
        #     # print("nodetype:", [nodetype,entype], nodetype==entype)
        #     # if nodetype == entype:
        #     G.apply_nodes(lambda nodes: {'h': torch.sum(nodes.data['h'], dim=1)}, ntype=nodetype)
        #     # else:
            #     G.apply_nodes(lambda nodes: {'h': nodes.data['x']}, ntype=nodetype)

        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HGATLayer(nn.Module):
    def __init__(self,  in_dim, out_dim, etypes,ntypes,entype='vul'):
        super(HGATLayer, self).__init__()
        # self.g = g

        self.eweight = nn.ModuleDict({
            etype: nn.Linear(in_dim, out_dim, bias=False) for etype in etypes
        })
        self.nweight = nn.ModuleDict({
            name: nn.Linear(in_dim, out_dim) for name in ntypes
        })
        # self.intype_attn = nn.Linear(2*out_dim, 1, bias = False)
        # self.attn_fcs = nn.ModuleDict({etype: nn.Linear(2 * out_dim, 1, bias=False) for etype in etypes})
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for etype, fc in self.eweight.items():
            nn.init.xavier_normal_(fc.weight, gain=gain)
        for ntype, fc in self.nweight.items():
            nn.init.xavier_normal_(fc.weight, gain=gain)
        # for etype, fc in self.attn_fc.items():
        #     nn.init.xavier_normal_(fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.intype_attn.weight, gain=gain)
        # nn.init.xavier_normal_(self.type_attn.weight, gain=gain)

    def edge_attention(self, edges):
        s = torch.cosine_similarity(edges.src['hr'], edges.dst['ht']).reshape(-1,1)
        return {'s': s}

    def message(self, edges):
        return {'a': edges.data['s'], 'h': edges.src['hr']}


    def reduce(self,nodes,rt = None):

        # alpha = nodes.mailbox['a'].squeeze(2) / torch.sum(nodes.mailbox['a'], dim=1)
        # alpha = alpha.unsqueeze(2)
        # alpha = alpha.reshape(-1,-1,1)
        # alpha = F.softmax(nodes.mailbox['a'], dim=1)
        # m_alpha = torch.mean(alpha, dim=1)
        # print("Alpha:", alpha.shape, "neighbor:", nodes.mailbox['h'].shape)
        return {'h': torch.sum(nodes.mailbox['a'] * nodes.mailbox['h'], dim = 1),
                'ma':torch.mean(nodes.mailbox['a'], dim=1)}


    def forward(self, G, feat_dict, entype='vul'):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # phi(j,r) = W_r * h_j
            # print(etype)
            G.nodes[srctype].data['hr'] = self.eweight[etype](feat_dict[srctype])

            G.nodes[dsttype].data['ht'] = self.nweight[dsttype](feat_dict[dsttype])
            # print(dsttype, G.nodes[dsttype].data['Wh'].shape)
            G.apply_edges(self.edge_attention, etype=etype)
            # print("edge attention:", G.edges[etype].data['e'].shape)

            funcs[etype] = (self.message, self.reduce)



        # print("!!!",G.nodes['vul'].data['h'].shape)
        G.multi_update_all(funcs, 'stack')
        for nodetype in G.ntypes:

            G.apply_nodes(lambda nodes: {'h': torch.sum(F.softmax(nodes.data['ma'],dim=1)*nodes.data['h'],dim=1)},ntype=nodetype)
            # G.nodes[nodetype].data['h'] = torch.hstack([G.nodes[nodetype].data['ht'], G.nodes[nodetype].data['h']])
        # print("???",G.nodes['vul'].data['h'].shape)
        # G.multi_update_all(funcs,'stack')
        # for nodetype in G.ntypes:
        #     # print("nodetype:", [nodetype,entype], nodetype==entype)
        #     # if nodetype == entype:
        #     G.apply_nodes(lambda nodes: {'h': torch.sum(nodes.data['h'], dim=1)}, ntype=nodetype)
        #     # else:
            #     G.apply_nodes(lambda nodes: {'h': nodes.data['x']}, ntype=nodetype)
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}
class HeteroRGNN(nn.Module):
    def __init__(self, G, kg, agg, mask, gnn_type, use_gpu, in_size, hidden_size, out_size):
        super(HeteroRGNN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        # embed_dict = {ntype: nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
        #               for ntype in G.ntypes}
        # for key, embed in embed_dict.items():
        #     nn.init.xavier_uniform_(embed)

        embed_dict = {ntype:nn.Parameter(G.nodes[ntype].data['x']) for ntype in G.ntypes}
        if mask:
            self.agg_layer = AggregateLayer(G, kg,use_gpu)
        else:
            self.agg_layer = FirstLayer(in_size, hidden_size, G.etypes)
        #agg = True
        if agg:
            self.agg_layer(G, embed_dict)
            embed_dict = {ntype:nn.Parameter(G.nodes[ntype].data['h']) for ntype in G.ntypes}
        # print(embed_dict)
        self.embed = nn.ParameterDict(embed_dict)

        # create layers
        self.layers = nn.ModuleList()
        if gnn_type == 'pgat':
            self.layers.append(PHGATLayer(in_size, int(hidden_size / 2), G.etypes, G.ntypes))
            self.layers.append(HGnnLayer(hidden_size, int(out_size / 2), G.etypes, G.ntypes))
        if gnn_type == 'gat':
            self.layers.append(HGATLayer(in_size, hidden_size, G.etypes, G.ntypes))
            self.layers.append(HGATLayer(hidden_size, out_size, G.etypes, G.ntypes))
        if gnn_type == 'gcn':
            self.layers.append(HeteroRGCNLayer( in_size, hidden_size, G.etypes))
            self.layers.append(HeteroRGCNLayer(hidden_size, out_size, G.etypes))
        if gnn_type == 'gsage':
            self.layers.append(HGnnLayer(in_size, int(hidden_size/2), G.etypes, G.ntypes))
            self.layers.append(HGnnLayer(hidden_size, int(out_size / 2), G.etypes, G.ntypes))

        # self.layers.append(HeteroRGCNLayer(in_size, hidden_size, G.etypes))
        # self.layers.append(HGnnLayer(in_size, int(hidden_size/2), G.etypes, G.ntypes))
        # self.layers.append(PHGATLayer(in_size, int(hidden_size/2), G.etypes, G.ntypes))
        # self.layers.append(HeteroRGCNLayer( hidden_size, hidden_size, G.etypes))
        # self.layers.append(HeteroRGCNLayer(hidden_size, out_size, G.etypes))
        # self.layers.append(HGATLayer(in_size, hidden_size, G.etypes, G.ntypes))
        # self.layers.append(HGATLayer(hidden_size,out_size, G.etypes, G.ntypes))
        # self.layers.append(HGnnLayer(hidden_size, int(out_size/2), G.etypes, G.ntypes))


    def forward(self, G):
        # embed_dict = {ntype:nn.Parameter(G.nodes[ntype].data['x']) for ntype in G.ntypes}
        # # print(embed_dict)
        # self.embed = nn.ParameterDict(embed_dict)
        h_dict = self.embed
        for l,layer in enumerate(self.layers):
            # print('&&&', l, h_dict['vul'].shape)
            h_dict = layer(G, h_dict)

            h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}

        # get paper logits
        return h_dict['vul']



class HGAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim):
        super(HGAT, self).__init__()
        self.g = g
        embed_dict = {ntype: nn.Parameter(g.nodes[ntype].data['x']) for ntype in g.ntypes}
        self.embed = nn.ParameterDict(embed_dict)
        self.layer1 = HGATLayer(g, in_dim, hidden_dim)
        self.layer2 = HGATLayer(g, hidden_dim, out_dim)

    def forward(self, g):
        h_dict = self.layer1(g, self.embed)
        # h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = {k: F.elu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(g, h_dict)
        return h_dict['vul']
