import numpy as np
import dgl
import torch
import os
import json

def mark_rel_importance(g,entype = 'vul'):
    for srctype, etype, dsttype in g.canonical_etypes:
        g.edges[etype].data['rd'] = 0


def attribute_degree(g,kg, entype,path ):
    path = path +  'attr_degree.json'
    if os.path.exists(path):
        print("Load Node Degree from: ", path)
        with open(path) as f:
            df = json.load(f)
            for ntype in g.ntypes:
                if ntype != entype:
                    g.nodes[ntype].data['d1'] = torch.FloatTensor(df[ntype]['d1']).reshape(len(df[ntype]['d1']),1)
                    g.nodes[ntype].data['d2'] = torch.FloatTensor(df[ntype]['d2']).reshape(len(df[ntype]['d2']),1)
                    g.nodes[ntype].data['d'] = torch.FloatTensor(df[ntype]['d']).reshape(len(df[ntype]['d']),1)
            # print( g.nodes[ntype].data['d1'] )
            print("Success: load node degree")
    else:
        df = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            if 'of' in etype:
                df[srctype] = {'d1':[],'d2':[],'d':[]}
                # attributes in a type
                g.nodes[srctype].data['d1'] = torch.zeros((g.number_of_nodes(srctype),1))
                g.nodes[srctype].data['d2'] = torch.zeros((g.number_of_nodes(srctype), 1))
                g.nodes[srctype].data['d'] = torch.zeros((g.number_of_nodes(srctype), 1))
                for i in range(g.number_of_nodes(srctype)):
                    succ = g.successors(i, etype)
                    # print("Etype:{}, Src Type:{}, Successors:{}".format(etype, srctype, succ))
                    g.nodes[srctype].data['d1'][i] = len([x for x in succ if x < kg.splitvulid])
                    g.nodes[srctype].data['d2'][i] = len([x for x in succ if x >= kg.splitvulid])
                    g.nodes[srctype].data['d'][i] =  g.nodes[srctype].data['d1'][i] +  g.nodes[srctype].data['d2'][i]
                # print(">>", type(g.nodes[srctype].data['d1'][0]), type(g.nodes[srctype].data['d1']))
                for k in df[srctype].keys():
                    df[srctype][k] =  [int(x) for x in g.nodes[srctype].data[k]]
        # print(g.nodes[srctype].data['d1'])

        with open(path, 'w') as f:
            json.dump(df, f, indent=4)
            print('Save node dgree at:',path)

def get_candidates(g,kg,entype, path):
    import pickle
    path = path + 'candidates.pkl'
    if os.path.exists(path):
        with open(path,'rb') as f:
            g_candidates = pickle.load(f)
            print("Success: load candidates")
    else:
        print('Generate candidates...')
        g_candidates = []
        num_en = g.number_of_nodes(entype)
        for i in range(num_en):
            if i%100 ==0: print(i, i/num_en*100,'%')
            candidates = []
            maxcount = 0
            for srctype, etype, dsttype in g.canonical_etypes:
                if srctype == entype and ('product' in etype or 'vendor' in etype): #etype.replace('has_','') in kg.critical_etypes :
                    neighbors = g.successors(i, etype)
                    reverse_etype = etype.replace('has_','') + ('_of')
                    for n in neighbors:
                        can = g.successors(n,reverse_etype)
                        for c in can:
                            if c not in candidates:
                                candidates.append(c)
                            # if candidates.count(c)>maxcount:maxcount =candidates.count(c)

            # candidates = [x for x in candidates if candidates.count(x)==maxcount]
            if i<kg.splitvulid: candidates = [x for x in candidates if x!= i and x >= kg.splitvulid]
            else:candidates = [x for x in candidates if x!= i and x < kg.splitvulid]
            candidates = list(candidates)
            g_candidates.append(candidates)

        with open(path, 'wb') as f:
            # json.dump(df, f, indent=4)
            pickle.dump(g_candidates,f)

    return g_candidates

def get_graph_id(kg, data):
    id_a, id_b = data[:, 0], data[:, 1]
    #print(kg.id2idg['a'].keys())

    idg_a = np.array([kg.id2idg['a'][x] for x in id_a])
    idg_b = np.array([kg.id2idg['b'][x] for x in id_b])

    return idg_a, idg_b


def get_neighborhood(g, dst):
    frontier = dgl.in_subgraph(g, [dst])
    return np.array(frontier.edges()[0])

def get_edge_mirror(g,kg):
    # print('Get mirrow dst for each edge...')
    neighbor = [get_neighborhood(g, i) for i in range(g.num_nodes())]
    # print(neighbor)
    g_mirror= []
    for src,dst in zip(g.edges()[0],g.edges()[1]):
        m_srcs = []
        # print(kg.entity_type[dst])
        if kg.entity_type[dst]!='id':

            g_mirror.append(m_srcs)
            continue
        # print('src:',src,'dst:',dst)
        n = list(neighbor[dst])
        # print('neighbor of dst:',n)
        n.pop(n.index(src))
        # print('exclusive neighbor of dst',n)

        if n:
            m_dsts = get_neighborhood(g, n[0])
            for node in n:
                m_dsts = list(np.intersect1d(m_dsts,get_neighborhood(g, node)))
            if dst in m_dsts:
                m_dsts.pop(m_dsts.index(dst))

            for m_dst in m_dsts:
                for j in get_neighborhood(g, m_dst):
                    if kg.entity_type[j] == kg.entity_type[src]:
                        m_srcs.append(j)
        g_mirror.append(m_srcs)
        # if m_srcs:
        #     print(src,dst,m_srcs)
        # print('mirror center:',ce)

    # print("g_mirror",g_mirror)
    return np.array(g_mirror, dtype=object)
