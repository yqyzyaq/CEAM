import argparse, pickle, time
import torch as th
from utils.preprocess import *
from utils.hetero_gnn import HeteroRGNN
from utils.gnn_module import GAT, GCN, GNN
from utils.align import *
from utils.mask import *
from torch.optim.lr_scheduler import StepLR

import json
from tqdm import tqdm
from sklearn.metrics import classification_report, average_precision_score

# import dgl.function as fn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

def write_node_embeddings( h, lid,rid, mode,labels):
    h = h.detach().numpy()
    print("Export node embeddings...")
    if mode == 'concat':
        h = np.hstack((h[lid], h[rid]))
    elif mode == 'diff':
        h = h[lid] - h[rid]
    np.savetxt(os.path.join(args.dp, 'feature.txt'), h)
    np.savetxt(os.path.join(args.dp, 'label.txt'), labels)

def validate(align_model, g, kg, val_data, epoch, use_gpu):
    with torch.no_grad():
        val_batch = val_data
        if not len(val_batch): return  False
        lid, rid = get_embedding_pair(kg, val_batch)
        # print('left:', lid, 'right:', rid)
        # logits = align_model(initial_emb(g.ndata['x']), lid, rid)
        if use_gpu:
            logits = align_model(g, lid, rid).cpu()
        else:
            logits = align_model(g, lid, rid)
        # output = align_model(g.ndata['x'], val_batch[:, 0], val_batch[:, 1])
        # print("batch:{}, output shape:{}".format(b,output.shape))

        # loss = loss_fcn(logits,train_data[:,2])

        # prediction =  output.argmax(dim=1)
        labels = torch.unsqueeze(torch.tensor(val_batch[:, 2]).float(), 1).numpy()
        # labels = torch.tensor(val_batch[:, 2]).float().long()
        # print("labels:{}\n, output:{}\n,prediction:{}".format(labels.numpy().flatten(), output.numpy().flatten(),prediction))
        best_f1, best_roc, precision, recall = 0, 0, 0, 0
        best_pr = 0
        best_th = 5
        prediction = []
        # Select an optimal threshold
        for i in range(3, 10):
            for j in range(9):
                crr_th = 0.1 * i + 0.01 * j
                # prediction = torch.where(output > 0.1, torch.ones_like(output), torch.zeros_like(output))
                pred = torch.sigmoid(logits).numpy() > crr_th
                # f1 = f1_score(batc
                # h[:, 2].numpy(), torch.sigmoid(score).detach().cpu().numpy() > 0.1 * i)
                # f1 = f1_score(labels, pred)
                roc_score = roc_auc_score(labels, pred)
                prauc = average_precision_score(labels, pred, average='weighted')
                # print("ROC_AUC score", i, roc_score)
                #best_f1 = max(best_f1, f1)
                # precision = precision_score(labels, pred)
                # recall = recall_score(labels, pred)
                if roc_score > best_roc:
                    # print("Best roc threshold:",i)
                    best_roc = roc_score
                    best_pr = prauc
                    # print("Best Threshold",i*0.1, pred.flatten())
                    # tp = np.sum(pred[pred == labels])
                    # precision = tp/np.sum(pred); recall = tp/np.sum(labels)
                    precision = precision_score(labels, pred);
                    recall = recall_score(labels, pred)
                    best_f1 = f1_score(labels, pred)
                    # print('F1:{:.4f}, Precision:{:.4f}, Recall:{:.4f}'.format(f1_score(labels, pred),precision, recall))

                    best_th = crr_th

        if precision and recall:
            print('Validation --Epoch:{} Best ROCAUC F1:{:.4f}, Precision:{:.4f}, Recall:{:.4f}, PRAUC:{:.4f}, Threshold:{}'.format(epoch, best_f1,
                                                                                                 precision, recall,best_pr, best_th))
        return best_roc, best_th



def test(align_model, g, kg, test_data, threshold, use_gpu):
    test_batch = test_data
    lid, rid = get_embedding_pair(kg, test_batch)
    # lid, rid = get_graph_id(kg, test_batch)
    # print('left:', lid, 'right:', rid)
    if use_gpu:
        logits = align_model(g, lid, rid).cpu()
    else:
        logits = align_model(g, lid, rid)
    labels = torch.unsqueeze(torch.tensor(test_batch[:, 2]).float(), 1).numpy()

    # cal_hit1(lid, rid, g, kg, labels, align_model)
    # th = 0.55
    pred = torch.sigmoid(logits).numpy() > threshold
    print("Threshold from validation,", threshold)
    f1, micro_f1, macro_f1 = f1_score(labels, pred), f1_score(labels, pred, average='micro'), \
                             f1_score(labels, pred, average='macro')

    acc = accuracy_score(labels, pred)
    # print("Test Logits (first ten) -- ", torch.flatten(logits)[:10])
    precision = precision_score(labels, pred)
    recall = recall_score(labels, pred)
    # tp = np.sum(pred[pred==labels])
    # # print(tp,np.sum(pred),np.sum(labels))
    # avg_pre += (tp/np.sum(pred))
    # avg_rec += (tp/np.sum(labels))

    print(
        'Test --  Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f},Micro F1:{:.4f}, Macro-F1:{:.4f}, Accuracy:{:.4f}'.format(
            precision, recall, f1, micro_f1, macro_f1, acc))
    print("PRAUC:",
    average_precision_score(labels, pred, average='weighted'),"ROCAUC:",roc_auc_score(labels, pred))
    print(classification_report(labels, pred))
    print(pred.flatten().astype(int), labels.flatten())
    pre_at_best_rec, rec, th1 = 0,0,0
    for i in range(30):
        th_ = 0.7- 0.01 *i
        pred = torch.sigmoid(logits).numpy() > th_
        precision = precision_score(labels, pred)
        recall = recall_score(labels, pred)
        pr = average_precision_score(labels, pred, average='weighted')
        print('Threshold:{:.4f}, Precision:{:.4f}, Recall:{:.4f}, PRAUC:{:.4f}'.format(th_, precision, recall,pr))
        if recall >=0.9:
            if precision > pre_at_best_rec:
                pre_at_best_rec, rec, th1 = precision, recall, th_
    if rec:
        print("Pre@Rec=0.9:", pre_at_best_rec, rec, 'th:',th1)




def get_graph(hetero, kg, with_mask, datapath):
    print(" The graph is hetero: ", hetero)

    # build homogeneous graph in DGL from KG
    if not hetero:
        g = kg.buildGraph()
        g_mirror = []
        print('Build Graph with {} nodes and {}-dim features...'.format(g.num_nodes(), kg.features.shape[1]))
        if with_mask: g_mirror = get_edge_mirror(g, kg)
        return g, g_mirror
    # build heterograph in DGL from KG
    else:
        g = kg.buildHeteroGraph()
        print("total nodes",len(kg.entity_list))
        num_of_nodes, num_of_edges = 0,0
        for ntype in g.ntypes:
            num_of_nodes += g.num_nodes(ntype)
        for etype in g.etypes:
            num_of_edges += g.num_edges(etype)
        print('Build Graph with {} nodes and {}-dim features...'.format(g.num_nodes(), kg.features.shape[1]))

        # print('Num of Nodes:', num_of_nodes)
        # print('Num of Edges:', num_of_edges)

        attribute_degree(g, kg, 'vul', datapath)
        # g_candidates = []
        # if with_mask:
        g_candidates = get_candidates(g,kg,'vul',datapath)
        return g, g_candidates

def get_gnn(kg, g, g_mirror, gnn_type, hdim,use_gpu):
    gnn = None
    if len(g.ntypes) == 1:
        feature_matrix = g.ndata['x']
        gnn =  GNN(g, kg, g_mirror,gnn_type, in_dim=feature_matrix.shape[1], hidden_dim = hdim, out_dim=hdim,
                      multihead = args.mh, num_heads=args.nh, mask = args.mask, learnable=args.learnable,num_layers = args.nl)

    elif len(g.ntypes) > 1:
        print("Initialize GNN for Heterogeneous Graph!")
        feature_matrix = g.nodes['vul'].data['x']
        gnn = HeteroRGNN(g, kg,args.agg, args.mask, gnn_type, use_gpu,in_size=feature_matrix.shape[1], hidden_size = hdim, out_size=hdim)
    return gnn

def get_embedding_pair( kg, batch):
    if not args.heterogeneous:
        lid, rid = get_graph_id(kg, batch)
        return ( lid, rid)
    else:
        # ids on homo graph
        lid_, rid_ = get_graph_id(kg, batch)

        lid, rid = kg.id_in_type[lid_], kg.id_in_type[rid_]
        return (lid, rid)


def get_subgraph(g, lid, rid, entype):
    # print('vul id on graph 1',lid)
    # print('vul id on graph 2',rid)
    en_ids = np.concatenate([lid, rid])
    subg_dict = {entype: torch.tensor(en_ids)}
    # print(subg_dict)
    for srctype, etype, dsttype in g.canonical_etypes:
        if 'of' in etype:
            # subg_dict[srctype] = []
            for eid in en_ids:
                attr_nodes = g.predecessors(eid, etype)
                if srctype not in subg_dict.keys(): subg_dict[srctype]= attr_nodes
                else:
                    # print(subg_dict[srctype],g.predecessors(eid, etype))
                    subg_dict[srctype] = torch.cat([subg_dict[srctype],attr_nodes])
                for aid in attr_nodes:
                    subg_dict[entype] = torch.cat([subg_dict[srctype], g.successors(aid, etype = srctype+'_of')])
            # print('&&:', srctype, subg_dict[srctype])
    subg = g.subgraph(subg_dict)
    # print(subg.nodes[entype].data[dgl.NID])
    return subg






def main(args):
    if args.gpu >= 0  and torch.cuda.is_available():
        use_gpu = True
        torch.cuda.set_device(args.gpu)
    else:
        use_gpu = False

    print("Use GPU?",use_gpu)
    kg = KG(args.dp,args.tp, args.cp, args.text_representation)
    # list of entities, relations, and types

    kg.buildKG([args.a, args.b])
    # print("Successfully build graph A:",kg.graph_a.ndata)
    with open('../check/id2idg.json','w') as f:
        json.dump(kg.id2idg, f)

    gnn_type = args.gnn
    hdim = args.hdim
    g, g_mirror = get_graph(args.heterogeneous, kg, args.mask, args.dp)
    kg.candidates = g_mirror
    # print("Candidate for vul 0 in A:",g_mirror[0])
    # print("Candidate for vul 0 in B:", g_mirror[986])
    if use_gpu:

        for ntype in g.ntypes:
            g.nodes[ntype].data['x'] = g.nodes[ntype].data['x'].cuda()

            if ntype!='vul':

                g.nodes[ntype].data['d'] = g.nodes[ntype].data['d'].cuda()
                g.nodes[ntype].data['d1'] = g.nodes[ntype].data['d1'].cuda()
                g.nodes[ntype].data['d2'] = g.nodes[ntype].data['d2'].cuda()
    GNN = get_gnn(kg, g, g_mirror, gnn_type, hdim, use_gpu)


    dur = []
    # initial_emb = EmbeddingLayer(kg, args.de)
    print("Initiate Align Model...")
    align_model = AlignNet(g,kg,in_dim = hdim, h_dim = int(hdim/2), mode = args.mode ,gnn = GNN)
    # print(align_model)
    train_data = load_data(args.dp,'train.csv')
    test_data = load_data(args.dp, 'test.csv')
    val_data =  load_data(args.dp, 'valid.csv')
    optimizer = torch.optim.SGD(align_model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    # print(align_model.parameters())
    criteria = nn.BCEWithLogitsLoss()
    if use_gpu:
        GNN = GNN.cuda()
        align_model = align_model.cuda()
        criteria = criteria.cuda()
        # kg.candidates.cuda()


    dist_opt =  HingeDistLoss()

    # criteria = torch.nn.CrossEntropyLoss()
    batch_size = args.batch_size
    avg_pre, avg_rec = 0, 0
    MULTI_TEST = False

    '''Train Align Model'''
    print("Train Graph Embedding...")
    BEST_ROC = 0
    th = 0.5
    for epoch in range(args.n_epochs):
        # if epoch: break
        t0 = time.time()
        align_model.train()
        train_loss = 0
        # GNN.get_weight()
        for b in tqdm(range(int(len(train_data)/batch_size))):

            batch_id = np.arange(b* batch_size,(b+1)*batch_size)
            train_batch = train_data[batch_id]
            '''Balance Train Data'''
            pos_samples= np.array([sample for i,sample in enumerate(list(train_batch)) if sample[2]==1])
            pos_samples_bid = np.array([i for i,sample in enumerate(list(train_batch)) if sample[2]==1])
            if len(pos_samples):
                for i in range(int((len(train_batch)-len(pos_samples))/(args.scale*len(pos_samples)))):
                    train_batch = np.vstack([train_batch, pos_samples])
                    batch_id = np.concatenate([batch_id, pos_samples_bid])
            '''------------'''

            lid, rid = get_embedding_pair(kg, train_batch)

            logits = align_model(g, lid, rid)

            labels = torch.unsqueeze(torch.tensor(train_batch[:, 2]).float(), 1)


            if use_gpu: labels = labels.cuda()

            loss = criteria(logits, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss += loss

        dur.append(time.time() - t0)

        # lid, rid = get_graph_id(kg, train_data)
        lid, rid = get_embedding_pair(kg, train_data)
        labels = train_data[:, 2]
        # logits = align_model(initial_emb(g.ndata['x']), lid, rid)
        if use_gpu:
            logits = align_model(g, lid, rid).cpu()
        else:
            logits = align_model(g, lid, rid)
        pred = torch.sigmoid(logits).detach().numpy() > 0.5
        acc = accuracy_score(labels, pred)
        f1, micro_f1, macro_f1 = f1_score(labels, pred), f1_score(labels, pred,average = 'micro'), f1_score(labels, pred,average = 'macro')

        # print("Train lables  (first ten) -- ", labels[:10])
        # print("Train Logits before sigmoid(first ten) -- ", torch.flatten(logits).detach().numpy()[:10])
        # print("Train Logits after sigmoid(first ten) -- ", torch.flatten(torch.sigmoid(logits)).detach().numpy()[:10])
        # print(emb[lid[0]])
        # print(emb[rid[0]])
        print("Epoch {:05d} | Train Loss {:.4f} | Train Accuracy {:.4f} |F1 {:.4f} | Time(s) {:.4f}".format(
            epoch, train_loss.item(), acc,f1, np.mean(dur)))

        '''---Validation---'''

        if epoch % 2 == 0 :

            best_roc, best_th = validate(align_model, g, kg, val_data, epoch, use_gpu)
            if best_roc >= BEST_ROC:
                BEST_ROC = best_roc
                best_model, th = align_model, best_th


    '''---- Test ---'''
    align_model.eval()
    best_model_name = gnn_type + '.' + str(th) + '.pkl'
    # save best model selected by validation set
    if os.path.exists('../models/' + best_model_name):
        os.remove('../models/' + best_model_name)
    torch.save(best_model.state_dict(), '../models/' + best_model_name)
    # Test phase
    align_model.load_state_dict(torch.load('../models/' + best_model_name))
    with torch.no_grad():
        test(align_model, g, kg, test_data, th, use_gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MKGNN')
    parser.add_argument("--a", type = str, default = "tableA",help="source graph")
    parser.add_argument("--b", type = str, default = "tableB",help="target graph")

    parser.add_argument("--dp", type = str, default = "../data/sample/",help="data path")
    parser.add_argument("--tp", type = str, default = "../text/model.bin",help="path to text embedding model")
    parser.add_argument("--cp", type = str, default = "../text/corpus.txt",help="path to corpus for text embedding model")
    parser.add_argument("--mp", type = str, default = "../models", help = "path to save trained models")
    # parser.add_argument("--de", type=int, default=0, help="dimension of embedding trained by the first layer")
    parser.add_argument("-ht", "--heterogeneous", type=int, default=1,
                        help="whether distinguish relation type")
    parser.add_argument("--agg", type=int, default=1,
                        help="whether use aggregate layer")
    parser.add_argument("--gnn", type = str, default = "pgat", help="type of GNN: gcn/gat/graphsage")
    parser.add_argument("-m","--mask", type=int, default = 1, help="whether apply mask in GNN")
    parser.add_argument("--text-representation", type=str, default='ft', help="text representation: ft(fasttext), bow(bag-of-words), emb(embedding layer)")
    parser.add_argument("--mode", type=str, default='multi', help="whether use the concatenation or the difference of two node representation in the alignment model")
    parser.add_argument("--mh", type=int, default=0, help="whether use multi-head attention")
    parser.add_argument("--nh", type=int, default=2, help="number of heads in multi-head attention")
    parser.add_argument("--nl", type=int, default=4, help="number of layers in GNN")
    parser.add_argument("--scale", type=int, default=1, help="scale for balancing positive data")
    #parser.add_argument("--learnable", type=int, default=0,help="whether set learnable scaled parameter in mask")
    parser.add_argument("--concat", type=bool, default=False, help="whether concat at each hidden layer")
    #parser.add_argument("--gat", action='store_true', help="whether GCN or GAT is chosen")
    parser.add_argument("--dist-opt", type=int, default=0,
            help="[1: hinge loss, 0: binary classification]")
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("-lr","--lr", type=float, default=0.08,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
            help="batch size")
    # parser.add_argument("--num-neighbors", type=int, default=10,
    #         help="number of neighbors to be sampled")
    # parser.add_argument("--num-negatives", type=int, default=10,
    #         help="number of negative links to be sampled")
    parser.add_argument("--num-test-negatives", type=int, default=10,
            help="number of negative links to be sampled in test setting")
    parser.add_argument("--hdim", type=int, default=64,
            help="number of hidden units")
    parser.add_argument("--dump", action='store_true',
            help="dump trained models (default=False)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
            help="Weight for L2 loss")
    parser.add_argument("--model-id", type=str,
        help="Identifier of the current model")
    parser.add_argument("--pretrain_path", type=str, default="../text/mode"
                                                             "l.bin",
        help="pretrained fastText path")
    args = parser.parse_args()


    print(args)

    main(args)
