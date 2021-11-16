import os
import dgl
import csv
import torch as th
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import re

def load_data(path,file):
    data = []
    lines = open(os.path.join(path, file)).readlines()
    for i, line in enumerate(lines):
        if i==0: continue
        line = line.strip('\r\n').split(',')
        line = [int(x) for x in line]
        data.append(line)

    return np.array(data)

class FeatureGenerator(object):

    def __init__(self, model_path, corpus_path, text_representation = 'ft'):
        super(FeatureGenerator, self).__init__()
        self.tr = text_representation
        # FastText
        if text_representation == 'ft':
            import fasttext as ft
            print("Initialize text representation with FastText!")
            if os.path.exists(model_path):
                print('Load text embedding model from ', model_path,'...')
                self.model = ft.load_model(model_path)
                print('Text embedding model loaded!')
            else:
                print('Train text embedding model from corpus', corpus_path, '...')
                self.model = ft.train_unsupervised(corpus_path, model = 'cbow',min_count = 1 )
                self.model.save_model(model_path)
                print('Finished!')
        elif text_representation == 'index':
            print("Initialize text representation with Padded Word Index Sequence!")
            self.vectorizer = CountVectorizer(binary=True, stop_words='english', lowercase=True, max_features=3000)
        # Bag-of-words
        elif  'bow' in text_representation:
            print("Initialize text representation with Bag-of-Words!")            # Character-level
            if 'c' in text_representation:
                self.vectorizer = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 2))
            else:
                self.vectorizer = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 2))




    def generateFastText(self, string, dim, sent=True):
        if sent == True:
            return self.model.get_sentence_vector(string.replace('"', ''))
        else:
            return self.model.get_word_vector(string.replace('"', ''))

    # def generateNumFeature(self,string, dim):
    #     nums = re.findall(r'\d', string)
    #     if not nums:
    #         return self.generateEmbFeature(string,dim)
    #     elif len[nums] == 1:
    #         return nums * dim
    #     else:
    #         n = float(nums[3] + '.' + ''.join(nums[4:]))
    #         return [n]*dim
    def generatePaddedSequence(self,data):
        # vectorizer = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 2))

        x_onehot = self.vectorizer.fit_transform(data)
        # print("vocab:",vectorizer.get_feature_names())
        word2idx = {word: idx for idx, word in enumerate(self.vectorizer.get_feature_names())}
        tokenizer = self.vectorizer.build_tokenizer()
        preprocessor = self.vectorizer.build_preprocessor()
        # print("data0",data[:10])
        print(tokenizer(preprocessor(data[0])))
        x_sequences = [[word2idx[word] for word in tokenizer(preprocessor(x)) if word in word2idx] for x in data]
        # print("x1",x_sequences[:10])
        x_sequences = [th.tensor(x) for x in x_sequences]
        # print(len(x_sequences),x_sequences[0].shape)
        MAX_SEQ_LENGHT = len(max(x_sequences, key=len))
        print("MAX_SEQ_LENGHT=", MAX_SEQ_LENGHT)

        from torch.nn.utils.rnn import pad_sequence

        N_FEATURES = len(self.vectorizer.get_feature_names())
        # print("x2",x_sequences[:10])
        feature_matrix = pad_sequence(x_sequences,  batch_first=True, padding_value=N_FEATURES)
        # print("x3", feature_matrix[:10])
        print(np.array(feature_matrix).shape, N_FEATURES)
        return N_FEATURES + 1, feature_matrix

    def generateBertEmbedding(self,data):
        # Compute the max length of a text
        # text_sequence = [x.split() for x in data]
        # MAX_SEQ_LENGHT = len(max(text_sequence, key=len))
        # print( "Max Sequence Length: ", MAX_SEQ_LENGHT, max(text_sequence, key = len) )
        input_ids = []
        attention_masks = []
        with th.no_grad():
            for item in data:
                encoded_dict = self.tokenizer.encode_plus(
                    item,  # Sentence to encode.
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    # padding = 'max_length',
                    max_length=256,  # Pad & truncate all sentences.
                    pad_to_max_length=True,
                    return_attention_mask=True,  # Construct attn. masks.
                    return_tensors='pt',  # Return pytorch tensors.
                )
                # print(self.tokenizer.tokenize(item))
                # Add the encoded sentence to the list.
                input_ids.append(encoded_dict['input_ids'])

                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_dict['attention_mask'])
            input_ids = th.cat(input_ids, dim=0)
            attention_masks = th.cat(attention_masks, dim=0)
            outputs = self.model(input_ids, attention_masks)
            hidden_states = outputs[2]
            token_embeddings = th.stack(hidden_states, dim=0)
            print(token_embeddings.shape)
            token_vecs = token_embeddings[-2]
            item_embeddings = th.mean(token_vecs, dim=1)
        return item_embeddings



    def generateFeatureMatrix(self,data, dim=100):
        # data: a list where each element is the text of an entity
        feature_matrix = []

        if self.tr == 'ft':

            for i,item in enumerate(data):
               feature_matrix.append(self.generateFastText(item, dim))

        elif 'bow' in self.tr:
            self.vectorizer.fit(data)
            #print(vectorizor.vocabulary_)
            feature_matrix = self.vectorizer.transform(data).toarray().astype(np.float32)+1e-4
        elif self.tr == 'bert':
            feature_matrix = self.generateBertEmbedding(data)

        print("Finished feature initialization!")

        return np.array(feature_matrix)




class KG(object):
    def __init__(self, path, text_emb_path, corpus_path, text_representation):
        super(KG, self).__init__()
        self.tr = text_representation
        self.path = path
        self.text_emb_path = text_emb_path
        self.corpus_path = corpus_path
        self.id2idg = {'a': {}, 'b': {}}
        self.idg2id = {'a': {}, 'b': {}}
        self.entity_list, self.entity_type = [], []  # value and type of entities. index is entity id.
        self.hetero_entity_list = {} # key: entity type, value: entity id on KG
        self.id_in_type = [] # index: entity id on KG, value: id in type
        self.features = [] # features of entities. index is entity id.
        self.edge_src, self.edge_dst, self.edge_type = [], [], []
        self.vocab_size = 0
        self.hkgdict = {}
        self.critical_etypes = ['weakness_name','vendor','product_name','impact']
        self.splitid = 0

    def buildKG(self, tablenames):
        # tablenames: a list of KG filenames
        # text_encode: fastText, bag-of-words, character-level bag of words, bert

        for tablename in tablenames:
            # The beginning entity id of the second table
            if 'B' in tablename:
                self.splitid = len(self.entity_list)

            filename = os.path.join(self.path, tablename + '.csv')
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                line = next(reader)
                # {0: 'vul', 1: 'weakness_name', 2: 'intrusion_action', 3: 'cve_id', 4: 'vendor', 5: 'product_name'}
                en_type_dict = {k: v for (k, v) in enumerate(line)}
                # {1: 'weakness_name_of', 2: 'intrusion_action_of', 3: 'cve_id_of', 4: 'vendor_of', 5: 'product_name_of'}
                rel_type_dict = {k + 1: v + '_of' for (k, v) in enumerate(line[1:])}
                if 'A' in tablename:self.hkgdict = {k:[] for k in line}

                # # initialize self.entity_table
                # if not self.entity_table:
                #     self.entity_table = {k:{} for k in ent_dict.values()}; print(self.entity_table)
                for r, line in enumerate(reader):

                    eid = len(self.entity_list)  # ID of the current entity
                    id = int(line[0])


                    if 'A' in filename:
                        self.idg2id['a'][eid]=id  # key is id on kg, value is original id
                        self.id2idg['a'][id] = eid
                    elif 'B' in filename:
                        self.idg2id['b'][eid]=id
                        self.id2idg['b'][id] = eid

                    # First column: central entities
                    evalue = en_type_dict[0] + '_' + line[0]
                    # if 'B' in filename: evalue += '10000'
                    self.entity_list.append(evalue)
                    self.entity_type.append(en_type_dict[0])
                    self.hkgdict[en_type_dict[0]].append(evalue)

                    # Following columns: neighborhood entities
                    for cm, column in enumerate(line[1:]):

                        type_id = cm + 1

                        if column == '': continue
                        # each neighborhood entity is profiled by one attribute

                        split_token = ';;'
                        # if 'previous' in self.path: split_token = ','
                        for node in column.split(split_token):
                            if node == '': continue
                            if node in self.entity_list and en_type_dict[type_id]== self.entity_type[self.entity_list.index(node)]:
                                nodeid = self.entity_list.index(node)
                            else:
                                nodeid = len(self.entity_list)
                                self.entity_list.append(node)
                                self.entity_type.append(en_type_dict[type_id])

                                self.hkgdict[en_type_dict[type_id]].append(node)
                            # elif en_type_dict[type_id]!= self.entity_type(self.entity_list.index(node)):

                            # print(node,nodeid,eid)

                            edge_type, edge_src, edge_dst = rel_type_dict[type_id], nodeid, eid
                            self.edge_src.append(edge_src)
                            self.edge_dst.append(edge_dst)
                            self.edge_type.append(edge_type)

        featureGenerator = FeatureGenerator(self.text_emb_path, self.corpus_path, self.tr)
        self.edge_src = np.array(self.edge_src)
        self.edge_dst = np.array(self.edge_dst)
        self.edge_type = np.array(self.edge_type)

        if self.tr == 'index':
            self.vocab_size, self.features = featureGenerator.generatePaddedSequence(self.entity_list)
        else:
            self.features = featureGenerator.generateFeatureMatrix(self.entity_list)
        # If use Embedding layer
        # self.vocab_size, self.features = feat.generatePaddedSequence(self.entity_list)
        # print(type(self.features.shape))


    def buildHeteroGraph(self):
        print("The split entity is in type:",self.entity_type[self.splitid])
        self.id_in_type = np.arange(len(self.entity_list)) # index: id on kg, value: id in type
        #  change node id for each entity type: start from 0
        edge_types = np.unique(self.edge_type)
        # keys = [(edge_type.replace('_of',''), edge_type, 'vulnerability') for edge_type in edge_types]
        kgdict = {}
        entity_types = np.unique(self.entity_type)
        #print(entity_types)
        vendor_ids = []
        for t in entity_types:
            entity_ids = [i for i, x in enumerate(self.entity_type) if x == t]
            self.hetero_entity_list[t] = entity_ids
            self.id_in_type[entity_ids] = np.arange(len(entity_ids))

        for i, t in enumerate(edge_types):
            edge_ids = [i for i, x in enumerate(self.edge_type) if x == t]

            kgdict[(t.replace('_of', ''), t, 'vul')] = (self.id_in_type[self.edge_src[edge_ids]], self.id_in_type[self.edge_dst[edge_ids]])
            kgdict[('vul', 'has_' + t.replace('_of', ''), t.replace('_of', ''))] = (
            self.id_in_type[self.edge_dst[edge_ids]], self.id_in_type[self.edge_src[edge_ids]])

        # print(kgdict)
        g = dgl.heterograph(kgdict)
        # print(g.number_of_nodes('v2_score'))
        for t in entity_types:
            # print(t)
            entity_ids = [i for i, x in enumerate(self.entity_type) if x == t]

            # if t == 'vendor':print("vendor in g: ",g.nodes[t].data)
            # print("^^", t,len(entity_ids),g.number_of_nodes(t))
            g.nodes[t].data['x'] = th.tensor(self.features[entity_ids])

        self.splitvulid= self.id_in_type[self.splitid]
        print("split vulnerability",self.splitvulid)
        return g

    def buildGraph(self):
        #print(np.hstack([self.edge_src, self.edge_dst]).shape)
        g = dgl.DGLGraph((th.tensor(np.hstack([self.edge_src, self.edge_dst])), th.tensor(np.hstack([self.edge_dst, self.edge_src]))))
        g.ndata['x'] = th.tensor(self.features)
        # print(type(g), g.ndata['x'].shape,g.ntypes)
        return g
