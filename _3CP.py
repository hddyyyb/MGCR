from __future__ import absolute_import, print_function, division, unicode_literals

import sklearn
from sklearn.metrics import ndcg_score
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import tensorflow as tf
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from tensorflow.python.ops import control_flow_util
import random
from time import *

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

print(tf.__version__)
print(tf.executing_eagerly())

from collections import defaultdict
from collections import Counter  

EmbFinDim = 30  
lowDim = 20  


EpochsCPTrain = 200  

DimM = 30
factorDimStu = 30
factorDimCou = 30
GATHiddenDimCou = 30
GATHiddenDimStu = 30
GATHiddenDimSC = 30
Mnum = 20  
padIndices = 0
SampleNegative = 1  
recall_Top_k = 50


os.environ['CUDA_VISIBLE_DEVICES'] ="0"

def adj_to_bias(adj, sizes, nhood=1):
    
    
    nb_graphs = adj.shape[0]  
    
    
    mt = np.empty(adj.shape)

    for g in range(nb_graphs):  
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):  
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0  
    return -1e9 * (1.0 - mt)



def sample_mask(List, PosDict, l):  
    """Create mask."""
    mask = np.zeros(l)  
    for item in List:  
        mask[PosDict[item]] = 1  
    
    return np.array(mask, dtype=np.bool_)  


def load_data(dataset_str):  
    
    
    CCgraph = defaultdict(list)
    CCStrugraph = defaultdict(list)

    
    SCgraph = defaultdict(list)
    SCgraphValScore = defaultdict(defaultdict)
    SCgraphValSeme = defaultdict(defaultdict)
    StuIdList = []
    CouIdList = []
    CouCodeList = []
    TeaList = []
    DorList = []
    StuIdPosDict = defaultdict(int)
    CouIdPosDict = defaultdict(int)
    CouCodePosDict = defaultdict(int)
    TeaPosDict = defaultdict(int)
    DorPosDict = defaultdict(int)

    
    with open(path + dataset + '\\' + "_4StuId", 'r', encoding="utf-8") as fSId:
        i = 0
        for iSId in fSId:  
            StuIdList.append(iSId.strip("\n"))
            StuIdPosDict[iSId.strip("\n")] = i
            i = i+1
        fSId.close()
    with open(path + dataset + '\\' + "_3CouId", 'r', encoding="utf-8") as fCId:
        i = 0
        for iCId in fCId:
            CouIdList.append(iCId.strip("\n"))
            CouIdPosDict[iCId.strip("\n")] = i
            i = i + 1
        fCId.close()
    with open(path + dataset + '\\' + "_5CouCode", 'r', encoding="utf-8") as fCCode:
        i = 0
        for iCC in fCCode:
            CouCodeList.append(iCC.strip("\n"))
            CouCodePosDict[iCC.strip("\n")] = i
            i = i + 1
        fCCode.close()
    with open(path + dataset + '\\' + "_6ZTeaList.txt", 'r', encoding="GBK") as fT:
        i = 0
        for iT in fT:
            TeaList.append(iT.strip("\n"))
            TeaPosDict[iT.strip("\n")] = i
            i = i + 1
        fCCode.close()
    with open(path + dataset + '\\' + "_8ZDorList", 'r', encoding="utf-8") as fD:
        i = 0
        for iD in fD:
            DorList.append(iD.strip("\n"))
            DorPosDict[iD.strip("\n")] = i
            i = i + 1
        fD.close()
    
    StuNum = len(StuIdList)
    CouIdNum = len(CouIdList)
    CouCodeNum = len(CouCodeList)
    TeaNum = len(TeaList)
    DorNum = len(DorList)

    
    Stu_CouScoreM = np.array([[-1] * CouIdNum for _ in range(StuNum)])  
    Stu_CouSemeM = np.array([[-1] * CouIdNum for _ in range(StuNum)])  
    Cou_TeaM = np.array([[0] * TeaNum for _ in range(CouIdNum)])  
    Stu_DorM = np.array([[0] * DorNum for _ in range(StuNum)])  
    CouId_CodeM = np.array([[0] * CouCodeNum for _ in range(CouIdNum)])  

    
    with open(path + dataset + '\\' + "_BPrerequisiteCourses1-1", 'r', encoding="utf-8") as fRelCC:
        for cc in fRelCC:  
            course, precourse = cc.strip("\n").split(' ')
            CCgraph[course].append(precourse)
        fRelCC.close()
    
    GCCgraph = nx.from_dict_of_lists(CCgraph, create_using=nx.DiGraph)  
    degree_sequencesort = sorted((d for n, d in GCCgraph.degree()), reverse=True)
    degree_sequence = GCCgraph.degree()
    d_c_dict = dict()
    colori=20
    for degree in degree_sequencesort:
        if degree not in d_c_dict:
            d_c_dict[degree]=colori
            colori=colori-1
    node_color = []
    for n,d in degree_sequence:
        node_color.append(d_c_dict[d])

    edge_colorlist=['grey']*len(GCCgraph.edges)
    Tnum = 0
    e1i,e2i,e3i=0,0,0

    class getOutOfLoop(Exception):
        pass
    try:
        for edge1 in GCCgraph.edges:
            e2i=0
            for edge2 in GCCgraph.edges:
                e3i=0
                for edge3 in GCCgraph.edges:
                    if edge1[0]==edge3[0] and edge2[1]==edge3[1] and edge1[1]==edge2[0]:
                        
                        edge_colorlist[e1i] = "#00B2EE"
                        edge_colorlist[e2i] = "#00B2EE"
                        edge_colorlist[e3i] = "#00B2EE"
                        Tnum=Tnum+1
                        
                    e3i=e3i+1
                e2i=e2i+1
            e1i=e1i+1
    except getOutOfLoop:
        pass

    pos = nx.spring_layout(GCCgraph,k=0.15,iterations=20)
    
    options = {
        "node_color": node_color,
        "edge_color": edge_colorlist,
        "width": 0.5,
        "with_labels": False,
        "node_size": 10,
        "arrows": True,
        "cmap": plt.cm.Blues,
        "node_shape": 's',
        "arrowstyle": '->',
        "arrowsize" : 5,
        "vmin":1,
        "vmax": 5,
    }
    nx.draw_networkx(GCCgraph, pos, **options)
    plt.axis('off')
    
    plt.savefig(os.path.join(path + dataset + '\\' + "_BPrerequisiteCourses1-1"+ '.pdf'), bbox_inches='tight')
    
    plt.savefig("C:\\Users\\...\\Desktop\\_BPrerequisiteCourses1-1.pdf", bbox_inches='tight')
    plt.close()
    nodelistCC = []  
    for node_i in GCCgraph.nodes():
        nodelistCC.append(node_i)
    adjCC = nx.adjacency_matrix(GCCgraph, nodelist = CouCodeList)
    
    weightlistS = []
    with open(path + dataset + '\\' + "_GStruSimNei", 'r', encoding="utf-8") as fRelCCStru:
        for ccS in fRelCCStru:
            c_c, sS = ccS.strip("\n").split(':')
            c1, c2 = c_c.split(' ')
            CCStrugraph[c1].append(c2)
            weightlistS.append((c1, c2, float(sS)))
        fRelCCStru.close()
    
    GCCStrugraph = nx.from_dict_of_lists(CCStrugraph)
    pos = nx.spring_layout(GCCgraph)
    nx.draw_networkx(GCCgraph, pos)
    plt.savefig(os.path.join(path + dataset + '\\' + "_GStruSimNei"+ '.png'))
    
    nodelistCCS = []
    for node_i in GCCStrugraph.nodes():
        nodelistCCS.append(node_i)
    
    GCCStrugraph.add_weighted_edges_from(weightlistS, "StruSimilarity")
    
    
    adjCCS = nx.adjacency_matrix(GCCStrugraph, nodelist = CouCodeList, weight= "StruSimilarity")  
    
    
    with open(path+dataset+'\\'+"_0StuCouRel", 'r', encoding="utf-8") as fRelSC:
        for SCSS in fRelSC:  
            student, CV = SCSS.strip("\n").split('-')
            courseId, VSS = CV.split(':')
            score, semester = VSS.split(',') 
            
            Stu_CouScoreM[StuIdPosDict[student]][CouIdPosDict[courseId]] = score
            Stu_CouSemeM[StuIdPosDict[student]][CouIdPosDict[courseId]] = semester
            SCgraph[student].append(courseId)
            SCgraphValScore[student][courseId] = score
            SCgraphValSeme[student][courseId] = semester
        fRelSC.close()
    
    with open(path+dataset+'\\'+"_6CouIdTeaRel.txt", 'r', encoding="GBK") as fRelCT:
        for iCT in fRelCT:  
            iCouId, iTea = iCT.strip("\n").split(' ', 1)
            Cou_TeaM[CouIdPosDict[iCouId]][TeaPosDict[iTea]] = 1
        fRelCT.close()
    CouTeaCouM = tf.matmul(Cou_TeaM, Cou_TeaM.transpose())
    
    with open(path+dataset+'\\'+"_7CouIdCodeRel", 'r', encoding="utf-8") as fRelCIC:
        for iCic in fRelCIC:  
            iCouId, iCouCode = iCic.strip("\n").split(' ')
            CouId_CodeM[CouIdPosDict[iCouId]][CouCodePosDict[iCouCode]] = 1
        fRelCT.close()
    CIdCodeIdM = tf.matmul(CouId_CodeM, CouId_CodeM.transpose())
    
    with open(path+dataset+'\\'+"_8StuDorRel", 'r', encoding="utf-8") as fRelSD:
        for iSD in fRelSD:  
            iStu, iDor = iSD.strip("\n").split(' ')
            Stu_DorM[StuIdPosDict[iStu]][DorPosDict[iDor]] = 1
        fRelSD.close()
    StuDorStuM = tf.matmul(Stu_DorM, Stu_DorM.transpose())

    
    SList_train = []
    with open(path + dataset + '\\' + "_HList_train", 'r', encoding="utf-8") as SL:
        for idx in SL:  
            SList_train.append(idx.strip("\n"))
        SL.close()
    SList_test = []
    for temp in StuIdList:
        if temp not in SList_train:
            SList_test.append(temp)
    S_mask_train = sample_mask(SList_train, StuIdPosDict, len(StuIdList))  
    S_mask_test = sample_mask(SList_test, StuIdPosDict, len(StuIdList))  

    
    
    FeatureStuM = np.loadtxt(path+dataset+'\\'+'_1FeatureStu', dtype=np.float32, delimiter=' ')
    FeatureStuSM = sp.csr_matrix(FeatureStuM)
    FeatureCouM = np.loadtxt(path+dataset+'\\'+'_2FeatureCou', dtype=np.float32, delimiter=' ')
    FeatureCouSM = sp.csr_matrix(FeatureCouM)
    EmbCouName = np.loadtxt(path+dataset+'\\'+'_9CouNameEmb', dtype=np.float32, delimiter=' ')
    EmbCouNameM = EmbCouName.reshape((EmbCouName.shape[0], -1, lowDim))
    EmbStuMajor = np.loadtxt(path+dataset+'\\'+'_AStuMajorEmb', dtype=np.float32, delimiter=' ')
    EmbStuMajorM = EmbStuMajor.reshape((EmbStuMajor.shape[0], -1, lowDim))  
    

    
    
    StuCouScore_train = (Stu_CouScoreM.transpose() * S_mask_train).transpose()
    StuCouScore_test = (Stu_CouScoreM.transpose() * S_mask_test).transpose()
    StuCouSeme_train = (Stu_CouSemeM.transpose() * S_mask_train).transpose()
    StuCouSeme_test = (Stu_CouSemeM.transpose() * S_mask_test).transpose()

    print('------relation matrix------')
    print(f'adjCC.shape: {adjCC.shape}')
    print(f'adjCCS.shape: {adjCCS.shape}')
    print(f'Stu_CouScoreM.shape: {Stu_CouScoreM.shape}')
    print(f'Stu_CouSemeM.shape: {Stu_CouSemeM.shape}')
    print(f'Cou_TeaM.shape: {Cou_TeaM.shape}')
    print(f'CouTeaCouM.shape: {CouTeaCouM.shape}')
    print(f'CouId_CodeM.shape: {CouId_CodeM.shape}')
    print(f'CIdCodeIdM.shape: {CIdCodeIdM.shape}')
    print(f'Stu_DorM.shape: {Stu_DorM.shape}')
    print(f'StuDorStuM.shape: {StuDorStuM.shape}')
    print('------Feather&Emb------')
    print(f'FeatureStuSM.shape: {FeatureStuSM.shape}')
    print(f'FeatureCouSM.shape: {FeatureCouSM.shape}')
    print(f'EmbCouNameM.shape: {EmbCouNameM.shape}')
    print(f'EmbStuMajorM.shape: {EmbStuMajorM.shape}')
    print('------after mask------')
    print(f'StuCouTrainScore.shape: {StuCouScore_train.shape}')  
    print(f'StuCouTestScore.shape: {StuCouScore_test.shape}')
    print(f'StuCouTrainSeme.shape: {StuCouSeme_train.shape}')
    print(f'StuCouTestSeme.shape: {StuCouSeme_test.shape}')

    return StuIdList, StuIdPosDict, CouIdList, CouIdPosDict, CouCodeList, CouCodePosDict, TeaList, TeaPosDict, DorList, DorPosDict, adjCC, adjCCS, Stu_CouScoreM, Stu_CouSemeM, Cou_TeaM, CouTeaCouM, CouId_CodeM, CIdCodeIdM, Stu_DorM, \
           StuDorStuM, FeatureStuSM, FeatureCouSM, EmbCouNameM, EmbStuMajorM, SList_train, SList_test, S_mask_train, S_mask_test, StuCouScore_train, StuCouScore_test, StuCouSeme_train, StuCouSeme_test


class RawToFactor(tf.keras.layers.Layer):
    def __init__(self, emb_dim=20, factorDim=20, in_drop=0.0, activation=tf.nn.elu, residual=False):

        
        super(RawToFactor, self).__init__()
        
        self.mask = tf.keras.layers.Masking(mask_value=-1)
        self.gru = tf.keras.layers.GRU(emb_dim, kernel_regularizer='l2')
        self.activation = activation
        self.residual = residual  
        self.in_dropout = tf.keras.layers.Dropout(in_drop)  
        self.connect_ConEmb = tf.keras.layers.Concatenate()
        self.W_to_factor = tf.keras.layers.Dense(factorDim, activation=activation, kernel_regularizer='l2')

    def __call__(self, EmbRaw, Fea, training):
        Emb0 = self.mask(EmbRaw[0])  
        Emb = self.gru(Emb0)  
        Emb1 = Emb[np.newaxis]
        seq = self.in_dropout(Fea, training=training)  
        InputEmb = self.connect_ConEmb([Emb1, seq])  
        factor = self.activation(self.W_to_factor(InputEmb))  

        return factor


class w_atten(tf.keras.layers.Layer):
    def __init__(self, axes=None):
        super(w_atten, self).__init__()
        self.supports_masking = True
        self.get_w_att = tf.keras.layers.Dot(axes=axes)

    def call(self, inputs):
        return self.get_w_att(inputs)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return inputs[0]._keras_mask  



class attn_head(tf.keras.layers.Layer):
    
    def __init__(self, hidden_dim, nb_nodes=None, in_drop=0.0, coef_drop=0.0, activation=tf.nn.elu, residual=False):
        super(attn_head, self).__init__()
        self.activation = activation
        self.residual = residual
        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)
        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim, 1, use_bias=False, kernel_regularizer='l2')  
        self.conv_f1 = tf.keras.layers.Conv1D(1, 1, kernel_regularizer='l2')
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1, kernel_regularizer='l2')
        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1, kernel_regularizer='l2')
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))  

    def __call__(self, seq, bias_mat, training):
        
        seq = self.in_dropout(seq, training=training)  
        seq_fts = self.conv_no_bias(seq)  
        f_1 = self.conv_f1(seq_fts)  
        f_2 = self.conv_f2(seq_fts)  
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat) 
        coefs = self.coef_dropout(coefs, training=training)  
        seq_fts = self.in_dropout(seq_fts, training=training)  
        vals = tf.matmul(coefs, seq_fts)  
        
        vals = tf.cast(vals, dtype=tf.float32)  
        ret = vals + self.bias_zero  
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)
            else:
                ret = ret + seq
        return self.activation(ret)


class DKFCPcell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, M, embedding_size, **kwargs):
        
        self.units = units
        self.supports_masking = True
        super(DKFCPcell, self).__init__(**kwargs)
        

        
        self.Mv = self.add_weight(shape=(M, embedding_size),initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),trainable=True)
        self.Mv = tf.expand_dims(self.Mv, axis=0)  
        
    @property
    def state_size(self):
        return self.units

    def get_Mv(self,stunum, M, embedding_size,mean,stddev):
        Mv_1 = tf.Variable(tf.random.normal(shape=(stunum, M, embedding_size), mean=mean, stddev=stddev))
        
        return Mv_1

    def __call__(self, w_attention,  states, mask=None):
        
        r = tf.matmul(tf.expand_dims(w_attention, axis=1), states)  
        r = r[:, 0, :]  

        return r







def compute_ndcg(scores, relevance, n=10):

    scores = list(map(float,scores))
    relevance = list(map(float, relevance))
    array = list(zip(scores, relevance))
    dcg_array = sorted(array, key = lambda x:x[0], reverse = True)[:n]
    idcg_array = sorted(dcg_array, key = lambda x:x[1], reverse = True)
    dcg = 0
    for i, (_, rel) in enumerate(dcg_array):
        dcg += (2**rel - 1)/np.log2(i+2)
    idcg = 0
    for i, (_, rel) in enumerate(idcg_array):
        idcg += (2**rel - 1)/np.log2(i+2)

    ndcg = dcg/(idcg+1e-6)
    return float(ndcg)

def NDCG(y_true,pt,n):
    ndcg = 0
    ndi = 0
    for (pti, y_truei) in zip(pt, y_true):
        ndi = ndi + 1
        ndcg = ndcg + compute_ndcg(pti, y_truei, n)
    ndcg = ndcg / ndi
    return ndcg

class inference(tf.keras.layers.Layer):
    def __init__(self, n_heads, hid_units, nbStu_nodes, nbCou_nodes, emb_dim, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):

        super(inference, self).__init__()



        self.final_attns = []
        self.ICCI_attns = []
        self.ICCSI_attns = []
        self.CTC_attns = []
        self.CCC_attns = []

        self.final_sum = n_heads[-1]
        self.emb_dim = emb_dim


        self.stuFactor = RawToFactor(emb_dim=emb_dim, factorDim=factorDimStu, in_drop=ffd_drop, activation=activation, residual=residual)
        self.CouFactor = RawToFactor(emb_dim=emb_dim, factorDim=factorDimCou, in_drop=ffd_drop, activation=activation, residual=residual)



        for i in range(n_heads[-1]):
            self.ICCI_attns.append(attn_head(hidden_dim=hid_units[0], nb_nodes=nbCou_nodes,
                                                in_drop=ffd_drop, coef_drop=attn_drop,
                                                activation=lambda x: x,
                                                residual=residual))

        for i in range(n_heads[-1]):
            self.ICCSI_attns.append(attn_head(hidden_dim=hid_units[0], nb_nodes=nbCou_nodes,
                                                in_drop=ffd_drop, coef_drop=attn_drop,
                                                activation=lambda x: x,
                                                residual=residual))

        for i in range(n_heads[-1]):
            self.CTC_attns.append(attn_head(hidden_dim=hid_units[0], nb_nodes=nbCou_nodes,
                                                in_drop=ffd_drop, coef_drop=attn_drop,
                                                activation=lambda x: x,
                                                residual=residual))

        for i in range(n_heads[-1]):
            self.CCC_attns.append(attn_head(hidden_dim=hid_units[0], nb_nodes=nbCou_nodes,
                                                in_drop=ffd_drop, coef_drop=attn_drop,
                                                activation=lambda x: x,
                                                residual=residual))


        self.HIN_MLPICCSI = tf.keras.layers.Dense(emb_dim, kernel_regularizer='l2')
        self.HIN_MLPICCI = tf.keras.layers.Dense(emb_dim, kernel_regularizer='l2')
        self.HIN_MLPCTC = tf.keras.layers.Dense(emb_dim, kernel_regularizer='l2')
        self.HIN_MLPCCC = tf.keras.layers.Dense(emb_dim, kernel_regularizer='l2')






        self.mask = tf.keras.layers.Masking(mask_value=float("inf"))


        self.att = w_atten(axes=(2, 1))
        self.softmax_att = tf.keras.layers.Softmax()
        self.cell = DKFCPcell(units=10, M=Mnum, embedding_size=emb_dim)
        self.Wm = tf.keras.layers.Dense(Mnum, kernel_regularizer='l2')





        self.Mk = self.add_weight(shape=(Mnum, emb_dim+1), initializer='random_normal', trainable=True)

        self.erase = tf.keras.layers.Dense(emb_dim, activation="sigmoid", kernel_regularizer='l2')
        self.add = tf.keras.layers.Dense(emb_dim, activation="tanh", kernel_regularizer='l2')
        self.r = tf.keras.layers.Dense(emb_dim, activation="tanh", kernel_regularizer='l2')
        self.p = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer='l2')


    def __call__(self, inputSCon, inputSEmbRaw, inputCCon, inputCEmbRaw, bias_mat_adj_x4,bias_mat_ICCI,
                 bias_mat_ICCSI,bias_mat_CTC,bias_mat_CCC,training, train_index,S_CPopAll, tmpAll,lenSCnsNegInput,ifndcg=0):


        SFactor = self.stuFactor(inputSEmbRaw, inputSCon, training)
        CFactor = self.CouFactor(inputCEmbRaw, inputCCon, training)

        out = []
        for indiv_attn in self.final_attns:
            out.append(indiv_attn(seq=CFactor, bias_mat=bias_mat_adj_x4, training=training))



        ICCIout = []
        for indiv_attn in self.ICCI_attns:
            ICCIout.append(indiv_attn(seq=CFactor, bias_mat=bias_mat_ICCI, training=training))
        Factor_compICCI = tf.add_n(ICCIout) / self.final_sum

        ICCSIout = []
        for indiv_attn in self.ICCSI_attns:
            ICCSIout.append(indiv_attn(seq=CFactor, bias_mat=bias_mat_ICCSI, training=training))
        Factor_compICCSI = tf.add_n(ICCSIout) / self.final_sum

        CTCout = []
        for indiv_attn in self.CTC_attns:
            CTCout.append(indiv_attn(seq=CFactor, bias_mat=bias_mat_CTC, training=training))
        Factor_compCTC = tf.add_n(CTCout) / self.final_sum

        CCCout = []
        for indiv_attn in self.CCC_attns:
            CCCout.append(indiv_attn(seq=CFactor, bias_mat=bias_mat_CCC, training=training))
        Factor_compCCC = tf.add_n(CCCout) / self.final_sum


        Factor_comp = (self.HIN_MLPICCI(Factor_compICCI)+self.HIN_MLPICCSI(Factor_compICCSI)+self.HIN_MLPCTC(Factor_compCTC)+self.HIN_MLPCCC(Factor_compCCC))/4

        Course_Factor = Factor_comp[0]
        SFforM0 = SFactor[0]
        StuCouFacAll = tf.matmul(tmpAll, Course_Factor)
        S_CFAll = tf.concat([StuCouFacAll, S_CPopAll], axis=-1)

        shuffle_index = tf.concat([[0], tf.random.shuffle(tf.range(tf.shape(S_CFAll)[1])[1:])], axis=0)

        S_C_FInputPN = tf.gather(S_CFAll, shuffle_index, axis=1)
        _, temp_seme, _ = tf.split(S_C_FInputPN, [-1, 1, 1], axis=2)
        temp_seme_indes0 = tf.argsort(temp_seme, direction='ASCENDING', axis=1)
        S_C_FInputPNSS = tf.gather_nd(S_C_FInputPN, temp_seme_indes0, batch_dims = 1)
        S_C_FInputPNPre, S_C_seme, y_true0 = tf.split(S_C_FInputPNSS, [-1,1,1], axis=2)
        y_true = tf.logical_and(tf.greater_equal(y_true0, tf.zeros(shape=y_true0.shape, dtype=tf.float32)),
                       tf.less_equal(y_true0, tf.ones(shape=y_true0.shape, dtype=tf.float32)), name='y_and')

        S_C_FPNSS = self.mask(S_C_FInputPNSS)
        S_C_FPNPre = self.mask(S_C_FInputPNPre)
        S_C_FPNPre_T = tf.convert_to_tensor(S_C_FPNPre, tf.float32, name='S_C_FPNPre_T')
        w_attention = self.att([S_C_FPNPre_T, tf.tile(tf.expand_dims(tf.transpose(self.Mk), axis=0), multiples=[S_C_FPNPre_T.shape[0],1,1])])

        print(w_attention.shape)
        w_attention = self.softmax_att(w_attention)

        erase_signal = self.erase(S_C_FPNSS)
        add_signal = self.add(S_C_FPNSS)



        trainstu = tf.expand_dims(SFforM0[train_index[0]], axis=0)
        for i in range(len(train_index))[1:]:
            trainstu = tf.concat([trainstu, tf.expand_dims(SFforM0[train_index[i]], axis=0)], axis=0)
        MSmean = self.Wm(trainstu)
        MSmean = tf.expand_dims(MSmean, axis=-1)

        print(self.cell.Mv.shape)
        print(MSmean.shape)
        MVVVV=self.cell.Mv
        for i in range(len(train_index))[1:]:
            MVVVV = tf.concat([MVVVV, self.cell.Mv], axis=0)

        states = tf.multiply(MVVVV, MSmean)
        stateseme = tf.multiply(MVVVV, MSmean)

        print(states.shape)
        print(stateseme.shape)

        erase_signal_mul = 1 - tf.expand_dims(w_attention, axis=-1) * tf.expand_dims(erase_signal, axis=2)
        add_signal_mul = tf.expand_dims(w_attention, axis=-1) * tf.expand_dims(add_signal, axis=2)
        shape = S_C_FPNPre.shape
        r = self.cell(w_attention[:, 0],  stateseme, mask=w_attention._keras_mask[:, 0])

        r = tf.concat([r, S_C_FPNPre[:, 0]], axis=-1)
        pt = self.p(self.r(r))
        ptT_F = tf.greater_equal(pt, tf.constant(0.5, shape=(shape[0], 1)))
        pt = tf.expand_dims(pt, axis=1)
        ptT_F = tf.expand_dims(ptT_F, axis=1)
        y_t = tf.reshape(ptT_F[:, 0], [-1])
        true_index = tf.where(y_t == True)
        states = tf.tensor_scatter_nd_update(states, true_index,(states * erase_signal_mul[:, 0] + add_signal_mul[:, 0])[
                                                 y_t])



        for i in range(shape[1])[1:]:
            new_seme = S_C_seme[:, i]-S_C_seme[:, i-1]
            ns_index = tf.reshape(new_seme, [-1])
            bns_index = tf.greater(ns_index, 0)
            new_seme_index = tf.where(bns_index == True)

            stateseme=tf.tensor_scatter_nd_update(stateseme, new_seme_index, (states)[bns_index])

            r = self.cell(w_attention[:, i], stateseme, mask=w_attention._keras_mask[:, i])

            r = tf.concat([r, S_C_FPNPre[:, i]], axis=-1)
            temp_pt = self.p(self.r(r))
            temp_ptT_F = tf.greater_equal(temp_pt, tf.constant(0.5, shape=(shape[0], 1)))
            temp_pt = tf.expand_dims(temp_pt, axis=1)
            temp_ptT_F = tf.expand_dims(temp_ptT_F, axis=1)
            pt = tf.concat([pt, temp_pt], axis=1)
            ptT_F = tf.concat([ptT_F, temp_ptT_F], axis=1)
            y_t = tf.reshape(ptT_F[:, i], [-1])
            true_index = tf.where(y_t == True)

            states = tf.tensor_scatter_nd_update(states, true_index,(states  + add_signal_mul[:, i])[y_t])




        loss = tf.keras.losses.binary_crossentropy(y_true, pt)


        accuracY = tf.keras.metrics.BinaryAccuracy()
        accuracY.update_state(y_true, pt)
        accuracy = accuracY.result().numpy()


        rmsE = tf.keras.metrics.RootMeanSquaredError()
        rmsE.update_state(y_true, pt)
        rmse = rmsE.result().numpy()

        auC = tf.keras.metrics.AUC()
        auC.update_state(y_true, pt)
        auc = auC.result().numpy()

        ptRecall = tf.squeeze(pt)
        y_trueRecall = tf.squeeze(y_true)

        ndcg = ndcg_score(y_trueRecall, ptRecall, k=10)
        ndcg20=ndcg_score(y_trueRecall, ptRecall, k=20)
        ndcg30 = ndcg_score(y_trueRecall, ptRecall, k=30)
        ndcg40 = ndcg_score(y_trueRecall, ptRecall, k=40)
        ndcg50 = ndcg_score(y_trueRecall, ptRecall, k=50)

        recalL10 = tf.keras.metrics.Recall(top_k=10)
        recalL10.update_state(y_trueRecall, ptRecall)
        recall10 = recalL10.result().numpy()
        print('----shape-----')
        print(y_trueRecall.shape)
        print(ptRecall.shape)
        recalL20 = tf.keras.metrics.Recall(top_k=20)
        recalL20.update_state(y_trueRecall, ptRecall)
        recall20 = recalL20.result().numpy()
        recalL30 = tf.keras.metrics.Recall(top_k=30)
        recalL30.update_state(y_trueRecall, ptRecall)
        recall30 = recalL30.result().numpy()
        recalL40 = tf.keras.metrics.Recall(top_k=40)
        recalL40.update_state(y_trueRecall, ptRecall)
        recall40 = recalL40.result().numpy()
        recalL50 = tf.keras.metrics.Recall(top_k=50)
        recalL50.update_state(y_trueRecall, ptRecall)
        recall50 = recalL50.result().numpy()
        recalL60 = tf.keras.metrics.Recall(top_k=60)
        recalL60.update_state(y_trueRecall, ptRecall)
        recall60 = recalL60.result().numpy()
        recalL70 = tf.keras.metrics.Recall(top_k=70)
        recalL70.update_state(y_trueRecall, ptRecall)
        recall70 = recalL70.result().numpy()
        recalL80 = tf.keras.metrics.Recall(top_k=80)
        recalL80.update_state(y_trueRecall, ptRecall)
        recall80 = recalL80.result().numpy()
        recalL90 = tf.keras.metrics.Recall(top_k=90)
        recalL90.update_state(y_trueRecall, ptRecall)
        recall90 = recalL90.result().numpy()
        recalL = tf.keras.metrics.Recall()
        recalL.update_state(y_trueRecall, ptRecall)
        recall = recalL.result().numpy()

        y_trueRecall1,y_trueRecall2,y_trueRecall3,y_trueRecall4,y_trueRecall5,y_trueRecall6=tf.split(y_trueRecall, [118,180,68,54,28,33], axis=-1)
        ptRecall1,ptRecall2,ptRecall3,ptRecall4,ptRecall5,ptRecall6=tf.split(ptRecall, [118,180,68,54,28,33], axis=-1)
        auC10 = tf.keras.metrics.Recall(top_k=10)
        auC10.update_state(y_trueRecall1, ptRecall1)
        auc10 = auC10.result().numpy()

        auC20 = tf.keras.metrics.Recall(top_k=10)
        auC20.update_state(y_trueRecall2, ptRecall2)
        auc20 = auC20.result().numpy()
        auC30 = tf.keras.metrics.Recall(top_k=10)
        auC30.update_state(y_trueRecall3, ptRecall3)
        auc30 = auC30.result().numpy()
        auC40 = tf.keras.metrics.Recall(top_k=10)
        auC40.update_state(y_trueRecall4, ptRecall4)
        auc40 = auC40.result().numpy()
        auC50 = tf.keras.metrics.Recall(top_k=10)
        auC50.update_state(y_trueRecall5, ptRecall5)
        auc50 = auC50.result().numpy()
        auC60 = tf.keras.metrics.Recall(top_k=10)
        auC60.update_state(y_trueRecall6, ptRecall6)
        auc60 = auC60.result().numpy()
        auC70 = tf.keras.metrics.AUC()
        auC70.update_state(y_true[:70], pt[:70])
        auc70 = auC70.result().numpy()
        print(f'ndcg50:{ndcg50}; recall50: {recall50}')
        return loss, accuracy, auc, rmse, [recall10,recall20,recall30,recall40,recall50,recall60,recall70,recall80,recall90,recall], [auc10,auc20,auc30,auc40,auc50,auc60,auc70], [ndcg,ndcg20,ndcg30,ndcg40,ndcg50]



class CoursePrediction(tf.keras.Model):
    def __init__(self, hid_units, n_heads, nbStu_nodes, nbCou_nodes, emb_dim, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):

        super(CoursePrediction, self).__init__()

        self.hid_units = hid_units
        self.n_heads = n_heads
        self.nbStu_nodes = nbStu_nodes
        self.nbCou_nodes = nbCou_nodes
        self.activation = activation
        self.residual = residual
        self.inferencing = inference(n_heads, hid_units, nbStu_nodes, nbCou_nodes, emb_dim, ffd_drop=ffd_drop,
                                     attn_drop=attn_drop, activation=activation, residual=residual)

    def __call__(self, inputSCon, inputSEmbRaw, inputCCon, inputCEmbRaw, training, bias_mat_adj_x4,bias_mat_ICCI,
          bias_mat_ICCSI, bias_mat_CTC, bias_mat_CCC, train_index,S_CPopAll,tmpAll, lenSCnsNegInput,ifndcg):

        loss, accuracy, auc, rmse, recall, aucf7,ndcg = self.inferencing( inputSCon=inputSCon,
                                  inputSEmbRaw=inputSEmbRaw,
                                  inputCCon=inputCCon,
                                  inputCEmbRaw=inputCEmbRaw,
                                  bias_mat_adj_x4=bias_mat_adj_x4,
                                  bias_mat_ICCI=bias_mat_ICCI,
                                  bias_mat_ICCSI=bias_mat_ICCSI,
                                  bias_mat_CTC=bias_mat_CTC,
                                  bias_mat_CCC=bias_mat_CCC,
                                  training=training,
                                  train_index=train_index,
                                  S_CPopAll=S_CPopAll,
                                  tmpAll=tmpAll,
                                  lenSCnsNegInput=lenSCnsNegInput,
                                ifndcg=ifndcg
                                  )
        return loss, accuracy, auc, rmse, recall, aucf7,ndcg


def train(model, inputSCon, inputSEmbRaw, inputCCon, inputCEmbRaw, bias_mat_adj_x4,bias_mat_ICCI,
          bias_mat_ICCSI, bias_mat_CTC, bias_mat_CCC, training, train_index,S_CPopAll, tmpAll, lenSCnsNegInput,ifndcg):

    with tf.GradientTape() as tape:

        loss, accuracy, auc, rmse, recall, aucf7 ,ndcg= model(inputSCon=inputSCon, inputSEmbRaw=inputSEmbRaw,
                               inputCCon=inputCCon, inputCEmbRaw=inputCEmbRaw,
                               training=training,
                               bias_mat_adj_x4=bias_mat_adj_x4,
                               bias_mat_ICCI=bias_mat_ICCI,
                               bias_mat_ICCSI=bias_mat_ICCSI,
                               bias_mat_CTC=bias_mat_CTC,
                               bias_mat_CCC=bias_mat_CCC,
                               train_index=train_index,
                               S_CPopAll=S_CPopAll,
                               tmpAll=tmpAll,
                               lenSCnsNegInput=lenSCnsNegInput,ifndcg =ifndcg)

    gradients = tape.gradient(loss, model.trainable_variables)

    gradient_variables = zip(gradients, model.trainable_variables)
    optimizer.apply_gradients(gradient_variables)

    return loss, accuracy, auc, rmse, recall, aucf7, ndcg


def evaluate(model, inputSCon, inputSEmbRaw, inputCCon, inputCEmbRaw, bias_mat_adj_x4,bias_mat_ICCI,
          bias_mat_ICCSI, bias_mat_CTC, bias_mat_CCC,training, test_index,S_CPopAll,  tmpAll, lenSCnsNegInput,ifndcg):

    loss, accuracy, auc, rmse, recall,aucf7,ndcg = model(inputSCon=inputSCon, inputSEmbRaw=inputSEmbRaw,
                           inputCCon=inputCCon, inputCEmbRaw=inputCEmbRaw,
                           training=False,
                           bias_mat_adj_x4=bias_mat_adj_x4,
                           bias_mat_ICCI=bias_mat_ICCI,
                           bias_mat_ICCSI=bias_mat_ICCSI,
                           bias_mat_CTC=bias_mat_CTC,
                           bias_mat_CCC=bias_mat_CCC,
                           train_index=test_index,
                           S_CPopAll=S_CPopAll,
                           tmpAll=tmpAll,
                           lenSCnsNegInput=lenSCnsNegInput,
                            ifndcg=ifndcg
                           )


    return loss, accuracy, auc, rmse, recall,aucf7, ndcg

begin_time = time()


path = 'CPData\\'
Dataset = 'Stu2014'
Sparse = False
Batch_Size = 1
Epochs = EpochsCPTrain
Patience = 100
Learning_Rate = 0.005
Weight_Decay = 0.0005
ffd_drop = 0.3
attn_drop = 0.3
Residual = True
dataset = Dataset


batch_size = Batch_Size
nb_epochs = Epochs
patience = Patience
lr = Learning_Rate
l2_coef = Weight_Decay
residual = Residual
hid_units = [GATHiddenDimCou]
n_heads = [8, 1]
nonlinearity = tf.nn.elu
optimizer = tf.keras.optimizers.Adam(lr = lr)




print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))


StuIdList, StuIdPosDict, CouIdList, CouIdPosDict, CouCodeList, CouCodePosDict, TeaList, TeaPosDict, DorList, DorPosDict, \
adjCC, adjCCS, Stu_CouScoreM, Stu_CouSemeM, \
Cou_TeaM, CouTeaCouM, CouId_CodeM, CIdCodeIdM, Stu_DorM, StuDorStuM, \
FeatureStuSM, FeatureCouSM, EmbCouNameM, EmbStuMajorM, \
SList_train, SList_test, S_mask_train, S_mask_test, \
StuCouScore_train, StuCouScore_test, StuCouSeme_train, StuCouSeme_test \
    = load_data(path+dataset+'\\')

nbStu_nodes = FeatureStuSM.shape[0]
nbCou_nodes = FeatureCouSM.shape[0]
ftStu_size = FeatureStuSM.shape[1]
ftCou_size = FeatureCouSM.shape[1]


FeatureStuSM = FeatureStuSM.todense()[np.newaxis]
FeatureCouSM = FeatureCouSM.todense()[np.newaxis]
EmbCouNameM = EmbCouNameM[np.newaxis]
EmbStuMajorM = EmbStuMajorM[np.newaxis]
S_mask_train = S_mask_train[np.newaxis]
S_mask_test = S_mask_test[np.newaxis]
print(f'These are some of the parameters')
print(f'batch_size: {batch_size}')
print(f'nbStu_nodes: {nbStu_nodes}')
print(f'nbCou_nodes: {nbCou_nodes}')
print(f'ftStu_size: {ftStu_size}')
print(f'ftCou_size: {ftCou_size}')


adjCC = adjCC.todense()
adjCCS = adjCCS.todense()
Stu_CouScoreM = np.mat(Stu_CouScoreM)
Stu_CouSemeM = np.mat(Stu_CouSemeM)
Cou_TeaM = np.mat(Cou_TeaM)
CouId_CodeM = np.mat(CouId_CodeM)
Stu_DorM = np.mat(Stu_DorM)


ICCI = np.dot(np.dot(CouId_CodeM, adjCC), CouId_CodeM.transpose())
ICCSI = tf.matmul(tf.matmul(CouId_CodeM, adjCCS), CouId_CodeM.transpose())
adj_x4 = tf.add_n([ICCI, ICCSI, CouTeaCouM, CIdCodeIdM])


ICCI = ICCI[np.newaxis]
ICCSI = ICCSI[np.newaxis]
CouTeaCouM = CouTeaCouM[np.newaxis]
CIdCodeIdM = CIdCodeIdM[np.newaxis]
StuDorStuM = StuDorStuM[np.newaxis]
adj_x4 = adj_x4[np.newaxis]
Stu_CouScoreM = Stu_CouScoreM[np.newaxis]
Stu_CouSemeM = Stu_CouSemeM[np.newaxis]



biasesICCI = adj_to_bias(ICCI, [nbCou_nodes], nhood=1)
biasesICCSI = adj_to_bias(ICCSI, [nbCou_nodes], nhood=1)
biasesCTC = adj_to_bias(CouTeaCouM, [nbCou_nodes], nhood=1)
biasesCCC = adj_to_bias(CIdCodeIdM, [nbCou_nodes], nhood=1)
biasesSDS = adj_to_bias(StuDorStuM, [nbStu_nodes], nhood=1)
biasesadj_x4 = adj_to_bias(adj_x4, [nbCou_nodes], nhood=1)

vlss_mn = np.inf
vacc_mx = 0.0
curr_step = 0
train_loss_avg = 0
train_accuracy_avg = 0
val_loss_avg = 0
val_accuracy_avg = 0

train_auc_avg = 0
train_rmse_avg = 0
train_recall_avg = [0,0,0,0,0,0,0,0,0,0]
val_auc_avg = 0
val_rmse_avg = 0
val_recall_avg = [0,0,0,0,0,0,0,0,0,0]

train_aucf7_avg = [0,0,0,0,0,0,0]
val_aucf7_avg = [0,0,0,0,0,0,0]
train_ndcg_avg = [0,0,0,0,0]
val_ndcg_avg = [0,0,0,0,0]

model_number = 0
StuId = np.array(StuIdList)
CouId = np.array(CouIdList)
StuId = StuId[np.newaxis]
CouId = CouId[np.newaxis]



msk_train = S_mask_train
mask_test = S_mask_test
inputSCon = FeatureStuSM
inputSEmbRaw = EmbStuMajorM
inputCCon = FeatureCouSM
inputCEmbRaw = EmbCouNameM

_, SemeStatistics=tf.split(inputCCon, [-1,1], axis=2)
SemeStatistics1 = tf.reshape(SemeStatistics[0], [-1]).numpy()
print(f'详细的课程学期情况：{Counter(SemeStatistics1)}')



train_index = np.where(msk_train[0] == True)[0]
Stu_CouScoreMT = Stu_CouScoreM[0, msk_train[0],:][np.newaxis]
Stu_CouSemeMT = Stu_CouSemeM[0, msk_train[0], :][np.newaxis]
Stu_CouScore = Stu_CouScoreMT[0].getA()
Stu_CouSeme = Stu_CouSemeMT[0].getA()
stuNum = Stu_CouScore.shape[0]
couNum = Stu_CouScore.shape[1]

where_1 = np.where(Stu_CouScore != -1)
where_0 = np.where(Stu_CouScore == -1)
SCPop = np.zeros((stuNum, couNum))
SCPop[where_1] = 1
course_pop = np.mean(SCPop, axis=0)


test_index = np.where(mask_test[0] == True)[0]
Stu_CouScoreMTtest = Stu_CouScoreM[0, mask_test[0],:][np.newaxis]
Stu_CouSemeMTtest = Stu_CouSemeM[0, mask_test[0], :][np.newaxis]
Stu_CouScoretest = Stu_CouScoreMTtest[0].getA()
Stu_CouSemetest = Stu_CouSemeMTtest[0].getA()
stuNumtest = Stu_CouScoretest.shape[0]
couNumtest = Stu_CouSemetest.shape[1]
SC_index_score = np.where(Stu_CouScore != -1)
SC_index_null = np.where(Stu_CouScore == -1)
SC_index_PN = np.where(Stu_CouScore != None)
stu_index_cho = SC_index_score[0]
cou_index_cho = SC_index_score[1]
stu_index_ncho = SC_index_null[0]
cou_index_ncho = SC_index_null[1]
stu_index = SC_index_PN[0]
cou_index = SC_index_PN[1]
lenSCFactor = max(Counter(stu_index_cho).values())
lenSCnull = max(Counter(stu_index_ncho).values())
lenSC = max(Counter(stu_index).values())
print(f'lenSC: {lenSC}')
lenSCnsNegInput = min(lenSCnull, SampleNegative*lenSCFactor)

SC_index_scoretest = np.where(Stu_CouScoretest != -1)
SC_index_nulltest = np.where(Stu_CouScoretest == -1)
SC_index_test = np.where(Stu_CouScoretest != None)
stu_index_chotest = SC_index_scoretest[0]
cou_index_chotest = SC_index_scoretest[1]
stu_index_nchotest = SC_index_nulltest[0]
cou_index_nchotest = SC_index_nulltest[1]
stu_index_test = SC_index_test[0]
cou_index_test = SC_index_test[1]
lenSCFactortest = max(Counter(stu_index_chotest).values())
lenSCnulltest = max(Counter(stu_index_nchotest).values())
lenSCtest = max(Counter(stu_index_test).values())
lenSCnsNegInputtest = min(lenSCnulltest, SampleNegative*lenSCFactortest)


indicesAll = [np.concatenate((np.where(Stu_CouScore[0] != None)[0],  np.array([-1]*(lenSC-len(np.where(Stu_CouScore[0] != None)[0])))), axis=0)]
for i in Stu_CouScore[1:]:
    listiAll = np.where(i != None)[0]
    listiAll = np.concatenate((listiAll, np.array([-1]*(lenSC - len(listiAll)))), axis=0)
    indicesAll = np.concatenate((indicesAll, [listiAll]), axis=0)
tmpAll = tf.one_hot(indicesAll, couNum, on_value=1.0, off_value=0.0, axis=-1)


indicesAlltest = [np.concatenate((np.where(Stu_CouScoretest[0] != None)[0],  np.array([-1]*(lenSCtest-len(np.where(Stu_CouScoretest[0] != None)[0])))), axis=0)]
for i in Stu_CouScoretest[1:]:
    listiAlltest = np.where(i != None)[0]
    listiAlltest = np.concatenate((listiAlltest, np.array([-1]*(lenSCtest - len(listiAlltest)))), axis=0)
    indicesAlltest = np.concatenate((indicesAlltest, [listiAlltest]), axis=0)
tmpAlltest = tf.one_hot(indicesAlltest, couNumtest, on_value=1.0, off_value=0.0, axis=-1)



S_CPopSeScPN = 0
S_CPopSemeScore = [[[float(-1) for k in range(3)] for j in range(lenSCFactor)] for i in range(len(Counter(stu_index_cho)))]
S_CPopNotChange = [[[float(-1) for k in range(3)] for j in range(lenSCnull)] for i in range(len(Counter(stu_index_ncho)))]
S_CPopAll = [[[float(-1) for k in range(3)] for j in range(lenSC)] for i in range(len(Counter(stu_index)))]
S_CPopSemeScore = np.array(S_CPopSemeScore)
S_CPopNotChange = np.array(S_CPopNotChange)
S_CPopAll = np.array(S_CPopAll)


SCNAll = [0 for i in range(len(Counter(stu_index)))]

S_CPopSemeScoretest = [[[float(-1) for k in range(3)] for j in range(lenSCFactortest)] for i in range(len(Counter(stu_index_chotest)))]
S_CPopNotChangetest = [[[float(-1) for k in range(3)] for j in range(lenSCnulltest)] for i in range(len(Counter(stu_index_nchotest)))]
S_CPopAlltest = [[[float(-1) for k in range(3)] for j in range(lenSCtest)] for i in range(len(Counter(stu_index_test)))]
S_CPopSemeScoretest = np.array(S_CPopSemeScoretest)
S_CPopNotChangetest = np.array(S_CPopNotChangetest)
S_CPopAlltest = np.array(S_CPopAlltest)


SCNAlltest = [0 for i in range(len(Counter(stu_index_test)))]

for i in range(len(stu_index)):
    if Stu_CouScore[stu_index[i]][cou_index[i]] != -1:
        tempAll = tf.concat([[course_pop[cou_index[i]]], [Stu_CouSeme[stu_index[i]][cou_index[i]] / 5],
                           [Stu_CouScore[stu_index[i]][cou_index[i]] / 100]], axis=0)
    else:
        tempAll = tf.concat([[course_pop[cou_index[i]]], [float("%0.2f"%inputCCon.getA()[0][cou_index[i]][-1])], [-1]], axis=0)
    S_CPopAll[stu_index[i]][SCNAll[stu_index[i]]] = tempAll
    SCNAll[stu_index[i]] = SCNAll[stu_index[i]] + 1



for i in range(len(stu_index_test)):
    if Stu_CouScoretest[stu_index_test[i]][cou_index_test[i]] != -1:
        tempAlltest = tf.concat([[course_pop[cou_index_test[i]]], [Stu_CouSemetest[stu_index_test[i]][cou_index_test[i]] / 5],
                           [Stu_CouScoretest[stu_index_test[i]][cou_index_test[i]] / 100]], axis=0)
    else:
        tempAlltest = tf.concat([[course_pop[cou_index_test[i]]], [float("%0.2f"%inputCCon.getA()[0][cou_index_test[i]][-1])], [-1]], axis=0)
    S_CPopAlltest[stu_index_test[i]][SCNAlltest[stu_index_test[i]]] = tempAlltest
    SCNAlltest[stu_index_test[i]] = SCNAlltest[stu_index_test[i]] + 1


model = CoursePrediction(hid_units, n_heads, nbStu_nodes, nbCou_nodes, emb_dim=EmbFinDim, ffd_drop=ffd_drop, attn_drop=attn_drop, activation=tf.nn.elu, residual=residual)
print('model: ' + str('SpCoursePrediction' if Sparse else 'CoursePrediction'))
dataprocess_time = time()
run_time = dataprocess_time-begin_time
print ('data process time before train:', run_time)
print ('-----start train and evaluate-----')

for epoch in range(nb_epochs):
    
    tr_step = 0
    tr_size = FeatureStuSM.shape[0]  
    print(f'epoch:{epoch}')
    while tr_step * batch_size < tr_size:  
        bbiasICCI = biasesICCI[tr_step * batch_size:(tr_step + 1) * batch_size]  
        bbiasICCSI = biasesICCSI[tr_step * batch_size:(tr_step + 1) * batch_size]
        bbiasCTC = biasesCTC[tr_step * batch_size:(tr_step + 1) * batch_size]
        bbiasCCC = biasesCCC[tr_step * batch_size:(tr_step + 1) * batch_size]
        bbiasSDS = biasesSDS[tr_step * batch_size:(tr_step + 1) * batch_size]
        bbiasadj_x4 = biasesadj_x4[tr_step * batch_size:(tr_step + 1) * batch_size]
        loss_value_tr, accuracy_value_tr, auc_value_tr, rmse_value_tr, recall_value_tr, aucf7_value_tr, recall_ndcg_tr = train(model,inputSCon=inputSCon,
                                         inputSEmbRaw=inputSEmbRaw,
                                         inputCCon=inputCCon,
                                         inputCEmbRaw=inputCEmbRaw,
                                         bias_mat_adj_x4=bbiasadj_x4,
                                         bias_mat_ICCI = bbiasICCI,
                                         bias_mat_ICCSI = bbiasICCSI,
                                         bias_mat_CTC=bbiasCTC,
                                         bias_mat_CCC=bbiasCCC,
                                         training=True,
                                         train_index=train_index,
                                         S_CPopAll = S_CPopAll,
                                         tmpAll=tmpAll,
                                         lenSCnsNegInput=lenSCnsNegInput,ifndcg=0)

        loss_value_tr_mean = tf.reduce_mean(loss_value_tr)  
        train_loss_avg += loss_value_tr_mean  
        train_accuracy_avg += accuracy_value_tr
        train_auc_avg += auc_value_tr
        train_rmse_avg += rmse_value_tr
        train_recall_avg += recall_value_tr
        train_ndcg_avg += recall_ndcg_tr
        train_aucf7_avg += aucf7_value_tr
        tr_step += 1
    after_train_time = time()
    run_time = after_train_time-dataprocess_time
    print('train time:', run_time)  

    
    vl_step = 0
    vl_size = FeatureStuSM.shape[0]
    while vl_step * batch_size < vl_size:
        bbiasICCI = biasesICCI[vl_step * batch_size:(vl_step + 1) * batch_size]
        bbiasICCSI = biasesICCSI[vl_step * batch_size:(vl_step + 1) * batch_size]
        bbiasCTC = biasesCTC[vl_step * batch_size:(vl_step + 1) * batch_size]
        bbiasCCC = biasesCCC[vl_step * batch_size:(vl_step + 1) * batch_size]
        bbiasSDS = biasesSDS[vl_step * batch_size:(vl_step + 1) * batch_size]
        bbiasadj_x4 = biasesadj_x4[vl_step * batch_size:(vl_step + 1) * batch_size]
        
        


        loss_value_vl, accuracy_value_vl, auc_value_vl, rmse_value_vl, recall_value_vl,aucf7_value_vl ,ndcg_value_vl= evaluate(model, inputSCon=inputSCon,
                                                    inputSEmbRaw=inputSEmbRaw,
                                                    inputCCon=inputCCon,
                                                    inputCEmbRaw=inputCEmbRaw,
                                                    bias_mat_adj_x4=bbiasadj_x4,
                                                    bias_mat_ICCI=bbiasICCI,
                                                    bias_mat_ICCSI=bbiasICCSI,
                                                    bias_mat_CTC=bbiasCTC,
                                                    bias_mat_CCC=bbiasCCC,
                                                    training=False,
                                                    test_index=test_index,
                                                    S_CPopAll=S_CPopAlltest,
                                                    tmpAll=tmpAlltest,
                                                    lenSCnsNegInput=lenSCnsNegInputtest,
                                                    ifndcg=0)
        loss_value_vl_mean = tf.reduce_mean(loss_value_vl)
        val_loss_avg += loss_value_vl_mean  
        val_accuracy_avg += accuracy_value_vl  
        val_auc_avg += auc_value_vl
        val_rmse_avg += rmse_value_vl
        val_recall_avg += recall_value_vl
        val_aucf7_avg += aucf7_value_vl
        val_ndcg_avg += ndcg_value_vl
        vl_step += 1
    after_evlauate_time = time()
    run_time = after_evlauate_time-dataprocess_time
    print('train time:', run_time)  
    print('Training: loss = %.5f | Val: loss = %.5f' %(train_loss_avg / tr_step, val_loss_avg / vl_step))
    print('Training: accuracy = %.5f | Val: accuracy = %.5f' % (train_accuracy_avg / tr_step, val_accuracy_avg / vl_step))
    print('Training: auc = %.5f | Val: auc = %.5f' % (train_auc_avg / tr_step, val_auc_avg / vl_step))
    print('Training: rmse = %.5f | Val: rmse = %.5f' % (train_rmse_avg / tr_step, val_rmse_avg / vl_step))
    print('training-recall: 10-70:', train_recall_avg)
    print('evaluate-recall: 10-70:', val_recall_avg)
    print('training-recall10: 1-6 & AUC:', train_aucf7_avg)
    print('evaluate-recall10: 1-6 & AUC:', val_aucf7_avg)
    print('training-ndcg: ', train_ndcg_avg)
    print('evaluate-ndcg: ', val_ndcg_avg)

    if vl_step>1:
        print('error')
        break

    train_loss_avg = 0
    val_loss_avg = 0
    train_accuracy_avg = 0
    val_accuracy_avg = 0
    train_auc_avg = 0
    val_auc_avg = 0
    train_rmse_avg = 0
    val_rmse_avg = 0
    train_recall_avg = [0,0,0,0,0,0,0,0,0,0]
    val_recall_avg = [0,0,0,0,0,0,0,0,0,0]
    train_aucf7_avg = [0, 0, 0, 0, 0, 0, 0]
    val_aucf7_avg = [0, 0, 0, 0, 0, 0, 0]
    val_ndcg_avg = [0,0,0,0,0]
    train_ndcg_avg = [0,0,0,0,0]



ts_step = 0
ts_size = FeatureStuSM.shape[0]
ts_loss_avg = 0.0
ts_accuracy_avg = 0.0
ts_auc_avg = 0.0
ts_rmse_avg = 0.0
ts_recall_avg = [0,0,0,0,0,0,0,0,0,0]
ts_auc7_avg = [0,0,0,0,0,0,0]
ts_ndcg_avg = [0,0,0,0,0]

while ts_step * batch_size < ts_size:
    bbiasICCI = biasesICCI[ts_step * batch_size:(ts_step + 1) * batch_size]
    bbiasICCSI = biasesICCSI[ts_step * batch_size:(ts_step + 1) * batch_size]
    bbiasCTC = biasesCTC[ts_step * batch_size:(ts_step + 1) * batch_size]
    bbiasCCC = biasesCCC[ts_step * batch_size:(ts_step + 1) * batch_size]
    bbiasSDS = biasesSDS[ts_step * batch_size:(ts_step + 1) * batch_size]
    bbiasadj_x4 = biasesadj_x4[ts_step * batch_size:(ts_step + 1) * batch_size]

    loss_value_ts, accuracy_value_ts, auc_value_ts, rmse_value_ts, recall_value_ts, auc7_value_ts, auc7_ndcg_ts = evaluate(model, inputSCon=inputSCon, inputSEmbRaw=inputSEmbRaw,
                                                inputCCon=inputCCon, inputCEmbRaw=inputCEmbRaw,
                                                bias_mat_adj_x4=bbiasadj_x4,
                                                bias_mat_ICCI=bbiasICCI,
                                                bias_mat_ICCSI=bbiasICCSI,
                                                bias_mat_CTC=bbiasCTC,
                                                bias_mat_CCC=bbiasCCC,
                                                training=False,
                                                test_index=test_index,
                                                S_CPopAll=S_CPopAlltest,
                                                tmpAll=tmpAlltest,
                                                lenSCnsNegInput=lenSCnsNegInputtest,
                                                ifndcg=1)
    loss_value_ts_mean = tf.reduce_mean(loss_value_ts)  
    ts_loss_avg += loss_value_ts_mean
    ts_accuracy_avg += accuracy_value_ts
    ts_auc_avg += auc_value_ts
    ts_rmse_avg += rmse_value_ts
    ts_recall_avg += recall_value_ts
    ts_auc7_avg += auc7_value_ts
    ts_ndcg_avg += auc7_ndcg_ts
    ts_step += 1

print('Test: loss = %.5f ' %(ts_loss_avg / ts_step))
print('Test: accuracy = %.5f ' %(ts_accuracy_avg / ts_step))
print('Test: auc = %.5f ' %(ts_auc_avg / ts_step))
print('Test: rmse = %.5f ' %(ts_rmse_avg / ts_step))
print('test-recall: 10-90,all:', ts_recall_avg)
print('test-ndcg: ', ts_ndcg_avg)
print('test-recall10: 1-6 & AUC:', ts_auc7_avg)
