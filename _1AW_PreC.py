import copy
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

AWLen = 5
WalkNum = 50
NeiNum = 7
AcceptLowSim = 0.3

path2014 = ''

def generate_anonym_walks(length):  
    anonymous_walks = []
    def generate_anonymous_walk(totlen, pre):  
        if len(pre) == totlen:
            anonymous_walks.append(pre)  
            return
        else:
            candidate = max(pre) + 1  
            for i in range(1, candidate+1):  
                if i != pre[-1]:  
                    npre = copy.deepcopy(pre)
                    npre.append(i)
                    generate_anonymous_walk(totlen, npre)  
    generate_anonymous_walk(length, [1])
    anonymous_walks.insert(0,[1 for x in range(0,length)])

    return anonymous_walks


def generate_walk2num_dict(length):  
    anonym_walks = generate_anonym_walks(length)
    
    anonym_dict = dict()
    curid = 0
    for walk in anonym_walks:
        swalk = intlist_to_str(walk)  
        anonym_dict[swalk] = curid
        curid += 1

    return anonym_dict  


def intlist_to_str(lst):
    slst = [str(x) for x in lst]
    strlst = "".join(slst)
    return strlst


def to_anonym_walk(walk):
    
    num_app = 0
    apped = dict()
    anonym = []
    for node in walk:
        if node not in apped:
            num_app += 1
            apped[node] = num_app
        anonym.append(apped[node])

    return anonym


class AW_PreC:
    def __init__(self, anonym_walk_len=5, num_paths=50, path_length=10, minSimStrPat=0.4):
        self.anonym_walk_len = anonym_walk_len
        self.num_paths = num_paths
        self.path_length = path_length
        self.cc_dict_dir = dict()
        self.cc_dict_no_dir = dict()
        self.cc_list = []
        self.random_walks = []
        self.random_walks_dict = dict()
        self.node_anony_walks_d = dict()
        self.minSimStrPat = minSimStrPat

        
        with open(path2014+"_BPrerequisiteCourses1-1", 'r', encoding="utf-8") as cc11:
            for cci in cc11:
                self.cc_list.append(cci.strip("\n").split(' '))
                if cci.strip("\n").split(' ')[0] not in self.cc_dict_dir:
                    self.cc_dict_dir[cci.strip("\n").split(' ')[0]] = []
                if cci.strip("\n").split(' ')[1] not in self.cc_dict_dir[cci.strip("\n").split(' ')[0]]:
                    self.cc_dict_dir[cci.strip("\n").split(' ')[0]].append(cci.strip("\n").split(' ')[1])
                if cci.strip("\n").split(' ')[0] not in self.cc_dict_no_dir:
                    self.cc_dict_no_dir[cci.strip("\n").split(' ')[0]] = []
                if cci.strip("\n").split(' ')[1] not in self.cc_dict_no_dir[cci.strip("\n").split(' ')[0]]:
                    self.cc_dict_no_dir[cci.strip("\n").split(' ')[0]].append(cci.strip("\n").split(' ')[1])
                if cci.strip("\n").split(' ')[1] not in self.cc_dict_no_dir:
                    self.cc_dict_no_dir[cci.strip("\n").split(' ')[1]] = []
                if cci.strip("\n").split(' ')[0] not in self.cc_dict_no_dir[cci.strip("\n").split(' ')[1]]:
                    self.cc_dict_no_dir[cci.strip("\n").split(' ')[1]].append(cci.strip("\n").split(' ')[0])

        del self.cc_dict_no_dir['noPC']
        self.num_nodes = len(self.cc_dict_no_dir)
        self.node_anonym_walktypes = np.zeros((self.num_nodes, self.num_paths))
        self.node_list = list(self.cc_dict_no_dir.keys())

        
        nodes = list(self.cc_dict_no_dir.keys())
        
        for _ in range(self.num_paths):  
            random.shuffle(nodes)  
            for node in nodes:
                if node not in self.random_walks_dict:
                    self.random_walks_dict[node] = []
                
                walk = self.node_walk_list(node, self.path_length)  
                self.random_walks.append(walk)
                self.random_walks_dict[node].append(walk)  
        for ws in self.random_walks_dict.keys():
            self.node_anony_walks_d[ws] = []
            for w in self.random_walks_dict[ws]:  
                self.node_anony_walks_d[ws].append(to_anonym_walk(w))

        self.node_normalized_walk_distr = self.process_anonym_distr(self.anonym_walk_len)  
        self.anonym_walk_dim = len(self.node_normalized_walk_distr[0])


    def process_anonym_distr(self, length):  

        self.anonym_walk_dict = generate_walk2num_dict(length)  

        node_anonym_distr = np.zeros((self.num_nodes, len(self.anonym_walk_dict)))
        for ni in range(len(self.node_list)):  
            for idxw in range(len(self.node_anony_walks_d[self.node_list[ni]])):  
                w = self.node_anony_walks_d[self.node_list[ni]][idxw]
                strw = intlist_to_str(w[:length])  
                wtype = self.anonym_walk_dict[strw]  
                self.node_anonym_walktypes[ni][idxw] = wtype  
                node_anonym_distr[ni][wtype] += 1  

        
        node_anonym_distr /=self.num_paths  
        self.graph_anonym_distr = np.mean(node_anonym_distr, axis = 0)
        graph_anonym_std = np.std(node_anonym_distr, axis = 0)
        graph_anonym_std[np.where(graph_anonym_std == 0)] = 0.001
        return (node_anonym_distr - self.graph_anonym_distr)/graph_anonym_std

    def node_walk_list(self, begin_node, path_length):
        
        
        walk = [begin_node]
        while (len(walk) < path_length):
            cur = walk[-1]
            cur_neighbors = self.cc_dict_no_dir[cur]
            cur_neighbors = sorted(cur_neighbors)
            if len(cur_neighbors):  
                if (len(cur_neighbors)==1) & (cur_neighbors[0] == 'noPC'):  
                    walk.append(cur)
                else:  
                    while 1:
                        nei = random.choice( cur_neighbors )
                        if nei != 'noPC':
                            walk.append(nei)
                            break
            else:
                break
        return walk

    def write(self, path):

        with open(path+'_DAWId', "w") as struPattEmb:
            for nd in range(len(self.node_list)):
                struPattEmb.write(self.node_list[nd] + '\n')
            struPattEmb.close()

        with open(path+'_EAWEmb', "w") as struPattId:
            for nd in range(len(self.node_list)):
                toWrite=''
                for cdemb in self.node_normalized_walk_distr[nd]:
                    toWrite = toWrite + str(cdemb) + ' '
                struPattId.write(toWrite[:-1] + '\n')
            struPattId.close()

    def cou_sim_list(self, path):  
        CCPre = []  
        s = cosine_similarity(self.node_normalized_walk_distr, self.node_normalized_walk_distr)
        with open(path + '_FStruSimVal', "w") as StrSimNei:
            for nd in range(len(self.node_list)):  
                toWrite = ''
                for simVnum in range(len(s[nd])):  
                    if s[nd][simVnum] >= self.minSimStrPat:  
                        CCPre.append([self.node_list[nd], self.node_list[simVnum], s[nd][simVnum]])
                    toWrite = toWrite + str(s[nd][simVnum]) + ' '
                StrSimNei.write(toWrite[:-1] + '\n')
            StrSimNei.close()

        f = open(path+'_GStruSimNei', "w+")
        for line in CCPre:
            f.write(line[0] + ' ' + line[1] + ':' + str(line[2]) + '\n')
        f.close()


if __name__ == '__main__':
    path = path2014
    aw_course_cc = AW_PreC(AWLen, WalkNum, NeiNum, AcceptLowSim)
    aw_course_cc.write(path)
    aw_course_cc.cou_sim_list(path)
