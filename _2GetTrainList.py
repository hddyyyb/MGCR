from collections import defaultdict
import random
path2014 = ''
path = ''
Dataset = ''  
dataset = Dataset
train_rate = 0.7
StuIdList = []
StuIdPosDict = defaultdict(int)

with open(path + dataset + '\\' + "_4StuId", 'r', encoding="utf-8") as fSId:
    i = 0
    for iSId in fSId:  
        StuIdList.append(iSId.strip("\n"))
        StuIdPosDict[iSId.strip("\n")] = i
        i = i + 1
    fSId.close()

train_num = int(train_rate * len(StuIdList))
SList_trainR = random.sample(StuIdList, train_num)
print('StuIdList: ', StuIdList)
print('SList_train: ', SList_trainR)

f = open(path2014 + "_HList_train", "w+")
for trainS in SList_trainR:
    f.write(trainS + '\n')  
f.close()

f = open(path2014 + "_JList_StuIdList", "w+")
for S in StuIdList:
    f.write(S + ',')  
f.close()

f = open(path2014 + "_IList_train_pb", "w+")
for trainS in SList_trainR:
    f.write(trainS + ',')  
f.close()
