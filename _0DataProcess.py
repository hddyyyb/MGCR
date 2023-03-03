import xlrd
import pymysql
from sklearn import preprocessing
import numpy as np
from gensim import models
import logging
import jieba
import re
import numpy.core.defchararray as np_f
from sklearn.decomposition import PCA
from collections import Counter
import scipy.sparse as sp
from collections import defaultdict
from scipy.stats import mode  


SqlCouData = []
SqlStuData = []  
SqlSCData = []  
path2014 = 'CPData\\Stu2014\\'
lowDim = 20  


def translateTea(CouIdTeaC):

    return CouIdTeaC


def get_stopwords():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
    
    stopword_set = set()
    with open("./toolData/chineseStopWords.txt",'r',encoding="GBK") as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip("\n"))
    return stopword_set


def excel_pc():
    
    PCdata = []
    PCdataRaw=[]
    PrerequisiteCourses = xlrd.open_workbook(r'RawData\CourseRelation.xls')  
    
    
    
    sheet1 = PrerequisiteCourses.sheets()[0]
    print('PrerequisiteCourses '+sheet1.name,sheet1.nrows,sheet1.ncols)  
    
    row =sheet1.nrows
    for i in range(row):
        rowdate = sheet1.row_values(i)  
        if type(rowdate[0]) == float:
            PCdataRaw.append([str(int(rowdate[0])), rowdate[-1]])
        else:
            PCdataRaw.append([rowdate[0], rowdate[-1]])
        for j in re.split('[,;]+', rowdate[-1]):  
            if j == rowdate[0]:
                continue
            if type(rowdate[0]) == float:
                
                if type(j) == float:
                    PCdata.append([str(int(rowdate[0])), str(int(j))])
                else:
                    PCdata.append([str(int(rowdate[0])), j])
            else:
                if type(j) == float:
                    PCdata.append([rowdate[0], str(int(j))])
                else:
                    PCdata.append([rowdate[0], j])
    fR = open(path2014+"_CPrerequisiteCourses1-n", "w+")
    for line in PCdataRaw:
        fR.write(line[0] + ' ' + line[1] + '\n')  
    fR.close()

    f = open(path2014+"_BPrerequisiteCourses1-1", "w+")
    for line in PCdata:
        f.write(line[0] + ' ' + line[1] + '\n')  
    f.close()

    return PCdataRaw, PCdata


def excel_dor():
    
    dordata = []
    Gendata = []
    Homdata = []
    StuGen = dict()
    StuHom = dict()
    StuDor = dict()
    Gen = dict()
    JG = dict()
    Gen['女']=0
    Gen['男'] =1
    Gen['性别'] = -1
    JG['湖南省'] =0
    JG['吉林省'] = 1
    JG['辽宁省'] = 2
    JG['江西省'] = 3
    JG['江苏省'] = 4
    JG['湖北省'] = 5
    JG['黑龙江省'] = 6
    JG['河南省'] = 7
    JG['河北省'] = 8
    JG['海南省'] = 9
    JG['贵州省'] = 10
    JG['广西自治区'] = 11
    JG['广东省'] = 12
    JG['甘肃省'] = 13
    JG['福建省'] = 14
    JG['安徽省'] = 15
    JG['内蒙古自治区'] = 16
    JG['宁夏自治区'] = 17
    JG['青海省'] = 18
    JG['山东省'] = 19
    JG['山西省'] = 20
    JG['陕西省'] = 21
    JG['四川省'] = 22
    JG['天津市'] = 23
    JG['云南省'] = 24
    JG['新疆自治区'] = 25
    JG['浙江省'] = 26
    JG['重庆市'] = 27
    JG['籍贯'] = -1

    StuDorRaw = xlrd.open_workbook(r'RawData\dormitory2014-16.xls')  
    sheet1 = StuDorRaw.sheets()[0]
    print('StuDormitory ' + sheet1.name, sheet1.nrows, sheet1.ncols)  
    row = sheet1.nrows
    for i in range(row):
        rowdate = sheet1.row_values(i)
        if rowdate[0]:
            StuGen[rowdate[0]] = Gen[rowdate[1].strip()]
            StuHom[rowdate[0]] = JG[rowdate[2].strip()]
            StuDor[rowdate[0]] = [rowdate[3].strip(), rowdate[4]]
            dordata.append([rowdate[0], rowdate[3], rowdate[4]])
            Gendata.append([rowdate[0], rowdate[1]])
            Homdata.append([rowdate[0], rowdate[2]])

    fdor = open(path2014+"StuDormitory.txt", "w")
    for line in dordata:
        fdor.write(str(line[0]) + ' ' + line[1] + ' ' + str(line[2]) + '\n')
    fdor.close()

    fGen = open(path2014+"StuGender.txt", "w")
    for line in Gendata:
        fGen.write(str(line[0]) + ' ' + line[1] + '\n')  
    fGen.close()

    fHom = open(path2014+"StuHome.txt", "w")
    for line in Homdata:
        fHom.write(str(line[0]) + '  ' + line[1] + '\n')  
    fHom.close()
    print(f'StuGen: {StuGen}')
    return StuDor, StuGen, StuHom  


def sql_scsc(StuGen, StuHom, StuDor):

    conn = pymysql.connect(host='localhost', user="root", passwd="53377432", db="CoursePre")
    
    cursor = conn.cursor()
    
    sqlc = "SELECT course_id,course_code,course_name," \
           "teacher,credit,if_elective,ct,cf FROM course;"
    temp_num=0
    maxEC = 0
    CouFor1hotFea = []  
    CouForValFea0 = []  
    CouForValFea1 = []  
    CouForValFea2ele = []  
    CouForValFea2nele = []  

    CouForValFeaSeme = []  

    CouCredit = []
    CouId = []  
    CouIdDict = defaultdict(int)
    CouCode = []  
    CouNam = []  
    CouIdCode = []
    CouIdTeaRaw = []
    CouIdTeaC = []
    TeaList = []
    cursor.execute(sqlc)  
    results = cursor.fetchall()  
    iCouIdDict = 0
    for row in results:
        course_id = row[0]
        course_code = row[1]
        course_name = row[2]
        teacher = row[3]
        credit = row[4]
        if_elective = ord(row[5])  
        ct = row[6]
        cf = row[7]
        temp_num = temp_num+1
        if teacher == '':
            teacher = '无名无名'
        if if_elective:  
            maxEC = max(maxEC, credit)
        CouIdCode.append([course_id, course_code])  
        if teacher != '无名无名':
            teacher = teacher.rstrip(',')
            CouIdTeaRaw.append([course_id, teacher])
            for itea in teacher.split(','):
                CouIdTeaC.append([course_id, itea])
        SqlCouData.append([course_id, course_code, course_name, ct, teacher, cf, credit, if_elective])
        CouFor1hotFea.append([ct, cf])
        CouCredit.append(credit)
        CouForValFea0.append(credit)
        CouForValFea1.append(if_elective)
        CouForValFea2ele.append(0)  
        CouForValFea2nele.append(0)  
        CouForValFeaSeme.append([float('inf')])  
        if course_code not in CouCode:
            CouCode.append(course_code)
        CouId.append(course_id)
        CouIdDict[course_id] = iCouIdDict
        iCouIdDict = iCouIdDict+1
        CouNam.append(course_name)
    CouForValFea = [maxEC if i > maxEC else i for i in CouForValFea0]  
    CouCredit = [maxEC if i > maxEC else i for i in CouForValFea0]
    CouForValFea1 = np.array(CouForValFea1)
    aCouForValFea = np.array(CouForValFea)
    
    CouCredit = np.array(CouCredit)  
    
    aCouForValFea = np.expand_dims(aCouForValFea, axis=1)  
    CouForValFea1 = np.expand_dims(CouForValFea1, axis=1)  

    CouIdTea = translateTea(CouIdTeaC)
    for _,t in CouIdTea:
        if t not in TeaList:
            TeaList.append(t)
    print(f'total course number: {temp_num}')

    
    StuForValFea = []  
    StuFor1hotFea = []  
    StuId = []  
    StuIdDict = defaultdict(int)
    StuMajorNam = []  
    SDRel = []
    sqls = "SELECT student_no, class, major, department, ele_no, nele_no FROM student where Normal_stu=1 and student_no<50000;"
    temp_num = 0
    cursor.execute(sqls)
    results = cursor.fetchall()
    iStuIdDict = 0
    for row in results:
        student_id = row[0]
        class_s = row[1]
        major = row[2]
        department = row[3]
        ele_no = row[4]
        nele_no = row[5]
        if student_id in StuGen and student_id in StuHom and student_id in StuDor:
        
            StuForValFea.append([ele_no, nele_no])
            StuFor1hotFea.append([class_s, StuGen[student_id], StuHom[student_id]])
            SqlStuData.append([student_id, class_s, major, department, ele_no, nele_no, StuGen[student_id], StuHom[student_id]])
            StuId.append(student_id)
            StuIdDict[student_id] = iStuIdDict
            iStuIdDict = iStuIdDict+1
            StuMajorNam.append(major)
            temp_num = temp_num + 1
            SDRel.append([student_id, StuDor[student_id][0].replace("一舍", "1").replace("五舍", "5")+str(int(StuDor[student_id][1]))])
        else:
            print(f'student_id: {student_id}')
    print(f'total student number: {temp_num}')
    StudentCourseGrade = [[0 for col in range(len(CouId))] for row in range(len(StuId))]
    CourseSemeChoose = [[] for i in range(len(CouId))]
    sqls = "SELECT student_no, course_id, grade, gpa, seme, Normal_stu, course_code, if_e FROM sc where student_no<50000;"
    temp_num = 0
    cursor.execute(sqls)
    results = cursor.fetchall()
    print('len11111111111111111111111111')
    print(len(results))
    for row in results:
        student_no = row[0]
        course_id = row[1]
        grade = row[2]
        gpa = row[3]
        seme = row[4]
        normal_stu = ord(row[5])
        course_code = row[6]
        if_elective = ord(row[7])
        if int(seme)>=0 and int(seme)<=5:
            if normal_stu == 1 and student_no in StuId:
                SqlSCData.append([student_no, course_id, grade, gpa, seme, course_code])
                temp_num = temp_num + 1
                
                StudentCourseGrade[StuIdDict[student_no]][CouIdDict[course_id]] = grade
                CourseSemeChoose[CouIdDict[course_id]].append(seme)
                if if_elective == 1:
                    CouForValFea2ele[CouIdDict[course_id]] = CouForValFea2ele[CouIdDict[course_id]] + 1
                else:
                    CouForValFea2nele[CouIdDict[course_id]] = CouForValFea2nele[CouIdDict[course_id]]+1
        else:
            print(f'seme: {seme}')

    CouForValFea2ele = np.expand_dims(CouForValFea2ele, axis=1)  
    CouForValFea2nele = np.expand_dims(CouForValFea2nele, axis=1)  
    print('len(CourseSemeChoose)')
    print(len(CourseSemeChoose))
    for i in range(len(CourseSemeChoose)):
        if CourseSemeChoose[i]:
            CouForValFeaSeme[i], _ = mode(np.array(CourseSemeChoose[i]))  
        else:
            print(f'CourseSemeChoose[i]: {CourseSemeChoose[i]}, CouId[i]:{CouId[i]},i:{i}')
    CouForValFeaSeme = np.array(CouForValFeaSeme)
    StuForValFea2 = np.matmul(StudentCourseGrade, CouCredit)
    StuForValFea2 = np.expand_dims(StuForValFea2, axis=1)  
    print(f'total sc number: {temp_num}')
    cursor.close()  

    
    fCou = open(path2014+"CourseData.txt", "w")
    for i in SqlCouData:
        
        fCou.write('' + i[0] + ' ' + i[1] + ' ' +i[2] + ' ' +str(i[3]) + ' ' +i[4] + ' ' +str(i[5]) + ' ' +str(i[6]) + ' ' + str(i[7]) + '\n')  
    fCou.close()
    fStu = open(path2014+"StudentData.txt", "w")
    for i in SqlStuData:  
        fStu.write(str(i[0]) + ' ' + str(i[1]) + ' ' + i[2] + ' ' + i[3] + ' ' + str(i[4]) + ' ' + str(i[5]) + ' ' + str(i[6]) + ' ' + str(i[7]) +'\n')
    fStu.close()
    fSC = open(path2014+"StuCouData.txt", "w")
    for i in SqlSCData:  
        fSC.write(str(i[0]) + ' ' + i[1] + ' ' + str(i[2]) + ' ' + str(i[3]) + ' ' + str(i[4]) + ' ' + i[5] + '\n')
    fSC.close()

    DorList = []
    for s, d in SDRel:
        if d not in DorList:
            DorList.append(d)
    print(CouForValFeaSeme.shape)

    return aCouForValFea, CouForValFea1, CouForValFea2ele, CouForValFea2nele, CouForValFeaSeme, CouFor1hotFea, np.array(CouId), np.array(CouCode), np.array(CouNam), StuForValFea, StuForValFea2, StuFor1hotFea, np.array(StuId), np.array(StuMajorNam), SDRel, CouIdCode, CouIdTeaRaw, CouIdTea, TeaList, DorList


def get_cou_feature(CouValFea, CouFor1hotFea):
    CouFor1hotFea = np.array(CouFor1hotFea)
    print(f'CouFor1hotFea.shape: {CouFor1hotFea.shape}')
    print(f'CouValFea.shape:{CouValFea.shape}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    CouFea0 = min_max_scaler.fit_transform(CouFor1hotFea)  
    print('cou.couvalfea.shape:', CouFea0.shape)
    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    print('000')
    print(CouValFea)
    couvalfea = min_max_scaler.fit_transform(CouValFea)  
    print('cou.couvalfea.shape:', couvalfea.shape)
    

    return CouFea0, couvalfea


def get_stu_feature(StuForValFea, StuFor1hotFea):
    
    
    
    
    

    aStuForValFea = np.array(StuForValFea)
    min_max_scaler = preprocessing.MinMaxScaler()
    stuvalfea = min_max_scaler.fit_transform(aStuForValFea)
    print('stu.valfea.shape:', stuvalfea.shape)

    StuFea = np.array(StuFor1hotFea)
    min_max_scaler = preprocessing.MinMaxScaler()
    stufea = min_max_scaler.fit_transform(StuFea)
    print('stu.valfea.shape:', stufea.shape)

    return stufea, stuvalfea



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    
    
    rowsum = np.array(features.sum(1))  
    
    
    r_inv = np.power(rowsum, -1).flatten()
    
    
    
    
    
    
    
    
    
    r_inv[np.isinf(r_inv)] = 0.  
    r_mat_inv = sp.diags(r_inv)
    
    
    
    
    
    
    
    
    
    
    features = r_mat_inv.dot(features)  
    
    
    
    
    
    
    
    
    
    
    
    return features
    


def get_cou_nemb(CouNamArray0):  
    
    

    NameWord = []
    NameWEmbHdim = []
    WordNEmb = dict()
    NameWEmb = []  
    WordList = []  
    HdimList = []  
    NotInVoc = []
    CouNamArray1 = np_f.replace(CouNamArray0, '㈠', '（一）')
    CouNamArray2 = np_f.replace(CouNamArray1, '㈡', '（二）')
    CouNamArray3 = np_f.replace(CouNamArray2, '㈢', '（三）')
    CouNamArray4 = np_f.replace(CouNamArray3, '㈣', '（四）')
    CouNamArray5 = np_f.replace(CouNamArray4, '㈤', '（五）')
    CouNamArray6 = np_f.replace(CouNamArray5, '㈥', '（六）')
    CouNamArray7 = np_f.replace(CouNamArray6, '1', '一')
    CouNamArray = np_f.replace(CouNamArray7, '2', '二')
    CouNamArray = np.array(CouNamArray)
    
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
    model = models.Word2Vec.load("W2vWikiModel/wiki_corpus.model")
    
    stopwords = get_stopwords()
    MaxWordLen = 0

    
    for coui in CouNamArray:  
        couiNW = []  
        couiNW.append(coui)
        couiNWE = []  
        couNotInVoc = []  
        couNotInVoc.append(coui)
        namewords = jieba.lcut(coui, cut_all=True)  
        for nm in namewords:
            if nm not in stopwords:  
                try:
                    nm_emb = model.wv.get_vector(nm)  
                    couiNW.append(nm)
                    couiNWE.append(nm_emb)
                    if nm not in WordList:
                        WordList.append(nm)
                        HdimList.append(nm_emb)
                except KeyError:
                    couNotInVoc.append(nm)
        MaxWordLen = max(MaxWordLen, len(couiNW))
        NameWord.append(couiNW)
        NameWEmbHdim.append(couiNWE)
        NotInVoc.append(couNotInVoc)

    
    pca = PCA(n_components = lowDim)
    LdimList = pca.fit_transform(HdimList)  
    invNameWEmb = pca.inverse_transform(LdimList)  
    print('----------pca.explained_variance_ratio_-------')
    print(pca.explained_variance_ratio_)
    sum = 0
    for i in pca.explained_variance_ratio_:
        sum = sum + i
    print(f'pca降维特征表示信息的总和: {sum}')
    for i in range(len(WordList)):
        WordNEmb[WordList[i]] = LdimList[i]

    
    for i in NameWord:
        iNWE = []  
        for ii in i[1:]:  
            iNWE.append(WordNEmb[ii])
        while len(iNWE) < MaxWordLen:
            iNWE.append([-1] * lowDim)  
        NameWEmb.append(iNWE)

    
    print('max word number in course name:', str(MaxWordLen))  
    CNWoutput = open(path2014+"CouNameWord.txt", "w+")  
    for i in NameWord:
        
        CNWoutput.write(i[0] + ':' + ' '.join(i[1:]) + '\n')
    CNWoutput.close()

    CNotInVoc = open(path2014+"CouNotInVoc.txt", "w+")  
    for i in NotInVoc:
        CNotInVoc.write(i[0] + ':' + ' '.join(i[1:]) + '\n')
    CNotInVoc.close()

    CNEmboutput = open(path2014+"CouNameEmbhigh.txt", "w+")  
    for i in NameWEmbHdim:
        writeii = ''
        for ii in i:  
            writeiii = ''
            for iii in ii:  
                writeiii = writeiii + str(iii) + ' '
            writeii = writeii + writeiii[:-1] + ','
        CNEmboutput.write(writeii[:-1]+'\n')
    CNEmboutput.close()

    return NameWord, NameWEmb, WordNEmb  


def get_stu_memb(StuMajorNam, WordNEmb):  
    
    
    MajEmb = []  
    MajWord = []
    NotInVoc = []
    StuMajorNam = np.array(StuMajorNam)
    stopwords = get_stopwords()
    lencw = 0

    for smajori in StuMajorNam:  
        stuMNotInVoc = []
        stuMNotInVoc.append(smajori)
        lencwi = 0
        stuimW = []
        stuimW.append(smajori)
        stuimWE = []
        namewords = jieba.lcut(smajori, cut_all=True)
        for nm in namewords:
            if nm not in stopwords:
                try:
                    stuimWE.append(WordNEmb[nm])
                    stuimW.append(nm)
                    lencwi += 1
                except KeyError:
                    stuMNotInVoc.append(nm)
        MajEmb.append(stuimWE)
        MajWord.append(stuimW)
        lencw = max(lencw, lencwi)
        NotInVoc.append(stuMNotInVoc)

    print('max word number in student major:',str(lencw))
    SMWoutput = open(path2014+"StuMajorWord.txt", "w+")
    for i in MajWord:
        SMWoutput.write(i[0] + ':' + ' '.join(i[1:]) + '\n')
    SMWoutput.close()

    SMNotInVoc = open(path2014+"StuMajNotInVoc.txt", "w+")  
    for i in NotInVoc:
        SMNotInVoc.write(i[0] + ':' + ' '.join(i[1:]) + '\n')
    SMNotInVoc.close()

    return MajWord, MajEmb

def get_sc_rel():
    scRel=[]
    
    with open(path2014+'_0StuCouRel', "w") as scRoutfile:
        for sci in SqlSCData:
            scRel.append([sci[0], sci[1], sci[2], sci[4]])  
            scRoutfile.write(str(sci[0]) + '-' + sci[1] + ':' + str(sci[2]) + ',' + str(sci[4]) + '\n')
        scRoutfile.close()
    return scRel  


if __name__ == '__main__':

    PCdataRaw, PCdata = excel_pc()  
    
    StuDor, StuGen, StuHom = excel_dor()  
    aCouForValFea, CouForValFea1, CouForValFea2ele, CouForValFea2nele, CouForValFeaSeme, CouFor1hotFea, CouId, CouCode, CouNamArray, StuForValFea, StuForValFea2, StuFor1hotFea, StuId, StuMajorNam, sdRel, CouIdCode, CouIdTeaRaw, CouIdTea, TeaList, DorList = sql_scsc(StuGen, StuHom, StuDor)
    
    
    
    
    Cou1hotFea, CouValFea = get_cou_feature(np.hstack((aCouForValFea, CouForValFea1, CouForValFea2ele, CouForValFea2nele, CouForValFeaSeme)), CouFor1hotFea)  

    NameWord, NameWEmb, WordNEmb = get_cou_nemb(CouNamArray)  
    CouFea = np.hstack((Cou1hotFea, CouValFea))
    
    Stu1hotFea, StuValFea = get_stu_feature(np.hstack((StuForValFea, StuForValFea2)), StuFor1hotFea)  
    MajWord, MajEmb = get_stu_memb(StuMajorNam, WordNEmb)  
    StuFea = np.hstack((Stu1hotFea, StuValFea))


    
    
    
    
    
    scRel = get_sc_rel()

    
    with open(path2014+'_1FeatureStu', "w") as StuFeafile:
        for sf in StuFea:
            write = ''
            for sfi in sf:
                write = write + str(sfi) + ' '
            StuFeafile.write(write[0:-1] + '\n')
        StuFeafile.close()

    with open(path2014+'_2FeatureCou', "w") as CouFeafile:
        for cf in CouFea:
            write = ''
            for cfi in cf:
                write = write + str(cfi) + ' '
            CouFeafile.write(write[0:-1] + '\n')
        CouFeafile.close()

    with open(path2014+'_3CouId', "w") as CouIdfile:
        for ci in CouId:
            CouIdfile.write(ci + '\n')
        CouIdfile.close()

    with open(path2014+'_4StuId', "w") as StuIdfile:
        for si in StuId:
            StuIdfile.write(str(si) + '\n')
        StuIdfile.close()

    with open(path2014+'_5CouCode', "w") as CouCodefile:
        for cc in CouCode:
            CouCodefile.write(str(cc) + '\n')
        CouCodefile.close()

    
    fct = open(path2014+'_6CouIdTeaRel.txt', "w", encoding = 'GBK')  
    for i in CouIdTea:  
        fct.write(str(i[0]) + ' ' + i[1] + '\n')
    fct.close()

    ft = open(path2014+'_6ZTeaList.txt', "w", encoding = 'GBK')  
    for i in TeaList:  
        ft.write(str(i) + '\n')
    ft.close()

    fCouIdCode = open(path2014+'_7CouIdCodeRel', "w")
    for i in CouIdCode:  
        fCouIdCode.write(i[0] + ' ' + i[1] + '\n')
    fCouIdCode.close()

    fsdRel = open(path2014+'_8StuDorRel', "w+")
    for i in sdRel:
        fsdRel.write(str(i[0]) + ' ' + i[1] + '\n')
    fsdRel.close()

    fd = open(path2014+'_8ZDorList', "w+")
    for i in DorList:
        fd.write(str(i) + '\n')
    fd.close()

    
    CNEmboutput = open(path2014+'_9CouNameEmb', "w+")  
    for i in NameWEmb:
        writeii = ''
        for ii in i:  
            writeiii = ''
            for iii in ii:  
                writeiii = writeiii + str(iii) + ' '
            
            writeii = writeii + writeiii[:-1] + ' '  
        CNEmboutput.write(writeii[:-1] + '\n')
    CNEmboutput.close()

    SMEmboutput = open(path2014+'_AStuMajorEmb', "w+")
    for i in MajEmb:
        writeii = ''
        for ii in i:
            writeiii = ''
            for iii in ii:
                writeiii = writeiii + str(iii) + ' '
            
            writeii = writeii + writeiii[:-1] + ' '
        SMEmboutput.write(writeii[:-1] + '\n')
    SMEmboutput.close()
