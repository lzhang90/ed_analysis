import numpy as np
import os

data_dir = '\\..\\data\\mnist\\'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def mnist(ntrain=60000,ntest=10000,onehot=True):
	fd = open(os.path.dirname(__file__) + data_dir + 'train-images-idx3-ubyte')
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.dirname(__file__) + data_dir + 'train-labels-idx1-ubyte')
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.dirname(__file__) + data_dir + 't10k-images-idx3-ubyte')
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.dirname(__file__) + data_dir + 't10k-labels-idx1-ubyte')
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX,teX,trY,teY

def geotutor():
    f=open(os.path.dirname(__file__)+'\\..\\data\\geo tutor\\all_data.txt')
    firstline=True
    colnames=list()
    data=list()
    y=list()
    del_index=list()
    for line in f:
        if(firstline):
            y_index=index=0
            for col in line.split('	'):
                colnames.append(col)   
                if(col=='Outcome'):
                    y_index=index
                    del_index.append(index)               
                if(col in ['Row', 'Transaction Id', 'Session Id', 'Time', 'Problem Start Time','Duration (sec)', 'Sample Name']):
                    del_index.append(index)
                if(col in ['KC (Geometry)', 'KC Category (Geometry)', 'KC (Area)', 'KC Category (Area)','KC (Textbook)', 'KC Category (Textbook)','KC (Textbook New)', 'KC Category (Textbook New)', 'KC (Decompose)', 'KC Category (Decompose)',
                'KC (Textbook_New_Decompose)', 'KC Category (Textbook_New_Decompose)', 'KC (DecomposeArith)', 'KC Category (DecomposeArith)', 'KC (DecompArithDiam)', 'KC Category (DecompArithDiam)', 'KC (xDecmpTrapCheat)', 'KC Category (xDecmpTrapCheat)',
                'KC (LFASearchAICWholeModel2)', 'KC Category (LFASearchAICWholeModel2)', 'KC (textbook2)', 'KC Category (textbook2)', 'KC (LFASearchAICWholeModel3)', 'KC Category (LFASearchAICWholeModel3)', 'KC (LFASearchModel1-renamed&chgd)',
                'KC Category (LFASearchModel1-renamed&chgd)', 'KC (LFASearchModel1-backward)', 'KC Category (LFASearchModel1-backward)', 'KC (LFASearchModel1-backward)','KC Category (LFASearchModel1-backward)', 'KC (LFASearchModel1-renamed&chgd.2)',
                'KC Category (LFASearchModel1-renamed&chgd.2)',	'KC (LFASearchModel1-renamed&chgd.3)','KC Category (LFASearchModel1-renamed&chgd.3)','KC (LFASearchModel1.context-single)','KC Category (LFASearchModel1.context-single)',
                'KC (LFASearchModel1-context)','KC Category (LFASearchModel1-context)',	'KC (LFASearchModel1-context)',	'KC Category (LFASearchModel1-context)','KC (LFASearchModel1-back-context)','KC Category (LFASearchModel1-back-context)',
                'KC (LFASearchModel1-back-context)','KC Category (LFASearchModel1-back-context)','KC (LFASearchModel1-back-context)','KC Category (LFASearchModel1-back-context)','KC (LFASearchModel1-renamed)','KC Category (LFASearchModel1-renamed)',
                'KC (Single-KC)','KC Category (Single-KC)','KC (Unique-step)','KC Category (Unique-step)','KC (Decompose_height)','KC Category (Decompose_height)','KC (Circle-Collapse)','KC Category (Circle-Collapse)','KC (Orig-Trap-Merge)',
                'KC Category (Orig-Trap-Merge)','KC (Orig-trap-merge)','KC Category (Orig-trap-merge)','KC (Concepts)','KC Category (Concepts)','KC (new trap merge)','KC Category (new trap merge)','KC (Merge-Trap)','KC Category (Merge-Trap)',
                'KC (DecompArithDiam2)','KC Category (DecompArithDiam2)','KC (LFASearchAICWholeModel3arith)','KC Category (LFASearchAICWholeModel3arith)','KC (LFASearchAIC1_no_textbook_new_decompose)','KC Category (LFASearchAIC1_no_textbook_new_decompose)',
                'KC (LFASearchAIC2_no_textbook_new_decompose)',	'KC Category (LFASearchAIC2_no_textbook_new_decompose)','KC (LFASearchAIC1_with_texkbook_new_decompose)','KC Category (LFASearchAIC1_with_texkbook_new_decompose)',
                'KC (LFASearchAIC2_with_texkbook_new_decompose)','KC Category (LFASearchAIC2_with_texkbook_new_decompose)','KC (Item)','KC Category (Item)']):
                 #keep KC (Original)	KC Category (Original)   
                    del_index.append(index)
                index+=1
            count=0
            for i in del_index:
                del colnames[i-count]
                count+=1
            firstline=False
        else:
            row=line.split('	')
            y.append(row[y_index])
            count=0
            for i in del_index:
                del row[i-count]
                count+=1
            data.append(row)
    #colnames=np.array(colnames)
    #data=np.array(data)
    #y=np.array(y)
    return colnames, data, y

def simple_process(cols, data, targets): #transform each feature as true/false
    features=list()
    for i in range(len(cols)):
        s=set(row[i] for row in data)    
        while(len(s)>0):
            feature=cols[i]+':'+s.pop()
            features.append(feature)
    
    newdata=list()
    for row in data:
        newrow=[0]*len(features)
        for i in range(len(row)):
            index=features.index(cols[i]+':'+row[i])
            newrow[index]=1
        newdata.append(newrow)

    newy=list()
    y_cat=list(set(targets))
    for y in targets:
        newy.append(y_cat.index(y))

    features=np.array(features)
    newdata=np.array(newdata)
    newy=np.array(newy)
    newy = one_hot(newy, 2)
    return features,newdata,newy



class question:
    correct_attempts=0.
    incorrect_attempts=0.
    upper_correct=0.
    lower_correct=0.
    def __init__(self,qid):
        self.qid=qid
    def difficulty(self):
        return self.incorrect_attempts/(self.incorrect_attempts+self.correct_attempts)
    def discrimination(self):
        return (self.upper_correct-self.lower_correct)/(self.upper_correct+self.lower_correct)

class student:
    kc_prac_num=dict()
    def __init__(self,sid,all_kcs):
        self.sid=sid
        for kc in all_kcs:
            self.kc_prac_num[kc]=0
    def inc_exec(self,kc_name):
        self.kc_prac_num[kc_name]+=1
    def get_prac_num(self,kc_name):
        return self.kc_prac_num[kc_name]
        
def items_features(cols,data,targets):
    q_dict=dict()
    prob_index=0
    index=0
    for col in cols:
        if(col=='Problem Name'):
            prob_index=index
        index+=1
    rownum=0
    for row in data:
        if(not q_dict.has_key(row[prob_index])):
            q=question(row[prob_index]) 
            q_dict[row[prob_index]]=q
        q=q_dict[row[prob_index]]
        if(targets[rownum]=='CORRECT'):
            q.correct_attempts+=1
        else:
            q.incorrect_attempts+=1
        rownum+=1
    '''for q in q_dict.values():
        print q.qid, q.correct_attempts, q.incorrect_attempts'''
    return q_dict

def process(cols,data,targets):
    features=list()
    newlogs=list()
    s_dict=dict()
    index=0
    index_kc=index_stu=0
    all_kcs=list()
    for col in cols:
        if(col=='KC (Original)'):
            index_kc=index
        if(col=='Anon Student Id'):
            index_stu=index
        index+=1

    features=list()
    for i in range(len(cols)):
        s=set(row[i] for row in data)    
        while(len(s)>0):
            feature=cols[i]+':'+s.pop()
            features.append(feature)
    for kc in set(row[index_kc] for row in data):
        features.append("prac_num_"+kc)
        all_kcs.append(kc)
    

    for record in data:
        newrecord=list()
        newrecord=[0]*len(features)
        for i in range(len(record)):
            index=features.index(cols[i]+':'+record[i])
            newrecord[index]=1

        stuid=record[index_stu] #student id
        kcid=record[index_kc]
        if(not s_dict.has_key(stuid)):
            s_dict[stuid]=student(stuid,all_kcs)
        stu=s_dict[stuid]
        stu.inc_exec(kcid)
        for kc_prac_num in stu.kc_prac_num.values():
            newrecord.append(kc_prac_num)
        newlogs.append(newrecord)
    newlogs=normalize(newlogs)

    newy=list()
    y_cat=list(set(targets))
    for y in targets:
        newy.append(y_cat.index(y))        
    newy=np.array(newy)
    newy = one_hot(newy, 2)
    return features,newlogs,newy

def normalize(data):
    max_index=np.argmax(data,axis=0)
    max_values=list(data[max_index[col]][col] for col in range(len(max_index)))
    for i in range(len(data)):
        for j in range(len(max_index)):
            if(data[max_index[j]][j] >0):
                data[i][j]/=(max_values[j]*1.)
    return data
        
    

col, trX, trY=geotutor()
items_features(col, trX, trY)
col, trX, trY=process(col, trX, trY)