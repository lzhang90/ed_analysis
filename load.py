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

def geotutor(train_p=0.7):
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
                if(col=='Outcome'):
                    y_index=index
                else:
                    colnames.append(col)   
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
            del row[y_index] #the output
            count=0
            for i in del_index:
                del row[i-count]
                count+=1
            data.append(row)
    #colnames=np.array(colnames)
    #data=np.array(data)
    #y=np.array(y)
    return colnames, data, y

def process(cols, data, targets):
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
        

col, trX, trY=geotutor()
col, trX, trY=process(col, trX, trY)