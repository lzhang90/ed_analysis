import theano
from theano import tensor as T
import numpy as np
import load as load
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as pl

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) *0.01))
    
def model(X, w):
    return T.nnet.softmax(T.dot(X,w))

def train_with(num, test=True):
    for i in range(10000):
        for start, end in zip(range(0, num, 128), range(128, num, 128)):
            cost = train(trX[start:end], trY[start:end])
        print i, np.mean(np.argmax(trY, axis=1) == predict(trX))
    trueY=teY
    predY=predict(teX)
    test_acc=np.mean(np.argmax(teY, axis=1) == predict(teX))

    trueY=trY[:num]
    predY=predict(trX[:num])
    train_acc=np.mean(np.argmax(trY[:num], axis=1) == predict(trX[:num]))

    #print num, precision_recall_fscore_support(list(y[1] for y in trueY.tolist()),predY.tolist(), average='micro')
    print num, train_acc,test_acc
    return train_acc,test_acc

def data_split(data,targets):
    trainX=data[:4908]
    trainY=targets[:4908]
    testX=data[4908:]
    testY=targets[4908:]
    return trainX,testX,trainY,testY
    
col, data, targets=load.geotutor()
col, data, targets=load.process(col, data, targets)
#trX,teX,trY,teY=cross_validation.train_test_split(data,targets,test_size=0.3,random_state=0)
trX,teX,trY,teY=data_split(data,targets)

X=T.fmatrix()
Y=T.fmatrix()

w=init_weights((407, 2))

py_x=model(X,w)
y_pred=T.argmax(py_x, axis=1)

cost=T.mean(T.nnet.categorical_crossentropy(py_x, Y))+0.001*T.sum(w**2)
gradient=T.grad(cost=cost, wrt=w)
update=[[w,w-gradient*0.005]]

train=theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict=theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

'''
num=200
train_num=list()
tr_accs=list()
te_accs=list()
while (num<len(trX)):
    train_num.append(num)
    tr_acc,te_acc=train_with(num)
    tr_accs.append(tr_acc)
    te_accs.append(te_acc)
    num+=200
pl.plot(train_num, tr_accs)
pl.plot(train_num, te_accs)

pl.xlabel("number of training examples")
pl.ylabel("accuracy")
pl.title("Accuracy changes with the number of training examples")
pl.grid(True)
pl.savefig("simple_log.png")
pl.show()
'''
tr_acc,te_acc=train_with(len(trX))