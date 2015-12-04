import theano
from theano import tensor as T
import numpy as np
import load as load
from sklearn import cross_validation
import matplotlib.pyplot as pl

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) *0.01))

def sgd(cost, params, lr=0.05):
    grads=T.grad(cost=cost, wrt=params)
    updates=[]
    for p, g in zip(params, grads):
        updates.append([p, p-g*lr])
    return updates
    
def model(X, w_h, w_o):
    h=T.nnet.sigmoid(T.dot(X, w_h))
    pyx=T.nnet.softmax(T.dot(h, w_o))
    return pyx

def train_with(num, test=True):
    for i in range(1000):
        for start, end in zip(range(0, num, 128), range(128, num, 128)):
            cost = train(trX[start:end], trY[start:end])
        #print i, np.mean(np.argmax(teY, axis=1) == predict(teX))
    if(test):
        acc=np.mean(np.argmax(teY, axis=1) == predict(teX))
    else:
        acc=np.mean(np.argmax(trY[:num], axis=1) == predict(trX[:num]))
    print num, acc
    return acc
    
col, data, targets=load.geotutor()
col, data, targets=load.process(col, data, targets)
trX,teX,trY,teY=cross_validation.train_test_split(data,targets,test_size=0.3,random_state=0)


X=T.fmatrix()
Y=T.fmatrix()

w_h=init_weights((364, 59))
w_o=init_weights((59, 2))

py_x=model(X,w_h, w_o)
y_pred=T.argmax(py_x, axis=1)

cost=T.mean(T.nnet.categorical_crossentropy(py_x, Y)) +0.001*(T.sum(w_h**2)+T.sum(w_o**2))
params=[w_h, w_o]
updates=sgd(cost, params)

train=theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict=theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

num=200
train_num=list()
accuracy=list()
while (num<len(trX)):
    train_num.append(num)
    acc=train_with(num)
    accuracy.append(acc)
    num+=200
pl.plot(train_num, accuracy)


num=200
train_num=list()
accuracy=list()
w_h=init_weights((364, 59))
w_o=init_weights((59, 2))
while (num<len(trX)):
    train_num.append(num)
    acc=train_with(num,test=False)
    accuracy.append(acc)
    num+=200
pl.plot(train_num, accuracy)

pl.xlabel("number of training examples")
pl.ylabel("accuracy")
pl.title("Accuracy changes with the number of training examples")
pl.grid(True)
pl.savefig("simple_net.png")
pl.show()