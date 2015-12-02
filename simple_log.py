import theano
from theano import tensor as T
import numpy as np
import load as load
from sklearn import cross_validation

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) *0.01))
    
def model(X, w):
    return T.nnet.softmax(T.dot(X,w))
    
col, data, targets=load.geotutor()
col, data, targets=load.process(col, data, targets)
trX,teX,trY,teY=cross_validation.train_test_split(data,targets,test_size=0.3,random_state=0)

X=T.fmatrix()
Y=T.fmatrix()

w=init_weights((364, 2))

py_x=model(X,w)
y_pred=T.argmax(py_x, axis=1)

cost=T.mean(T.nnet.categorical_crossentropy(py_x, Y))+0.001*T.sum(w**2)
gradient=T.grad(cost=cost, wrt=w)
update=[[w,w-gradient*0.5]]

train=theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict=theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(5000):
    cost=train(trX, trY)
    #for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
    #    cost=train(trX[start:end], trY[start:end])
    #print i, np.mean(np.argmax(teY, axis=1) == predict(teX))
    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))