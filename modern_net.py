import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import load as load
from sklearn import cross_validation
import matplotlib.pyplot as pl

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.0001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

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

X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((364, 59))
w_h2 = init_weights((59, 59))
w_o = init_weights((59, 2))

noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)

params = [w_h, w_h2, w_o]
#cost=T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))+ 0.01 * ((w_h ** 2).sum()+(w_h2 ** 2).sum()+(w_o ** 2).sum())
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y)) +0.001*(T.sum(w_h**2)+T.sum(w_h2**2)+T.sum(w_o**2))
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

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
w_h = init_weights((364, 59))
w_h2 = init_weights((59, 59))
w_o = init_weights((59, 2))
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
pl.savefig("modern_net.png")
pl.show()






