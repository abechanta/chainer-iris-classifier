import numpy as np
import chainer
from chainer import Chain, Variable, optimizers, cuda
import chainer.functions as F
import chainer.links as L
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import load_iris
import time

def load_data():
    body = load_iris()
    ss = StratifiedShuffleSplit(test_size=0.2)
    train, test = next(ss.split(body.data, body.target))
    return (
        body.data[train], body.data[test],
        body.target[train], body.target[test],
    )

class MyIrisClassifier(Chain):
    def __init__(self):
        super(MyIrisClassifier, self).__init__(
            l1=L.Linear(4, 6),
            l2=L.Linear(6, 3),
        )

    def __call__(self, X, y):
        y_raw = self.forward(X)
        return F.softmax_cross_entropy(y_raw, y)

    def forward(self, X):
        h1 = F.sigmoid(self.l1(X))
        h2 = self.l2(h1)
        return h2

    def predict(self, X):
        y_raw = self.forward(X)
        y_pred = F.argmax(F.softmax(y_raw).data, axis=1)
        return y_pred.data

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    model = MyIrisClassifier()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    start_time = time.time()

    for epoch in range(5001):
        model.to_gpu()
        model.zerograds()
        model.to_gpu()
        Xv = Variable(cuda.to_gpu(X_train.astype(np.float32)))
        yv = Variable(cuda.to_gpu(y_train.astype(np.int32)))
        loss = model(Xv, yv)
        loss.backward()
        optimizer.update()

        if epoch % 100 == 0:
            y_pred = model.predict(X_test.astype(np.float32))
            score = np.mean(y_pred == y_test)
            print("epoch={:<5d} accuracy={:.5f} loss={}".format(epoch, score, loss.data))

    elapsed_time = time.time() - start_time
    print("elapsed time={:.2f}sec".format(elapsed_time))
