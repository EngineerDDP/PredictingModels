import numpy as np


import shutil as shell
import os

from predict.model.interfaces import IModel
from datetime import datetime


class Linear:
    """
    Linear activation
    """

    def __init__(self):
        pass

    def activation(self, x):
        return x

    def gradient(self, x):
        return 1


class ReLU:
    """
    ReLU activation
    """
    def __init__(self):
        pass

    def activation(self, x):
        r = x.copy()
        r[r < 0] = 0
        return r

    def gradient(self, x):
        r = x.copy()
        r[r < 0] = 0
        r[r > 0] = 1
        return r


class Sigmoid:
    """
    Sigmoid type activation
    """

    def __init__(self, delta=0.0):
        self.Delta = delta

    def activation(self, x):
        return 1 / (1 + np.exp(-1 * (x + self.Delta)))

    def gradient(self, x):
        return np.multiply(self.activation(x), (1 - self.activation(x)))


class Tanh:
    """
    Hyperbolic tangent function
    """

    def __init__(self, **kwargs):
        pass

    def activation(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.multiply(self.activation(x), self.activation(x))


class FCLayer:

    def __init__(self, units, w_init=None, b_init=None, act=Linear()):

        # use lazy initialization
        if w_init is not None:
            self.W = w_init
        else:
            self.W = None

        if b_init is not None:
            self.B = b_init
        else:
            self.B = None

        self.Act = act
        self.Output = units

    def reset(self):
        self.W = None
        self.B = None

    def logit(self, x):
        """
            Calculate logit
        """
        # lazy initialization
        # w shape is [output, input]
        # b shape is [output]
        if self.W is None:
            high = np.sqrt(1 / x.shape[0])
            low = -high
            self.W = np.random.uniform(low=low, high=high, size=[self.Output, x.shape[0]])
        if self.B is None:
            self.B = np.zeros(shape=[self.Output, 1])

        return np.dot(self.W, x) + self.B

    def F(self, x):
        """
            output function
        """
        # activation
        return self.Act.activation(self.logit(x))

    def backpropagation(self, x, gradient):
        """
            Calculate gradient, adjust weight and bias and return gradients of this layer
            x shape=[input, samples count]
            grad shape=[output, samples count]
        """
        # calculate gradient
        act_grad = self.Act.gradient(self.logit(x))
        # y shape=[output, samples count]
        y = np.multiply(act_grad, gradient)

        # adjust weight
        batch_weight = y.dot(x.T) / y.shape[1]
        self.W = self.W - batch_weight
        # adjust bias
        self.B = self.B - y.mean(axis=1)
        # recalculate gradient to propagate
        grad = self.W.transpose().dot(y)
        return grad


class GradientDecentOptimizer:

    def __init__(self, loss, layers, learnrate=0.01):
        self.LR = learnrate
        self.Loss = loss
        self.Layers = layers

    def train(self, x, label):
        """
            train the network with labeled samples
        """

        # reshape x to [-1,1]

        x = np.asmatrix(x).T
        label = np.asmatrix(label).T

        # forward propagation

        intermediate = [x]
        for nn in self.Layers:
            intermediate.append(nn.F(intermediate[-1]))

        loss = self.Loss.loss(intermediate[-1], label)

        # apply learning rate

        self.Grad = self.LR * self.Loss.gradient(intermediate[-1], label)
        grad = self.Grad

        # backward propagation

        self.Layers.reverse()
        i = 2
        for nn in self.Layers:
            grad = nn.backpropagation(intermediate[-1 * i], grad)
            i += 1

        self.Layers.reverse()

        # return loss

        return np.mean(loss)


class AdagradOptimizer(GradientDecentOptimizer):

    def __init__(self, loss, layers, learnrate=0.01):
        super().__init__(loss, layers, learnrate)
        self.Gt = 0
        self.delta = 1e-8

    def train(self, x, label):
        # update learning rate
        learn_rate = self.LR
        if self.Gt != 0 and self.LR > 0.001:
            self.LR = self.LR / np.sqrt(self.Gt + self.delta)

        # train
        loss = super().train(x, label)
        # print(self.LR)

        # update Gt
        self.Gt = self.Gt + np.mean(np.square(self.Grad))
        self.LR = learn_rate

        return loss


#----------------------------------------------------------
#Build your own loss functions below

class ILoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        raise NotImplementedError()

    def gradient(self, y, label):
        raise NotImplementedError()


class MseLoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        result = np.mean(np.square(label - y), axis=0)
        return result

    def gradient(self, y, label):
        return (label - y) * -2


class CrossEntropyLoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        # multiply element-wise
        return np.mean(np.multiply(label, np.log(y)) + np.multiply((1 - label), np.log(1 - y))) * -1

    def gradient(self, y, label):
        return ((1 - label) / (1 - y) - label / y) / label.shape[1]


class CrossEntropyLossWithSigmoid:

    def __init__(self):
        pass

    def loss(self, y, label):
        # multiply element-wise
        return np.mean(np.multiply(label, np.log(y)) + np.multiply((1 - label), np.log(1 - y))) * -1

    def gradient(self, y, label):
        return np.multiply(y - label, 1 / np.multiply(y, 1 - y))


class TanhLoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        return np.mean(np.square(np.tanh(label - y)))

    def gradient(self, y, label):
        return -2.0 * np.multiply(np.tanh(label - y), (1 - np.square(np.tanh(label - y))))


class L1Loss:

    def __init__(self):
        self.L2Loss = MseLoss()

    def loss(self, y, label):
        return np.mean(label - y)

    def gradient(self, y, label):

        grad = label - y
        grad[np.where(grad < 0)] = -1
        grad[np.where(grad > 0)] = 1
        grad = grad * -1 + self.L2Loss.gradient(y, label)

        return grad


#----------------------------------------------------------


class Model:

    def __init__(self, nn, optimizer, onehot=False, debug=True):
        self.NN = nn
        self.Optimizer = optimizer
        self.Onehot = onehot
        self.Debug = debug

    def fit(self, x, y, epochs, batch_size, miniloss=None, minideltaloss=None):
        import time

        if batch_size > len(x):
            batch_size = len(x)
        preloss = 0.0
        time_t = time.time()
        interval = 0
        batches = int(len(x) / batch_size)

        # train
        for j in range(epochs):
            for i in range(batches):
                start = i * batch_size % (len(x) - batch_size + 1)
                end = start + batch_size
                part_x = x[start:end]
                part_y = y[start:end]

                loss = self.Optimizer.train(part_x, part_y)

                interval = time.time() - time_t

                if self.Debug and interval > 5:
                    print('epochs: {}/{}, batches: {}/{}, loss: {:.4f}'.format(j+1, epochs, i+1, batches, loss))
                    self.evalute(part_x, part_y)
                    time_t = time.time()
            
            if minideltaloss is not None and np.abs(loss - preloss) < minideltaloss:
                break
            if miniloss is not None and np.abs(loss) < miniloss:
                break

            preloss = loss

        return loss

    def reset(self):
        
        for nn in self.NN:
            nn.reset()

    def predict(self, x):

        # transpose x
        x = np.asmatrix(x).T

        for layer in self.NN:
            x = layer.F(x)

        x = x.T.getA()

        return x

    def evalute(self, x, y):

        predict = self.predict(x)
        if self.Onehot:
            y = y.argmax(axis=1)
            predict = predict.argmax(axis=1)
        else:
            predict = np.round(predict)

        acc = np.mean(np.equal(y, predict))

        if self.Debug:
            print('Accuracy: {:.4f}'.format(acc))

        return acc


class MLPPredictLocal(IModel):
    """
        Multi-layer perceptron with local implementation
    """

    def __init__(self, name, inputlen=30, outputlen=21):
        """
            @param name: Set model name to identify the model in local dir
            @param inputlen: Set model input length
            @param outputlen: Set model prediction length

            Config most parameters in here, instead of the __model_fn()
        """

        IModel.__init__(self, 'mlplc', name)
        self.__InputLength = inputlen
        self.__OutputLength = outputlen
        self.__BatchSize = 32
        self.__HiddenLayerUnits = [32,32,32,32,self.__OutputLength]
        self.__TrainEpoches = 2000
        self.Actvations = [Tanh, Tanh, Tanh, Tanh, Linear]

        self.__loadmodel()
        self.Loss = MseLoss()
        self.Op = GradientDecentOptimizer(self.Loss, self.NN, learnrate=0.01)
        self.Model = Model(self.NN, self.Op, debug=False)

        
    def __loadmodel(self):
        """
            Load model variables from local
        """
        count = len(self.__HiddenLayerUnits)

        if os.path.exists(self.ModelPath + 'model.npy'):
            with open(self.ModelPath + 'model.npy', 'rb') as file:
                weight_bias = np.load(file)

            self.NN = [FCLayer(self.__HiddenLayerUnits[i],
                        w_init = weight_bias[i]['weight'],
                        b_init = weight_bias[i]['bias'],
                        act = self.Actvations[i]()) for i in range(count)]

        else:
            self.NN = [FCLayer(self.__HiddenLayerUnits[i],
                       act = self.Actvations[i]()) for i in range(count)]


    def __savemodel(self):
        """
            Save model variables to local
        """
        weight_bias = [{'weight':nn.W, 'bias':nn.B} for nn in self.NN]

        with open(self.ModelPath + 'model.npy', 'wb+') as file:
            weight_bias = np.save(file, weight_bias)

        self.updatecheckpoint()


    def fit(self, x, y):
        # save dimension parameters
        self.Config['Dims'] = x.shape[2]
        # reshape inputs
        assert x.shape[2] == y.shape[2], 'Shapes not match between inputs and outputs, mlp.py line 41.'
        x = x.reshape([x.shape[0], x.shape[1] * x.shape[2]])
        y = y.reshape([y.shape[0], y.shape[1] * y.shape[2]])
        # save normalization parameters
        self.Config['Std'] = x.std()
        self.Config['Mean'] = x.mean()
        # normalize
        x = (x - self.Config['Mean']) / self.Config['Std']
        y = (y - self.Config['Mean']) / self.Config['Std']

        self.Model.fit(x, y, batch_size = self.__BatchSize, epochs = self.__TrainEpoches)

        self.__savemodel()


    def predict(self, x):
        x = np.asarray(x)
        # reshape x as one sample row
        x = x.reshape([1, self.__InputLength * self.Config['Dims']])
        # normalize inputs
        x = (x - self.Config['Mean']) / self.Config['Std']

        # get result
        result = self.Model.predict(x)

        # denormalize outputs
        result = result * self.Config['Std'] + self.Config['Mean']
        # reshape as origin
        result = result.reshape([self.__OutputLength, self.Config['Dims']])

        return result


    def inputrequire(self):
        return self.__InputLength


    def maxpredict(self):
        return self.__OutputLength


    def resetmodel(self):
        shell.rmtree(self.ModelPath, ignore_errors=True)
        self.__loadmodel()
        self.Model = Model(self.NN, self.Op, debug=False)


if __name__ == '__main__':
    model = MLPPredictLocal('test', inputlen=4, outputlen=1)
    print(model.lastcheckpoint())
    #model.resetmodel()

    x = np.linspace(0,1,4*10).reshape([-1,4,1])
    y = np.sum(x, axis=1).reshape([-1,1,1]) / 4

    model.fit(x, y)
    #model.eval_with_numpy_input(x, y)
    pred = model.predict([[0.44], [0.45], [0.55], [0.55]])
    print(pred)