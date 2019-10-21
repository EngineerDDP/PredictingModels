from predict.model.interfaces import IModel
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.backend import clear_session
import numpy as np
import os


class LSTM(IModel):

    def __init__(self, name, inputlen=30, outputlen=1):
        IModel.__init__(self, 'lstm', name)
        self.__InputLength = inputlen
        self.__OutputLength = outputlen
        self.pname = name
        self.__HiddenLayer=32

    def fit(self, x, y):
        self.updatecheckpoint()
        pass

    def inputrequire(self):
        return self.__InputLength

    def maxpredict(self):
        return self.__OutputLength

    def predict(self, x):
        x = np.asarray(x)
        pred = []
        dims=x.shape[1]
        scaler=MinMaxScaler(feature_range=(-1,1),copy=False)
        x=scaler.fit_transform(x)
        rootdir = os.getcwd() + '\\predict\\model\\modelweights\\'
        for dim in range(dims):
            xtest=x[-self.__InputLength:,dim]
            xtest=xtest.reshape(1,xtest.shape[0],1)
            readdir=rootdir+'LSTM{0}-{1}.h5'.format(self.pname,dim)
            clear_session()
            model=load_model(readdir)
            predict = model.predict(xtest)
            predict=scaler.inverse_transform(predict)[0]
            pred.append(predict)
        pred = np.asarray(pred).transpose()
        return pred