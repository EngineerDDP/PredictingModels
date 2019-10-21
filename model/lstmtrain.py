from predict.model.interfaces import IModel
from keras import Sequential,Model,layers,optimizers,callbacks
from keras.models import load_model
from keras.backend import  clear_session
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

class LMTRAIN(IModel):

    def __init__(self, name, inputlen=241, outputlen=1):
        IModel.__init__(self, 'lstmtrain', name)
        self.__InputLength = inputlen
        self.__OutputLength = outputlen
        self.pname=name
        self.__HiddenLayer=32

    def fit(self, x, y):
        model=Sequential()
        model.add(layers.LSTM(self.__HiddenLayer,input_shape=( x.shape[1],1),return_sequences=True))
        model.add(layers.LSTM(self.__HiddenLayer, activation='relu', return_sequences=True))
        model.add(layers.LSTM(self.__HiddenLayer, activation='relu'))
        model.add(layers.Dense(1))
        #callbacks_list=[callbacks.EarlyStopping(monitor='acc',patience=1,)]
        model.compile(loss='mse',optimizer='adam')
        #model.fit(x,y,epochs=10,batch_size=30,callbacks=callbacks_list,validation_split=0.1)
        model.fit(x, y, epochs=10, batch_size=30, validation_split=0.1)

        return model

    def inputrequire(self):
        return self.__InputLength

    def maxpredict(self):
        return self.__OutputLength


    def __createdate(self,dataset,lookback=30):
        datax,datay=[],[]
        for i in range(len(dataset)-lookback-1):
            datax.append(dataset[i:(i+lookback)].T)
            datay.append(dataset[i+lookback])
        return np.asarray(datax), np.asarray(datay)



    def predict(self, x):
        x = np.asarray(x)
        pred = []
        dims=x.shape[1]
        lookback = 30
        scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
        x=scaler.fit_transform(x)
        #rootdir=os.getcwd()+'\\modelweights\\'
        rootdir = os.getcwd() + '\\predict\\model\\modelweights\\'
        for dim in range(dims):
            train = x[-self.__InputLength:, dim]
            xtrain,ytrain=self.__createdate(train,lookback)
            xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
            ytrain=ytrain.reshape(ytrain.shape[0],1)
            clear_session()
            model = self.fit(xtrain, ytrain)
            savedir = rootdir + 'LSTM{0}-{1}.h5'.format(self.pname, dim)
            model.save(savedir)
        return np.asarray([[0]])

if __name__ == "__main__":

    import pandas as pd

    model = LMTRAIN('test', inputlen=241, outputlen=1)
    prep=[]
    datas=pd.read_csv("D:/test/S125-66c.csv").preprocess[:-2]
    for data in datas:
        prep.append([data])
    model.predict(prep)

'''
    prep = []
    datas = pd.read_csv("D:/test/S125-66c.csv").preprocess[-32:-2]
    for data in datas:
        prep.append([data])
    prep=np.asarray(prep)
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
    prep = scaler.fit_transform(prep)
    prep=prep[:,0]
    prep=prep.reshape(1,prep.shape[0],1)
    readdir='D:\ccj.git\\branches\\build_0.1\predict\model\modelweights\LSTMtest-0.h5'
    model=load_model(readdir)
    predict=model.predict(prep)
    predict=scaler.inverse_transform(predict)[0]
    print(predict)
'''


