from predict.model.interfaces import IModel
from keras import Sequential,Model,layers,optimizers,callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

class LSTM(IModel):

    def __init__(self, name, inputlen=30, outputlen=1):
        IModel.__init__(self, 'lstm', name)
        self.__InputLength = inputlen
        self.__OutputLength = outputlen
        self.__HiddenLayer=32
    def fit(self, x, y):
        #outlen=len(x[0])
        model=Sequential()
        model.add(layers.LSTM(self.__HiddenLayer,input_shape=( x.shape[1],1),return_sequences=True))
        #model.add(layers.LSTM(self.__HiddenLayer,activation='relu',return_sequences=True))
        model.add(layers.LSTM(self.__HiddenLayer, activation='relu'))
        model.add(layers.Dense(1))
        callbacks_list=[callbacks.EarlyStopping(monitor='acc',patience=1,)]
        #optimizer=optimizers.RMSprop(lr=0.01)
        model.compile(loss='mse',optimizer='adam')
        model.fit(x,y,epochs=10,batch_size=30,callbacks=callbacks_list,validation_split=0.1)

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
        scaler=MinMaxScaler(feature_range=(-1,1),copy=False)
        #scaler=StandardScaler()
        x=scaler.fit_transform(x)
        for dim in range(dims):
            x1=x[0:-self.__InputLength,dim]
            xtrain,ytrain=self.__createdate(x1,lookback)
            xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
            ytrain=ytrain.reshape(ytrain.shape[0],1)
            x2=x[-self.__InputLength:,dim]
            xtest=x2.reshape(1,x2.shape[0],1)
            model = self.fit(xtrain, ytrain)
            predict = model.predict(xtest)
            predict=scaler.inverse_transform(predict)[0]
            pred.append(predict)
        pred = np.asarray(pred).transpose()
        return pred

if __name__ == "__main__":
    model = LSTM('test', inputlen=30, outputlen=1)
    import pandas as pd
    prep=[]
    datas=pd.read_csv("D:/test/S125-66c.csv").preprocess[:-2]
    for data in datas:
        prep.append([data])
    #model.predict(prep)
    pred=model.predict(prep)
    print(pred)