# -*- coding:utf-8 -*-

import numpy as np

import predict.factory as factory
from predict.interfaces import IPredict
from datetime import datetime
from predict.exceptions import PredictionFailureException

class Predict(IPredict):
    """
        预测管理类，负责处理调度器发来的预测任务
    """

    def __init__(self, name):
        self.PredictModel = None
        self.Name = name

    
    def InitModel(self, x):
        self.PredictModel.resetmodel()
        self.UpdateModel(x)


    def InputLength(self):
        return self.PredictModel.inputrequire()


    def ModelInitializationRequired(self):
        print('直接查询模型是否需要更新的接口已经过时，请使用 GetLastUpdateTime() 来获取上次更新的检查点时间。')
        cp = self.PredictModel.lastcheckpoint()
        cp = datetime.strptime(cp, '%Y-%m-%d %H:%M:%S.%f')
        cp = cp.timestamp()
        now = datetime.now().timestamp()
        if now - cp > 38400:
            return True
        else:
            return False


    def GetLastUpdateTime(self):
        return self.PredictModel.lastcheckpoint()


    def OutputLength(self):
        return self.PredictModel.maxpredict()


    def Predict(self, x):
        try:
            return self.PredictModel.predict(x).tolist()
        except Exception:
            self.PredictModel.resetmodel()
            raise PredictionFailureException('Prediction on {} failed, model has been reset.'.format(self.Name))


    def SetModelSelections(self, models):
        self.PredictModel = factory.CreateModel(models, self.Name)


    def UpdateModel(self, x):
        x = np.asarray(x)
        train_x = x[:,:self.InputLength(),:]
        train_y = x[:,-self.OutputLength():,:]

        self.PredictModel.fit(train_x, train_y)