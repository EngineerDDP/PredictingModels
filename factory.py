# -*- coding:utf-8 -*-

from predict.model.lsr import LeastSquareEstimator

try:
    from predict.model.mlp import MLPPredictLocal
except ImportError:
    from predict.model.mlpLocal import MLPPredictLocal

from predict.model.poly import PolyFit
from predict.model.arima import ARIMA
from predict.model.nopredict import NoPrediction

def CreateModel(model='None', name='None'):
    """
        模型工厂方法，根据标志字符串构造模型类
    """

    # 逗号分隔
    models = model[1].split(',')

    # 默认值
    if len(models) >= 1:
        mname = models[0]
    else:
        mname = 'LSR'

    if len(models) >= 2:
        length = int(models[1])
    else:
        length = 7

    if len(models) >= 3:
        outlen = int(models[2])
    else:
        outlen = 1
    

    if mname.startswith('MLP'):
        predict = MLPPredictLocal(name, length, outlen)
    elif mname.startswith('LSR'):
        predict = LeastSquareEstimator(name, length, outlen)
    elif mname.startswith('Poly'):
        predict = PolyFit(name, length, outlen)
    elif mname.startswith('ARIMA'):
        predict = ARIMA(name, length, outlen)
    elif mname.startswith('None'):
        predict = NoPrediction(name, length, outlen)
    else:
        predict = LeastSquareEstimator(name, length, outlen)

    return predict
