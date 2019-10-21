# -*- coding:utf-8 -*-

from abc import abstractclassmethod, ABCMeta


class IPredict(metaclass=ABCMeta):

    @abstractclassmethod
    def InputLength(self):
        """
            获取模型需要的输入长度
        """
        pass


    @abstractclassmethod
    def OutputLength(self):
        """
            获取模型能够输出的数据长度
        """
        pass


    @abstractclassmethod
    def InitModel(self, x):
        """
            使用训练集合初始化预测模型
            @param x: 样本变化序列，按照等间距采样获得，每行一个样本
        """
        pass


    @abstractclassmethod
    def UpdateModel(self, x):
        """
            使用给定的样本序列更新现有模型
            @param x: 样本变化序列，按照等间距采样获得，每行一个样本
        """
        pass


    @abstractclassmethod
    def Predict(self, x):
        """
            使用给定的样本拟合并预测对应输入x的输出值
            @param x: 待预测的输入值
        """
        pass


    @abstractclassmethod
    def ModelInitializationRequired(self):
        """
            检查所配置的模型是否需要初始化
        """
        pass


    @abstractclassmethod
    def GetLastUpdateTime(self):
        """
            查询上次更新该模型的时间
            返回时间字符串
        """
        pass

    
    @abstractclassmethod
    def SetModelSelections(self, models):
        """
            配置各个流程所使用的模型
            @param models: 预测阶段使用的模型列表，按照初始化顺序排列，多余的选项将被忽略
        """
        pass


class TestPredict(IPredict):

    def InitModel(self, x):
        return None

    def InputLength(self):
        return 2

    def ModelInitializationRequired(self):
        return False

    def OutputLength(self):
        return 1

    def Predict(self, x):
        return x[-1]

    def SetModelSelections(self, models):
        return None

    def UpdateModel(self, x):
        return None