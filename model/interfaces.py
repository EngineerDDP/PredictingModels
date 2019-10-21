import os
import json
import shutil as shell

from datetime import datetime
from abc import abstractclassmethod, ABCMeta


class IModel(metaclass=ABCMeta):


    def __init__(self, modeltype, name):
        """
            初始化模型配置文件
        """
        self.ModelPath = './{}models/'.format(modeltype) + name + '/'
        self.Config = dict()
        self.__ConfigPath = self.ModelPath + 'config.json'

        self.__loadconfig()
        

    def __loadconfig(self):
        """
            载入配置文件
        """
        if os.path.exists(self.__ConfigPath):
            with open(self.__ConfigPath, 'r') as configfile:
                self.Config = json.load(configfile)
        else:
            self.Config['LastCheckPoint'] = '1970-01-02 00:01:01.000'
            self.__saveconfig()


    def __saveconfig(self):
        """
            存储配置文件
        """
        if not os.path.exists(self.ModelPath):
            os.makedirs(self.ModelPath)
        with open(self.__ConfigPath, 'w+') as configfile:
            json.dump(self.Config, configfile)


    @abstractclassmethod
    def fit(self, x, y):
        """
            初始化模型参数并拟合给定的数据
            @param x: 输入值
            @param y: 标签值
        """
        pass

    
    @abstractclassmethod
    def predict(self, x):
        """
            使用拟合结果对给定的自变量（输入值）x ，预测其因变量（输出值）
            @param x: 输入值
        """
        pass


    @abstractclassmethod
    def maxpredict(self):
        """
            返回最大可预测长度
        """
        pass


    @abstractclassmethod
    def inputrequire(self):
        """
            返回输入长度需求
        """
        pass


    def lastcheckpoint(self):
        """
            返回上一个训练检查点的日期的字符串形式
        """
        return self.Config['LastCheckPoint']


    def updatecheckpoint(self):
        """
            更新检查点时间到现在
        """
        self.Config['LastCheckPoint'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.__saveconfig()


    def resetmodel(self):
        """
            重置模型的状态，删除缓存
        """
        shell.rmtree(self.ModelPath)