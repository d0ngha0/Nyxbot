# built-in package for computing and ploting
import numpy as np
import math
import os
from matplotlib import rcParams
# custom package for force transformation
from Scripts.ForceMapping import forceMapping as fm
from Scripts.filehandle import get_data_in_directory
np.set_printoptions(suppress=True)


class sampleGet():
    def __init__(self, path):
        
        # retrieve data from the path
        self.data = get_data_in_directory(path)
        # the container for save the washed data
        self.data_sample = []
        # period for one circle of the locomotion
        self.period = 140

    def GetData(self):
        period = self.period
        for selected_id in np.arange(1,len(self.data)):
            '''data wash'''
            # 1. correct the mapping position between joint torque and GRFs
            # 2. eliminate the temperatue shifting of force sensor
            force_data_washed = self.force_washing(self.data[selected_id])
            Two_d_force = self.get_2d_force(force_data_washed)
            self.data_sample.append(np.hstack((force_data_washed[period*1:period*4,:32], Two_d_force[period*1:period*4,:])))
            
        return self.data_sample


    def get_2d_force(self, data):
        forceReader = fm()
        force_test = np.zeros((data.shape[0],2*4))
        # transfer the raw force gauge value to 2D GRFs
        for i in range(data.shape[0]):
            force_test[i,:2]=forceReader.get_fs2(data[i,32:35])[:2]
            force_test[i,2:4]=forceReader.get_fs6(data[i,36:39])[:2]
            force_test[i,4:6]=forceReader.get_fs4(data[i,40:43])[:2]
            force_test[i,6:8]=forceReader.get_fs3(data[i,44:47])[:2]
            
        force_test[:,1] = -force_test[:,1]
        force_test[:,-1] = -force_test[:,-1]
        return force_test
    
    def force_washing(self,data):
        # Swap the raw force data of FS4 and FS6 
        # The correct FS data to fit the limb position FS2 -> FS6 -> FS4 -> FS3
        data_swaped = self.swap_columns(data)
        # eliminate the error by substract the force sensor value at each stance phase
        for i in range(4):
            # the robot walk at diagonal gait, stance phase occur at different time
            # force in leg 1 and 3
            if i % 2 == 0:
                # caculate the baseline for eraser the error generate by tempearture shifting of force sensor
                data_swaped[:,32+4*i:36+4*i] -= self.calculate_column_averages(data_swaped[:,32+4*i:36+4*i],125,135)
            # force in leg 2 and 4
            else:
                data_swaped[:,32+4*i:36+4*i]-= self.calculate_column_averages(data_swaped[:,32+4*i:36+4*i],75,85 )
        return data_swaped
    
    def calculate_column_averages(self, data, x, y):
        """
        计算(n,16)数据每行指定列从x到y的平均值，并返回这一组平均值(16,)
        
        参数:
        data -- 输入数据，形状为(n, 16)的numpy数组
        x -- 起始列索引(包含)
        y -- 结束列索引(包含)
        
        返回:
        形状为(16,)的numpy数组，包含每列从x到y行的平均值
        """
        # 确保x和y在有效范围内
        if x < 0 or y >= data.shape[0] or x > y:
            raise ValueError("无效的x或y值")
        
        # 提取从x到y的所有行
        selected_rows = data[x:y+1, :]
        
        # 计算每列的平均值
        column_averages = np.mean(selected_rows, axis=0)
        
        return column_averages

    def swap_columns(self, data):
        """
        调换数据中[36:40]和[40:44]列的位置
        
        参数:
        data -- 输入数据，形状为(n, 54)的numpy数组
        
        返回:
        调换列后的新数组
        """
        # 创建数据的副本以避免修改原始数据
        swapped_data = data.copy()
        
        # 交换列[36:40]和[40:44]
        swapped_data[:, 36:40], swapped_data[:, 40:44] = data[:, 40:44], data[:, 36:40]
        
        return swapped_data