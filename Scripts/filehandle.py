
import numpy as np
import os


'''读取并返回文件夹中的数据'''
def get_data_in_directory(path):
    data = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        
        # 这里注意需要把读入的数据进行翻转
        data.append(np.load(file_path).T)
    return data