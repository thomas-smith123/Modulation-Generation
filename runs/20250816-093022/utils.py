#
# @Date: 2024-10-11 17:04:55
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-10-11 17:04:58
# @FilePath: /hy_bak_test_delete_after_used/complex_gru_single/utils.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import matplotlib as mpl
import numpy as np
from PIL import Image

def applyColorMap(heatmapshow,colormap='jet'):
    ##input :   img:np.array 2D
    cmhot = mpl.colormaps[colormap]
    heatmapshow=cmhot(heatmapshow)
    heatmapshow=np.uint8(heatmapshow*255)
    im = Image.fromarray(heatmapshow[:,:,0:3])
    return im

def NormMinandMax(arr, min=0, max=255,out_type=np.uint8):
    """"
    将数据npdarr 归一化到[min,max]区间的方法
    返回 副本
    """
    Ymax = np.max(arr)  # 计算最大值
    Ymin = np.min(arr)  # 计算最小值
    k = (max - min) / (Ymax - Ymin)
    last = min + k * (arr - Ymin)
    return last.astype(out_type)