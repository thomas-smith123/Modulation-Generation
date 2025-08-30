#
# @Date: 2024-05-28 09:40:53
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-08-23 10:22:54
# @FilePath: /hy_bak_test_delete_after_used/network_yolo_from_hy/utils/plot_sth.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import numpy as np
import torch
import matplotlib as mpl
from PIL import Image
from matplotlib import pyplot as plt
import plotly.graph_objects as go

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

def applyColorMap(heatmapshow,colormap='jet'):
    ##input :   img:np.array 2D
    bs = heatmapshow.shape[0]
    cmhot = mpl.colormaps[colormap]
    heatmapshow=cmhot(heatmapshow)
    heatmapshow=np.uint8(heatmapshow*255)
    # im = Image.fromarray(heatmapshow[...,0:3])
    # u = np.array([1,2,3])
    # plt.imshow(heatmapshow[0,...,u].transpose(1,2,0));plt.savefig('p.png')
    # if len(heatmapshow.shape)==5:
    #     im = np.stack([Image.fromarray(heatmapshow[i,0][...,0:3]) for i in range(bs)])
    # else:
    #     im = np.stack([Image.fromarray(heatmapshow[i,...,0:3]) for i in range(bs)])
    im = np.array(heatmapshow[...,0:3])
    im = im.transpose(2,0,1)
    return im

def plot_distribution(data_x,data_y,bins=[40,40],range=(-3, 3, -3, 3),name='3d_histogram.html'):

    # 计算二维直方图
    hist, bin_edges = torch.histogramdd(torch.stack([data_x,data_y],1), bins=bins, range=range)


    # 转换为 numpy 数组以便使用 Plotly
    hist = hist.numpy()
    x_edges = bin_edges[0].numpy()
    y_edges = bin_edges[1].numpy()

    # 创建网格
    x_center = (x_edges[:-1] + x_edges[1:]) / 2
    y_center = (y_edges[:-1] + y_edges[1:]) / 2

    # 创建二维数组表示每个bin的中心位置
    x, y = np.meshgrid(x_center, y_center)

    # 绘制3D直方图
    fig = go.Figure(data=[go.Surface(z=hist.T, x=x, y=y, colorscale='Viridis')])

    # 更新布局
    fig.update_layout(
        title='3D Histogram',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Num'
        )
    )

    # 保存为 HTML 文件
    html_file_path = name+'.html'
    fig.write_html(html_file_path)

if __name__ == "__main__":
    x = torch.randn(1000)
    y = torch.randn(1000)
    plot_distribution(x,y)