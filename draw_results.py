#
# @Date: 2024-11-18 20:45:07
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-12-26 10:45:21
# @FilePath: /hy_bak_test_delete_after_used/yolo_hy_complex_network_for_complex_input/draw_results.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import argparse
import os
import numpy as np
import pandas as pd
from collections import Counter

def process(dir):
    result = {}
    result_overall = {}
    folder = []
    folder_overall = []
    ## 先读文件
    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)
        if os.path.isdir(item_path):
            if item_path.count('_')<=1:
                folder_overall.append(item_path)
                continue
            folder.append(item_path)
    for item in folder_overall:
        f = pd.read_csv(item+'/confusion.csv',header=None)
        matrix = f.values
        result_overall['_'.join(item.split('_')[1:])] = matrix
        
    for item in folder:
        f = pd.read_csv(item+'/confusion.csv',header=None)
        matrix = f.values
        result['_'.join(item.split('_')[1:])] = matrix
        
    ## ==========读完文件了========= ##
    ## 开始处理
    ll = ['train', 'valid', 'test']
    ll_overall = ['train', 'val', 'test']
    
    for i in ll:
        exec('{}_snr = []'.format(i))
        exec('{}_accuracy = []'.format(i))
    for key in result_overall.keys():
        matrix = result_overall[key]
        for i in range(matrix.shape[0]):
            matrix[:,i]/=matrix[:,i].sum()
        np.savetxt(dir+'/'+key+'.csv',matrix,delimiter=',')
        
    ### 计算输出准确度
    for key in result.keys():
        idx = ll.index(key.split('_')[0])
        exec('{}_snr.append(key.split("_")[-1])'.format(ll[idx]))
        exec('{}_accuracy.append(calculate(result[key]))'.format(ll[idx]))
        # if ll[1] in key:
        #     valid_snr.append(key.split('_')[-1])
        #     valid_accuracy.append(calculate(result[key]))
        # elif 'test' in key:
        #     test_snr.append(key.split('_')[-1])
        #     test_accuracy.append(calculate(result[key]))
        # else:
        #     train_snr.append(key.split('_')[-1])
        #     train_accuracy.append(calculate(result[key]))
    # 创建一个DataFrame
    
    for i in ll:
        df_train = pd.DataFrame({'Column1': eval('train_snr'), 'Column2': eval('train_accuracy')})
        df_test = pd.DataFrame({'Column1': eval('test_snr'), 'Column2': eval('test_accuracy')})
        df_valid = pd.DataFrame({'Column1': eval('valid_snr'), 'Column2': eval('valid_accuracy')})
        
    # df_valid = pd.DataFrame({'Column1': valid_snr, 'Column2': valid_accuracy})
    # df_test = pd.DataFrame({'Column1': test_snr, 'Column2': test_accuracy})
    # 将DataFrame写入CSV文件 
    with pd.ExcelWriter(dir+'/classify_accuracy.xlsx') as writer: 
        for i in ll:
            exec("df_{}.to_excel(writer, sheet_name='{}', index=False)".format(i, i))
    pass

def calculate(matrix):
    correct = 0
    total = 0
    for i, j in enumerate(matrix):
        total = total + sum(j)
        correct = correct + j[i]
    return correct/total
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--result_dir', type=str, default='results_real', help='')
    opt = parser.parse_args()
    process(opt.result_dir)