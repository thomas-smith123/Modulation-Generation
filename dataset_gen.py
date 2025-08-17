from SignalDef import *
from SignalDescription import *

from SignalDef import *
from SignalDescription import *
from tqdm import tqdm
import random
from utils.transforms import Spectrogram,Normalize
from matplotlib import pyplot as plt
import numpy as np


# class SignalDataGen:
#     default_class: List[str] = [
#         "LFM",
#         "2FSK",
#         "4FSK",
#         "8FSK" ,
#         "4QAM",
#         "16QAM"
#         "AM",
#         "DSB", #@audit DSB总是很淡
#         "FM",
#         "2PSK",
#         "4PSK",
#         "8PSK",
#         "RADAR_Pulse",
#         "RADAR_Pulse_Compressed",
#         "NLFM",
#         "ADSB"
#     ]
    
#     def __init__(self,train_num:int=10000,valid_num:int=1000) -> None:
        
#         # enumerate
#         self.label_map = {}
#         for i,j in enumerate(self.default_class):
#             self.label_map[j] = i ## 这个好，不用update了
            
#     def yolo_label_gen(self,signal_sample:list,filename:str):## 主要看写txt有没有问题
#         signal = signal_sample[0]
#         signal_description = signal_sample[1]
#         f = open(file=filename,mode='w')
#         if signal_description != None:
#             for description in signal_description:
#                 tag = description.class_name
#                 lower = description.lower_frequency+0.5
#                 upper = description.upper_frequency+0.5
#                 ##########################################
#                 label = self.label_map[tag]
#                 center_x = (description.start+description.stop)/2
#                 center_y = (lower+upper)/2
#                 width = description.stop-description.start
#                 height = upper-lower
#             f.write(str(label)+' '+str(center_x)+' '+str(center_y)+' '+str(width)+' '+str(height)+'\n')
#         else:
#             f.close()
        
        
        
        
# if __name__=="__main__":
    
#     a = SignalDataGen()
    
#     pass
    