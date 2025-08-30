import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7,6"
from typing import Any
import commpy as cpy
import scipy.fftpack as fftpack
from SignalDef import *
from skcuda import cufft as cf
from SignalDescription import *
from tqdm import tqdm
import random,cv2,os,ctypes,json,argparse
from utils.transforms import Spectrogram,Normalize
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import multiprocessing as mul
from PIL import Image
import pycuda.autoinit
import matplotlib.pyplot as plt
from pycuda import gpuarray


cmaps = [ 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'twilight', 'twilight_shifted', 'hsv',
            'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

class SignalDataGen:
    default_class: List[str] = [
        "LFM",# 0
        "2FSK",# 1 ==> MSK
        "Costas",# 2
        "Frequency_HOP",# 4
        "16QAM",# 4
        "64QAM",# 5
        "256QAM",# 6
        "4PSK",# 7
        "8PSK",# 8
        "16PSK",# 9
        "32PSK",# 10
        "16APSK",# 11
        "32APSK",# 12
        "64APSK",# 13
        "RadarPulse_Compressed",# 15
        "NLFM",# 16
        "ADSB",# 17
        "OFDM", #18
        "Zigbee",
        "LoRa",
    ]
    ##
    # 
    ##
    def __init__(
        self,
        train:int=100,
        valid:int=100,
        fft_size:int=512,
        ) -> None:
        ## 训练样本数量&验证样本数量
        self.AllClass = self.default_class
        self.sample_rate = 5e9
        self.max_signal_frame = 1 #一帧最多5个信号
        self.train_num = train
        self.valid_num = valid
        self.num = self.train_num+self.valid_num
        ## 其他参数
        self.fft_size = fft_size #定义图像为正方形
        self.noverlap = self.fft_size // 8 #重叠点数
        self.overlap = fft_size-self.noverlap #步长
        self.nperseg = self.fft_size #窗长
        self.num_iq_samples = (self.fft_size-1)*(self.noverlap)+self.fft_size
        self.label_map = {}
        for i,j in enumerate(self.default_class):
            self.label_map[j] = i ## 这个好，不用update了
        t = ctypes.c_int(int(self.num_iq_samples))
        inembed_p = ctypes.pointer(t)
        n = ctypes.c_int(fft_size)
        n_p = ctypes.pointer(n)
        onembed = ctypes.c_int*2 ## 数组
        onembed_p = onembed(fft_size,fft_size)

        self.plan = cf.cufftPlanMany(rank=1, n=n_p, inembed=inembed_p, istride=1, idist=self.noverlap,onembed=onembed_p, ostride=1, odist=fft_size, fft_type=cf.CUFFT_C2C, batch=fft_size)
        self.data_o_gpu = gpuarray.zeros((512,512),dtype=np.complex64)

    def checkFrequency(self,frequency_list,bandwidth_list,signal_parameter: SignalParameter):
        for i,j in enumerate(frequency_list):
            if np.abs(signal_parameter.center_frequency-j)<(bandwidth_list[i]/2+np.abs(signal_parameter.bandwidth/2)): 
                if np.abs(j-signal_parameter.center_frequency)<bandwidth_list[i]/2 or np.abs(j-signal_parameter.center_frequency)<signal_parameter.bandwidth/2+0.01: #直接重合了
                    signal_parameter.center_frequency+=np.max((bandwidth_list[i],j))
                else:
                    overlap = -(np.abs(j-signal_parameter.center_frequency)-bandwidth_list[i]/2-signal_parameter.bandwidth/2)
                    if signal_parameter.center_frequency > j:
                        signal_parameter.center_frequency += overlap+0.02
                    else:
                        signal_parameter.center_frequency += bandwidth_list[i]+signal_parameter.bandwidth-overlap+0.02
                    if signal_parameter.center_frequency > 0.5*self.sample_rate:
                        signal_parameter.center_frequency = (self.sample_rate/2 - signal_parameter.center_frequency)
            else:
                continue
        return signal_parameter.center_frequency
        
    def genFrame(self,snr=0):
        frequency_list = [] # 中心频率
        bandwidth_list = []
        description_list = []
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        num = np.random.randint(1,self.max_signal_frame+1) #该帧信号的数量
        if np.random.uniform(0,1)>0.8:
            self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
            return [self.iq_data,None] #@audit 不能这样返回，至少得返回一个全0的信号
        choosed_signal = random.sample(self.default_class,num) ## 抽取信号的调制样式
        for i in choosed_signal:
            continue_flag = np.random.uniform(0,1) # 0:不继续，1:继续
            ## 对FM族信号
            signal_parameter = SignalParameter(
                num_iq_samples=self.num_iq_samples,
                sample_rate=self.sample_rate
                )
            if continue_flag > 0.1:# 标记为连续信号
                signal_parameter.start = 0
                signal_parameter.stop = 1*signal_parameter.num_iq_samples/signal_parameter.sample_rate
                signal_parameter.duration = 1*signal_parameter.num_iq_samples/signal_parameter.sample_rate
            if i == "Frequency_HOP":
                signal_parameter.center_frequency = np.random.uniform(-0.3,0.3)*self.sample_rate/2
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter.bandwidth = np.random.uniform(0.1,0.2)*self.sample_rate/2
                signal_parameter.num_symbol = np.random.randint(20,35)
                signal_parameter()
                signal = nFSK(signal_parameter,8)
            elif "FM" in i: #[FM LFM NLFM]
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)

                if i == "LFM":
                    signal = LFM(signal_parameter)
                    signal_parameter()
                elif i == "NLFM":
                    signal_parameter.bandwidth = (np.random.uniform(0.1,0.2))*self.sample_rate/2
                    signal_parameter()
                    signal = NLFM(signal_parameter)
            elif "FSK" in i: # nFSK
                n = int(i.split("FSK")[0])
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter.center_frequency = np.random.uniform(-0.2,0.2)*self.sample_rate
                signal_parameter.bandwidth = np.random.uniform(0.01,0.05)*self.sample_rate/2
                signal_parameter.num_symbol = np.random.randint(10,40)
                # signal_parameter.duration = np.random.uniform(0.2, 0.8)*self.num_iq_samples*1/self.sample_rate
                signal_parameter()
                signal = nFSK(signal_parameter,n)
            elif "QAM" in i: # nQAM
                if i == "16QAM":
                    n = 16
                elif i == "64QAM":
                    n = 64
                elif i == "256QAM":
                    n = 256
                else:
                    n = 16  # 默认值
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = nQAM(signal_parameter,n)
            elif "APSK" in i:
                if i == "16APSK":
                    n = 16
                elif i == "32APSK":
                    n = 32
                elif i == "64APSK":
                    n = 64
                else:
                    n = 16  # 默认值
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = nAPSK(signal_parameter,n)
            elif "PSK" in i:
                if i == "4PSK":
                    n = 4
                elif i == "8PSK":
                    n = 8
                elif i == "16PSK":
                    n = 16
                elif i == "32PSK":
                    n = 32
                else:
                    n = 4  # 默认值
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = nPSK(signal_parameter,n)
            elif "Radar" in i:
                if "Compressed" in i:
                    signal = RADAR_Pulse_Compressed(signal_parameter,random.uniform(0.1,0.4))
                    signal_parameter.num_symbol = np.random.randint(5,10)
                    signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                    signal_parameter()
            elif i == "ADSB":
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                # signal_parameter.samples_per_symbol = np.random.randint(3000,6000)
                signal_parameter()
                signal = ADSB(signal_parameter)
            elif i == "Costas":
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter.bandwidth = np.random.uniform(0.05,0.15)*self.sample_rate/2
                signal_parameter()
                signal = Costas(signal_parameter)

            elif i == "OFDM":
                # continue
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = OFDM(signal_parameter)
            elif i == "Zigbee":
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = Zigbee(signal_parameter)
            elif i == "LoRa":
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = LoRa(signal_parameter)
            frequency_list.append(signal_parameter.center_frequency)
            bandwidth_list.append(signal_parameter.bandwidth)
            signal() ## 生成信号
            description_list.append(signal.signal_description) # 信号的描述
            self.iq_data += signal.iq_data
        self.iq_data.real = cpy.awgn(self.iq_data.real,snr_dB=snr)
        self.iq_data.imag = cpy.awgn(self.iq_data.imag,snr_dB=snr)
        return [self.iq_data,description_list]
    
    def __yolo_label_gen__(self,signal_sample:list,snr,filename:str,seq:int,check:bool=False):
        signal_ = signal_sample[0] ## 这个要转换成图片
        signal_description = signal_sample[1]
        f = open(filename+'labels/'+str(seq)+'_'+str(snr)+'.txt','w')
        
        if check:
            font={
                'style': "italic",
                'weight': "normal",
                'fontsize':12,
                'color': "red",
            }
            fig, ax = plt.subplots()
            
        if signal_description != None:
            for description in signal_description:
                tag = description.class_name
                lower = description.lower_frequency+0.5
                upper = description.upper_frequency+0.5
                ##########################################
                label = self.label_map[tag]
                center_x = (description.start+description.stop)/2
                center_y = (lower+upper)/2
                width = description.stop-description.start ## 稍微放松一下
                height = np.abs(upper-lower)
                
                if center_y > 1:
                    center_y = center_y - 1
                if height < 0:
                    height = -height
                
                if center_y + height/2 > 1:
                    height = (1 - center_y) * 2
                if center_y - height/2 < 0:
                    height = center_y * 2
                
                if center_x+width/2 > 1:
                    width = (1-center_x)*2
                if center_x-width/2 < 0:
                    width = center_x*2
                f.writelines(str(label)+' '+str(center_x)+' '+str(center_y)+' '+str(width)+' '+str(height)+'\n')
                
                if check:
                    a,b,c = ConvertDescriptionToPatch(description,nfft=self.fft_size)
                    test_Rectangle1 = mpl.patches.Rectangle(a, width=b, height=c, angle=0, fill=False)
                    ax.add_patch(test_Rectangle1)
                    ax.text(a[0],a[1],description.class_name,fontdict=font)
        else:
            f.writelines('')
            # f.close()
        f.close()
        data_t_gpu  = gpuarray.to_gpu(signal_.astype(np.complex64))
        cf.cufftExecC2C(self.plan, int(data_t_gpu.gpudata), int(self.data_o_gpu.gpudata), cf.CUFFT_FORWARD)
        tf = fftpack.fftshift(self.data_o_gpu.get(),1)
        if signal_description==None:
            tf = np.random.normal(0, np.abs(np.random.normal()+0.1), tf.shape)*np.random.uniform(0,100)
        np.save(filename+'stft_complex/'+str(seq)+'_'+str(snr),tf.T)
        if check:
            plt.imshow(np.abs(tf.T),cmap='viridis')
            # plt.pcolormesh(np.abs(tf),shading='gouraud')
            plt.savefig(filename+'check/'+str(seq)+'_'+str(snr)+'.png')
            plt.close()
        np.save(filename+'raw_complex/'+str(seq)+'_'+str(snr),signal_.T)

        heatmapshow = NormMinandMax(np.abs(tf.T))
        heatmapshow = applyColorMap(heatmapshow,cmaps[random.randint(0,len(cmaps)-1)])
        heatmapshow.save(filename+'images/'+str(seq)+'_'+str(snr)+'.png')

    def __call__(self, snr_list:list,addr:str='DataSet', *args: Any, **kwds: Any) -> Any:
        if not os.path.exists(addr+'/check/'):
            os.mkdir(addr+'/check/')
        if not os.path.exists(addr+'/images/'):
            os.mkdir(addr+'/images/')
        if not os.path.exists(addr+'/labels/'):
            os.mkdir(addr+'/labels/')
        if not os.path.exists(addr+'/raw_complex/'):
            os.mkdir(addr+'/raw_complex/')
        if not os.path.exists(addr+'/stft_complex/'):
            os.mkdir(addr+'/stft_complex/')
        cnt = 0
        for snr in snr_list:
            pbar = tqdm(total=self.num,postfix='SNR:'+str(snr))
            for i in range(self.num):
                self.__yolo_label_gen__(self.genFrame(snr),snr,filename=addr+'/',seq=i+cnt,check=True)
                pbar.update(1)
            cnt += self.num
            ####划分训练集测试集和验证集
        file_list = os.listdir(os.path.join(addr,'images'))
        train_list = np.random.choice(file_list,int(0.7*cnt),replace=False)
        valid_list = np.random.choice(np.array(list(set(file_list)-set(train_list))),int(0.15*cnt),replace=False)
        test_list = np.array(list(set(file_list)-set(train_list)-set(valid_list)))
        
        f = open(addr+'/train.txt','w')
        for i in train_list:
            f.writelines('./images/'+i+'\n')
        f.close()
        f = open(addr+'/valid.txt','w')
        for i in valid_list:
            f.writelines('./images/'+i+'\n')
        f.close()
        f = open(addr+'/test.txt','w')
        for i in test_list:
            f.writelines('./images/'+i+'\n')
        f.close()
        print('All modulations:');print(self.default_class)
        self.yolo2coco(addr)
        # pass
    def yolo2coco(self,addr):
        train_path = f'{addr}/train.txt'
        valid_path = f'{addr}/valid.txt'
        test_path = f'{addr}/test.txt'
        self.__yolo2coco__(train_path)
        self.__yolo2coco__(valid_path)
        self.__yolo2coco__(test_path)
        
    def convert_yolo_to_coco(self,x_center, y_center, width, height, img_width=512, img_height=512):
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        width = width * img_width
        height = height * img_height
        return [x_min, y_min, width, height]

    def __yolo2coco__(self,filepath):
        categories = [{"id": j, "name": i} for j,i in enumerate(self.default_class)]
    #     categories = [
    #     {"id": 1, "name": "category1"},
    #     {"id": 2, "name": "category2"},
    #     # 添加更多类别
    # ]
        annotation_id = 1
        struc = {
            "images": [],
            "annotations": [],
            "categories": categories}

        label_path = filepath.split('/')[0]
        with open(filepath,'r') as f:
            ff = open(filepath.replace('txt','json'),'w')
            for line in f:
                images = line.split('/')[-1].split('\n')[0] ## 不确定后面的\n有没有用
                tmp_path = line.split('\n')[0].split('.')[1].replace('images','labels')
                label_file = open(label_path+tmp_path+'.txt','r')
                image_info = {
                    "file_name": images,
                    "id": len(struc["images"]) + 1,
                    "width": 512,
                    "height": 512
                }
                struc["images"].append(image_info)
                
                for i in label_file:
                    category_id, x_center, y_center, width, height = map(float, i.split())
                    bbox = self.convert_yolo_to_coco(x_center, y_center, width, height)
                    annotation = {
                                "id": annotation_id,
                                "image_id": image_info["id"],
                                "category_id": int(category_id) + 1,
                                "bbox": bbox,
                                "area": bbox[2] * bbox[3],
                                "iscrowd": 0
                            }
                    struc["annotations"].append(annotation)
                    annotation_id += 1
            json.dump(struc, ff,indent=4)
            ff.close()
        
        
    
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

def rc_h(t,beta=0,T=1,normlize=False):
    """TODO: RC的单位脉冲响应
    
    :t: 时刻,numpy array
    :beta: 滚降因子
    :T: 符号周期
    :normlize: 是否能量归一化
    :returns: numpy array

    """
    # https://en.wikipedia.org/wiki/Raised-cosine_filter
    # Raised-cosine filter
    pi = np.pi
    if beta == 0:
        h = 1/T*np.sinc(t/T)
    else:
        case1 = pi/(4*T)*np.sinc(1/(2*beta))
        case2 = 1/T*np.sinc(t/T)*np.cos(pi*beta*t/T)/(1-(2*beta*t/T)**2)
        # fix case2 bug when t = pm T/(4*beta)
        case2[np.isinf(case2)] = 0
        h = np.where(np.abs(t) <= T/(2*beta), case1,case2)
    return h if not normlize else h*np.sqrt(T) #能量归一化



if __name__=="__main__":
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data',type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
    parser.add_argument('--save_path', type=str,default='./train.json', help="if not split the dataset, give a path to a json file")
    parser.add_argument('--random_split', action='store_true', help="random split the dataset, default ratio is 8:1:1")
    parser.add_argument('--split_by_file', action='store_true', help="define how to split the dataset, include ./train.txt ./val.txt ./test.txt ")

    snr_list = [-15,-10,-5,0,5,10,15,20]
    # generator = SignalDataGen(train=600,valid=400)
    # snr_list = [20]
    generator = SignalDataGen(train=500,valid=400)
    generator(snr_list=snr_list,addr='DataSet99')
    