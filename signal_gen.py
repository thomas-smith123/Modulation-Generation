from typing import Any
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
import commpy as cpy
import multiprocessing as mul
from PIL import Image
import pycuda.autoinit
import matplotlib.pyplot as plt
from pycuda import gpuarray
# import awg_gen

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
        # "Frank",# 3
        # "4FSK",# 2
        # "8FSK" , # 3
        "Frequency_HOP",# 4
        "QAM",# 4
        # "16QAM",# 5
        # "AM",# 6
        "DSB", # 5 #@audit DSB总是很淡
        "FM",# 6
        "PSK",# 7
        # "RadarPulse",# 8
        "RadarPulse_Compressed",# 9
        "NLFM",# 10
        "ADSB",# 11
        "Otherwise", #12
        # "fixed_signal",
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
        self.sample_rate = 64e9
        self.max_signal_frame = 2 #一帧最多5个信号
        self.train_num = train
        self.valid_num = valid
        self.num = self.train_num+self.valid_num
        ## 其他参数
        self.fft_size = fft_size #定义图像为正方形
        ### num_iq_samples = fft_size * fft_size
        self.noverlap = int(fft_size*0.75) #重叠点数
        self.overlap = fft_size-self.noverlap #步长
        self.nperseg = self.fft_size #窗长
        ### 时域长度：窗长+（点数-1）*（窗长-重叠）
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
    # def gen(self):
    #     for i in tqdm(range(self.train_num)):
    #         signal = self.gen_signal()
    #     pass
    #     for i in tqdm(range(self.valid_num)):
    #         signal = self.gen_signal()
    def checkFrequency(self,frequency_list,bandwidth_list,signal_parameter: SignalParameter):
        ## 中频和带宽
        ## 检查频率是否重叠
        for i,j in enumerate(frequency_list):
            if np.abs(signal_parameter.center_frequency-j)<(bandwidth_list[i]/2+np.abs(signal_parameter.bandwidth/2)): #进入到其他信号带宽了
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
        # 需要统计每个信号的频率啥的，避免重叠
        frequency_list = [] # 中心频率
        bandwidth_list = []
        description_list = []
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        num = np.random.randint(1,self.max_signal_frame) #该帧信号的数量
        if num == 0:
            return [self.iq_data,None] #@audit 不能这样返回，至少得返回一个全0的信号
        choosed_signal = random.sample(self.default_class,num) ## 抽取信号的调制样式
        for i in choosed_signal:
            continue_flag = np.random.uniform(0,1) # 0:不继续，1:继续
            ## 对FM族信号
            signal_parameter = SignalParameter(
                num_iq_samples=self.num_iq_samples,
                sample_rate=self.sample_rate
                )
            if continue_flag > 0:# 标记为连续信号
            # if continue_flag > 0.1:# 标记为连续信号
                signal_parameter.start = 0
                signal_parameter.stop = 1*signal_parameter.num_iq_samples/signal_parameter.sample_rate
                signal_parameter.duration = 1*signal_parameter.num_iq_samples/signal_parameter.sample_rate
            if i == "Frequency_HOP":
                signal_parameter.center_frequency = np.random.uniform(-0.3,0.3)*self.sample_rate/2
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                # signal_parameter.start = np.random.uniform(0,0.3)*self.num_iq_samples/self.sample_rate
                # signal_parameter.duration = np.random.uniform(0.3,0.7)*self.num_iq_samples/self.sample_rate
                signal_parameter.bandwidth = np.random.uniform(0.1,0.2)*self.sample_rate/2
                signal_parameter.num_symbol = np.random.randint(20,35)
                signal_parameter()
                signal = nFSK(signal_parameter,8)
            elif "FM" in i: #[FM LFM NLFM]
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                if i == "FM":
                    signal_parameter.bandwidth = 3e6
                    signal_parameter()
                    signal = FM(signal_parameter)
                elif i == "LFM":
                    signal = LFM(signal_parameter)
                    signal_parameter()
                elif i == "NLFM":
                    signal_parameter.bandwidth = 25e6
                    signal_parameter()
                    signal = NLFM(signal_parameter)
            elif i == "DSB":
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter.bandwidth = np.random.uniform(0.002,0.003)*self.sample_rate/2
                signal_parameter()
                signal = DSB(signal_parameter)
            elif i == "AM":
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter.bandwidth = np.random.uniform(0.01,0.1)*self.sample_rate/2
                signal_parameter.center_frequency = np.random.uniform(-0.3,0.3)*self.sample_rate # 限制一下带宽
                signal_parameter()
                signal = AM(signal_parameter)
            ## 对FSK族信号
            elif "FSK" in i: # nFSK
                n = int(i.split("FSK")[0])
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter.center_frequency = np.random.uniform(0.67,0.77)*self.sample_rate/2
                signal_parameter.bandwidth = np.random.uniform(0.01,0.05)*self.sample_rate/2
                signal_parameter.num_symbol = np.random.randint(10,40)
                # signal_parameter.duration = np.random.uniform(0.2, 0.8)*self.num_iq_samples*1/self.sample_rate
                signal_parameter()
                signal = nFSK(signal_parameter,n)
            ## 对QAM族信号
            elif "QAM" in i: # nQAM
                # n = int(i.split("QAM")[0])
                n = np.random.choice([16,64])
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = nQAM(signal_parameter,n)
            ## 对PSK族信号
            elif "PSK" in i:
                n = np.random.choice([2,4,8])
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = nPSK(signal_parameter,n)
            ## 对雷达信号
            elif "Radar" in i:
                if "Compressed" in i:
                    signal_parameter.num_symbol = np.random.randint(5,10)
                    signal = RADAR_Pulse_Compressed(signal_parameter,random.uniform(0.1,0.4))
                    # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                    signal_parameter()
                else:
                    signal_parameter.num_symbol = np.random.randint(5,15)
                    signal = RADAR_Pulse(signal_parameter)
                    # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                    signal_parameter()
            ## 其他信号
            elif i == "ADSB":
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                # signal_parameter.samples_per_symbol = np.random.randint(3000,6000)
                signal_parameter()
                signal = ADSB(signal_parameter)
            elif i == "Costas":
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter.bandwidth = np.random.uniform(0.05,0.15)*self.sample_rate/2/12.8
                signal_parameter()
                signal = Costas(signal_parameter)
            elif i == "Frank":
                signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = Frank(signal_parameter)
            elif i == "Otherwise":
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = OFDM(signal_parameter)
            elif i == "fixed_signal":
                # signal_parameter.center_frequency = self.checkFrequency(frequency_list,bandwidth_list,signal_parameter)
                signal_parameter()
                signal = FixedSignal(signal_parameter)

            frequency_list.append(signal_parameter.center_frequency)
            bandwidth_list.append(signal_parameter.bandwidth)
            signal() ## 生成信号
            description_list.append(signal.signal_description) # 信号的描述
            self.iq_data += signal.iq_data
        ##@todo 信号加噪
        # snr = -20#np.random.uniform(0,15)
        
        # self.i=np.zeros((self.iq_data.shape[0],2),dtype=np.float32)
        # self.q=np.zeros((self.iq_data.shape[0],2),dtype=np.float32)
        # self.i[:,0] = self.iq_data.real;np.savetxt('i.csv',self.i,delimiter=',')
        # self.q[:,0] = self.iq_data.imag;np.savetxt('q.csv',self.q,delimiter=',')

        
        
        self.iq_data.real = cpy.awgn(self.iq_data.real,snr_dB=snr)
        self.iq_data.imag = cpy.awgn(self.iq_data.imag,snr_dB=snr)
        return [self.iq_data,description_list]
    
    def __yolo_label_gen__(self,signal_sample:list,filename:str,seq:int,check:bool=False,test=True):## 主要看写txt有没有问题
        ## filename为路径？ ##@todo 想一下
        signal_ = signal_sample[0] ## 这个要转换成图片
        signal_description = signal_sample[1]
        f = open(filename+'labels/'+str(seq)+'.txt','w')
        
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
                ## 防止出现负数的情况
                if center_y>1:
                    center_y = center_y-1
                if height < 0:
                    height = -height
                    height += 16/self.fft_size
                else:
                    height += 16/self.fft_size
                
                if center_y+height/2 < 1:
                    height += 16/self.fft_size
                    if center_y+height/2 > 1:
                        height = (1-center_y)*2
                    if center_y-height/2 < 0: ##@audit 加上这个应该没什么问题了
                        height = center_y*2
                
                if center_x+width/2 < 1:
                    width += 10/self.fft_size
                    if center_x+width/2 > 1:
                        width = (1-center_x)*2
                    if center_x-width/2 < 0: ##@audit 加上这个应该没什么问题了
                        width = center_x*2
                f.writelines(str(label)+' '+str(center_x)+' '+str(center_y)+' '+str(width)+' '+str(height)+'\n')
                
                if check:
                    a,b,c = ConvertDescriptionToPatch(description,nfft=self.fft_size)
                    test_Rectangle1 = mpl.patches.Rectangle(a, width=b, height=c, angle=0, fill=False)
                    ax.add_patch(test_Rectangle1)
                    ax.text(a[0],a[1],description.class_name,fontdict=font)
        else:
            f.close()
        f.close()
        data_t_gpu  = gpuarray.to_gpu(signal_.astype(np.complex64))
        np.save(filename+'raw_complex/'+str(seq),signal_)
        cf.cufftExecC2C(self.plan, int(data_t_gpu.gpudata), int(self.data_o_gpu.gpudata), cf.CUFFT_FORWARD)
        tf = fftpack.fftshift(self.data_o_gpu.get(),1)
        # stft = Spectrogram(nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.fft_size)
        # tf=stft(signal_)
        if check:
            plt.imshow(np.abs(tf.T),cmap='viridis')
            # plt.pcolormesh(np.abs(tf),shading='gouraud')
            plt.savefig(filename+'check/'+str(seq)+'.png')
            plt.close()
        # np.save(filename+'raw_complex/'+str(seq),tf.T)
        heatmapshow = NormMinandMax(np.abs(tf.T)) #再保存一波原始的复数矩阵
        heatmapshow = applyColorMap(heatmapshow,cmaps[random.randint(0,len(cmaps)-1)])
        heatmapshow.save(filename+'images/'+str(seq)+'.png')
        if not test:
            awg_gen.make_file(filename+'awg_file/'+str(seq)+'.csv',signal_.astype(np.complex64),self.sample_rate)
        else:
            awg_gen.make_file(filename+'awg_file/'+str(seq)+'_'+str(description.center_frequency*self.sample_rate/1e9)+'.csv',signal_.astype(np.complex64),self.sample_rate)
        # heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        
        # cv2.imwrite(filename+'images/'+str(seq)+'.jpg',heatmapshow)
        
    def __yolo_noise_gen__(self,seq:int,filename:str,snr:float):
        signal_ = np.zeros(self.num_iq_samples,dtype=np.complex64)
        signal_.real = np.random.randn(self.num_iq_samples)*(np.sqrt(np.power(10,-snr/10)*0.001)/50)
        signal_.imag = np.random.randn(self.num_iq_samples)*(np.sqrt(np.power(10,-snr/10)*0.001)/50)
        
        data_t_gpu  = gpuarray.to_gpu(signal_.astype(np.complex64))
        cf.cufftExecC2C(self.plan, int(data_t_gpu.gpudata), int(self.data_o_gpu.gpudata), cf.CUFFT_FORWARD)
        tf = fftpack.fftshift(self.data_o_gpu.get(),1)

        heatmapshow = NormMinandMax(np.abs(tf.T))
        heatmapshow = applyColorMap(heatmapshow,cmaps[random.randint(0,len(cmaps)-1)])
        heatmapshow.save(filename+'images/'+str(seq)+'.png')
    
    def __call__(self, snr_list:list,addr:str='DataSet', *args: Any, **kwds: Any) -> Any:
        if not os.path.exists(addr+'/check/'):
            os.mkdir(addr+'/check/')
        if not os.path.exists(addr+'/images/'):
            os.mkdir(addr+'/images/')
        if not os.path.exists(addr+'/labels/'):
            os.mkdir(addr+'/labels/')
        if not os.path.exists(addr+'/awg_file/'):
            os.mkdir(addr+'/awg_file/')
        if not os.path.exists(addr+'/raw_complex/'):
            os.mkdir(addr+'/raw_complex/')
        cnt = 0
        # pool = mul.Pool(len(snr_list))
        # pool.map(self.__yolo_label_gen__,self.genFrame(snr),filename=addr+'/',seq=i+cnt,check=True)
        for snr in snr_list:
            pbar = tqdm(total=self.num,postfix='SNR:'+str(snr))
            for i in range(self.num):
                self.__yolo_label_gen__(self.genFrame(snr),filename=addr+'/',seq=i+cnt,check=True)
                pbar.update(1)
            cnt += self.num
            for i in range(self.num):
                self.__yolo_noise_gen__(seq=cnt+i,filename=addr+'/',snr=snr)
            cnt += self.num
            ####划分训练集测试集和验证集
        train_list = np.random.choice(np.arange(0,cnt),int(0.7*cnt),replace=False)
        valid_list = np.random.choice(np.array(list(set(np.arange(0,cnt))-set(train_list))),int(0.15*cnt),replace=False)
        test_list = np.array(list(set(np.arange(0,cnt))-set(train_list)-set(valid_list)))
        
        f = open(addr+'/train.txt','w')
        for i in train_list:
            f.writelines('./images/'+str(i)+'.png'+'\n')
        f.close()
        f = open(addr+'/valid.txt','w')
        for i in valid_list:
            f.writelines('./images/'+str(i)+'.png'+'\n')
        f.close()
        f = open(addr+'/test.txt','w')
        for i in test_list:
            f.writelines('./images/'+str(i)+'.png'+'\n')
        f.close()
        print('All modulations:');print(self.default_class)
        # self.yolo2coco(addr)
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
                label_file = open(label_path+line.split('\n')[0].split('.')[-2].replace('images','labels')+'.txt','r')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data',type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
    parser.add_argument('--save_path', type=str,default='./train.json', help="if not split the dataset, give a path to a json file")
    parser.add_argument('--random_split', action='store_true', help="random split the dataset, default ratio is 8:1:1")
    parser.add_argument('--split_by_file', action='store_true', help="define how to split the dataset, include ./train.txt ./val.txt ./test.txt ")

    snr_list = [10,15,20]
    # snr_list = [45]
    generator = SignalDataGen(train=20,valid=0)
    generator(snr_list=snr_list,addr='DataSet_test')
    