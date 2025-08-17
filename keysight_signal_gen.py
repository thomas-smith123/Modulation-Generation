import matlab.engine
import numpy as np
import random,os,ctypes,json,argparse
from typing import List

eng = matlab.engine.start_matlab()
eng.cd(r'Program/iqtools', nargout=0)
# iqdata, newSampleRate, newNumSymbols, newNumSamples, chMapresult = eng.iqmod('sampleRate', 64e9, 'numSymbols', 60, 'data', 'Random', 'modType', "BPSK", 'oversampling', 64e3, 'filterType', 'Root Raised Cosine', 'filterNsym', 80.0, 'filterBeta', 0.35, 'carrierOffset', 0.0, 'quadErr', 0.0, 'iqskew', 0.0, 'gainImbalance', 0.0, 'correction', 0.0, 'function', 'download', 'channelMapping', [1,0],nargout=5)
# iqdata = np.array(iqdata)
# newSampleRate = np.array(newSampleRate)
# newNumSymbols = np.array(newNumSymbols)
# 输出结果
# print(iqdata)
## iqpulse 不用上升下降时间 数据类型的问题导致用不了
iqdata, marker, numRepeats, chMap= eng.iqpulse('sampleRate', 64e9, 'PRI', 4e-6, 'PW', 4e-6, 'pulseShape', 'Raised Cosine', 'span', 1e9, 'offset', 10e9, 'fmFormula', 'repelem((randi(2,30,1)-1)/(2-1),2000)','amplitude', 0, 'modulationType', 'User defined', 'correction', 0, 'channelMapping', [1,0], 'normalize', 1, nargout=4)
iqdata = np.array(iqdata)
print(iqdata)
# newSampleRate = np.array(newSampleRate)
# newNumSymbols = np.array(newNumSymbols)
eng.quit()

class keysight_signal_gen:
    ## 以下为使用matlab产生的信号种类
    digital_phase_modulate_class1: List[str] = [
        "BPSK",
        "BPSK_X",
        "QPSK",
        "OQPSK",
        "8-PSK",
        "QAM16",
        "QAM32",
        "QAM64",
        "QAM128",
        "QAM256",
        "QAM512",
        "QAM1024",
        "QAM2048",
        "QAM4096",
        "APSK16",
        "APSK32",
        "PAM4",
        "CPM",
        "QAM8",
        "OOK",
    ]
    pulse_modulate_class1: List[str] = [
        "none",
        "increasing",
        "decreasing",
        "v-shape",
        "inverted v",
        "barker-2 +-",
        "barker-2 ++",
        "barker-3",
        "barker-4 ++-+",
        "barker-4 +++-",
        "barker-5",
        "barker-7",
        "barker-11",
        "barker-13",
        "frank-4",
        "frank-6",
        "fmcw",
        ]
    pulse_modulate_class2: List[str] = [
        "2FSK",
        "costas",
        "frequency_hop",
        ]
    ## 以下为不需要使用matlab产生的信号种类
    others: List[str] = [
        "adsb",
        ]
    def __init__(self,args) -> None:
        ## system parameters
        self.all_modulation = self.digital_phase_modulate_class1+self.pulse_modulate_class1+self.pulse_modulate_class2
        self.fs = args.fs
        self.resample_fs = args.resample_fs
        self.max_signal_per_frame = args.max_signal_per_frame
        self.resolution = args.resolution
        ## detail parameters
        self.window_length = self.resolution[0]
        self.overlap = args.stft_overlap
        self.framePoints = (self.window_length-self.overlap)*(self.resolution[1]-1)+self.window_length # 一帧需要的点数
        ## others
        if os.path.exists(args.save_path):
            pass
        else:
            os.makedirs(args.save_path)
        ## start matlab engine
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(r'Program/iqtools', nargout=0)
    def __del__(self):
        ## stop matlab engine
        self.eng.quit()
    def __call__(self):
        self._get_frame_()
        pass
    def _get_frame_(self):
        ## 抽取调制方式
        num = np.random.randint(1,self.max_signal_per_frame)
        choosed_signal = random.sample(self.all_modulation,num)
        ## 开始产生
        for i in choosed_signal:
            i = "increasing"
            if i in self.digital_phase_modulate_class1: 
                ## 随机产生一些参数
                numSymbol = np.random.randint(5,30)
                signalFreq = np.random.randint(1e6,2.5e9,dtype=np.int64)*1.0
                symbolrate = np.random.randint(1,100)*1e6;oversampling = int(self.fs/symbolrate)*1.0
                position = np.random.randint(0,3) ## 0: 前不足，至最后; 1：全满; 2: 后不足，至最前
                ## 由此的参数计算
                bandwidth = (1+0.35)*symbolrate
                ## 产生信号
                iqdata, newSampleRate, newNumSymbols, newNumSamples, chMapresult = self.eng.iqmod('sampleRate', 64e9, 'numSymbols', numSymbol, 'data', 'Random', 'modType', i, 'oversampling', oversampling, 'filterType', 'Root Raised Cosine', 'filterNsym', 80.0, 'filterBeta', 0.35, 'carrierOffset', signalFreq, 'quadErr', 0.0, 'iqskew', 0.0, 'gainImbalance', 0.0, 'correction', 0.0, 'function', 'download', 'channelMapping', [1,0],nargout=5)
                iqdata = self.eng.resample(iqdata, self.resample_fs, self.fs, nargout=1) ## 这个是我们需要的
                iqdata = np.array(iqdata)
                ##
                startPoint = 0;endPoint = self.framePoints
                self.tf = np.zeros((self.resolution[0],self.resolution[1]),dtype=complex)
                if iqdata.shape[0] >= self.framePoints: ## 如果信号长度大于一帧的长度
                    if position == 0:
                        startPoint = np.random.randint(0,iqdata.shape[0]-self.framePoints) # 信号截取的位置
                        iqdata = iqdata[startPoint:self.framePoints+startPoint,:]
                        ## tf = stft(iqdata)
                        #@todo 缺stft
                        tf_offset = np.random.randint(int(0.1*self.resolution[0]),int(0.9*self.resolution[0]))
                        self.tf[-tf_offset:,:] = iqdata[:tf_offset,:]
                    elif position == 1:
                        startPoint = np.random.randint(0,iqdata.shape[0]-self.framePoints) # 信号截取的位置
                        iqdata = iqdata[startPoint:self.framePoints+startPoint,:]
                        ## tf = stft(iqdata)
                        self.tf = iqdata
                    else:
                        endPoint = np.random.randint(0,iqdata.shape[0]-self.framePoints) # 信号截取的位置
                        iqdata = iqdata[-endPoint:,:]
                        ## tf = stft(iqdata)
                        
                        tf_offset = np.random.randint(int(0.1*self.resolution[0]),int(0.9*self.resolution[0]))
                        self.tf[:tf_offset,:] = iqdata[-tf_offset:,:]
                else: ## 如果信号长度小于一帧的长度
                    tmp = np.floor((self.framePoints-iqdata.shape[0])/2)
                    startPoint = np.random.randint(0,tmp) # 信号截取的位置
                    self.tf[startPoint:startPoint+iqdata.shape[0],:] = iqdata
                    ## tf = stft(iqdata)
                ## 频率: signalFreq; 起始：startPoint; 终止：endPoint; 带宽：bandwidth; 调制方式：i
            elif i in self.pulse_modulate_class1: #@audit 这里可能有问题
                t = self.framePoints/self.resample_fs
                ## 随机产生一些参数
                centerFreq = np.random.randint(1e6,2.5e9,dtype=np.int64)*1.0
                spanFreq = np.random.randint(1e6,1e8,dtype=np.int64)*1.0
                
                pri = np.random.uniform(1e-6,5e-6)
                pw = np.random.uniform(1e-6,100e-6)
                if i in ["none","increasing","decreasing","v-shape","inverted v"]: 
                    repeat_ = np.random.randint(0,2) ## 0: 不重复; 1: 重复
                    iqdata, marker, numRepeats, chMap= self.eng.iqpulse('sampleRate', 64e9, 'PRI', pri, 'PW', pw, 'pulseShape', 'Raised Cosine', 'span', spanFreq, 'offset', centerFreq, 'amplitude', 0, 'modulationType', i, 'correction', 0, 'channelMapping', [1,0], 'normalize', 1, nargout=4)
                    iqdata = self.eng.resample(iqdata, self.resample_fs, self.fs, nargout=1) ## 这个是我们需要的
                    iqdata = np.array(iqdata)
                    
                    repeat_times = self.framePoints/iqdata.shape[0] ## 必须重复
                    iqdata = np.tile(iqdata,repeat_times)
                    
                    
                else: ## 脉冲相位调制
                    iqdata, marker, numRepeats, chMap= self.eng.iqpulse('sampleRate', 64e9, 'PRI', pri, 'PW', pw, 'pulseShape', 'Raised Cosine', 'span', 1e9, 'offset', centerFreq, 'amplitude', 0, 'modulationType', i, 'correction', 0, 'channelMapping', [1,0], 'normalize', 1, nargout=4)
                    pass

            elif i in self.pulse_modulate_class2:
                pri = np.random.uniform(1e-6,5e-6)
                centerFreq = np.random.randint(1e6,2.5e9,dtype=np.int64)*1.0
                spanFreq = np.random.randint(1e6,1e8,dtype=np.int64)*1.0
                if i == "2FSK":
                    ##@audit 注意带宽/码率
                    # bandwidth=spanFreq+symbolrate
                    repeat_rate = np.random.randint(10000,100000)*10
                    symbolrate = 64e9/repeat_rate ## 可能有问题
                    tmp_exp = 'repelem((randi(2,30,1)-1)/(2-1),{})'.format(repeat_rate)
                    iqdata, marker, numRepeats, chMap= self.eng.iqpulse('sampleRate', 64e9, 'PRI', pri, 'PW', pri, 'pulseShape', 'Raised Cosine', 'span', spanFreq, 'offset', centerFreq, 'fmFormula', tmp_exp,'amplitude', 0, 'modulationType', 'User defined', 'correction', 0, 'channelMapping', [1,0], 'normalize', 1, nargout=4)
                if i == "Frequency_hop":
                    ##@audit 注意带宽/码率
                    # bandwidth=spanFreq+symbolrate
                    repeat_rate = np.random.randint(5000,20000)*10
                    symbolrate = 64e9/repeat_rate ## 可能有问题
                    n = np.random.randint(3,6)
                    tmp_exp = 'repelem((randi({},30,1)-1)/({}-1),{}})'.format(n,n,repeat_rate)
                    iqdata, marker, numRepeats, chMap= self.eng.iqpulse('sampleRate', 64e9, 'PRI', pri, 'PW', pri, 'pulseShape', 'Raised Cosine', 'span', spanFreq, 'offset', centerFreq, 'fmFormula', tmp_exp,'amplitude', 0, 'modulationType', 'User defined', 'correction', 0, 'channelMapping', [1,0], 'normalize', 1, nargout=4)
                if i == "Costas":
                    n = np.random.randint(6,15)
                    repeat_rate = 1e4/n
                    symbolrate = 64e9/repeat_rate ## 可能有问题
                    tmp_exp = 'repelem(costas_sequence({})/{}},{}})'.format(n,n,repeat_rate)
                    iqdata, marker, numRepeats, chMap= self.eng.iqpulse('sampleRate', 64e9, 'PRI', pri, 'PW', pri, 'pulseShape', 'Raised Cosine', 'span', spanFreq, 'offset', centerFreq, 'fmFormula', tmp_exp,'amplitude', 0, 'modulationType', 'User defined', 'correction', 0, 'channelMapping', [1,0], 'normalize', 1, nargout=4)
                    pass
                iqdata = self.eng.resample(iqdata, self.resample_fs, self.fs, nargout=1)
            elif i in self.others:
                pass

    def digital_phase_modulate(self,modType:str):
        for i in self.digital_phase_modulate_class1:
            if modType == i:
                return True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs', default=64e9,type=float, help="awgn fs")
    parser.add_argument('--resample_fs', default=5e9,type=float, help="adc sample rate")
    parser.add_argument('--symbol_rate', type=float,default='1e6', help="symbol rate")
    # parser.add_argument('--numSymbols', default=60, type=int, help="random split the dataset, default ratio is 8:1:1")
    parser.add_argument('--carrierOffset', default=1e3, type=float, help="center frequency offset")
    parser.add_argument('--max_signal_per_frame', default=3,type=int, help="max signal per frame")
    parser.add_argument('--save_path', default='../dataset/',type=str, help="max signal per frame")
    parser.add_argument('--resolution', default=[512,512],type=list, help="max signal per frame")
    parser.add_argument('--stft_overlap', default=128,type=int, help="stft overlap")
    
    args = parser.parse_args()
    a = keysight_signal_gen(args)
    a()
    del a
    pass