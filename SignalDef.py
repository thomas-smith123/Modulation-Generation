from typing import Any, List, Optional, Union
from SignalDescription import SignalDescription
from SignalDescription import SignalData
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter, freqz
import commpy as cpy ## 省事写法
from commpy.filters import rcosfilter,rrcosfilter

class SignalParameter:
## 定义信号参数，采样率归一化
## fs为1，频率就在-0.5到0.5之间
## 有正负频率
## 时间片长度定为1
    def __init__(self,num_iq_samples:int=512*512,sample_rate=None) -> None:
        self.sample_rate = sample_rate
        self.num_iq_samples = int(num_iq_samples)#np.random.randint(0, 512*512) ##@todo 要有torchsig的效果的话就要改stft的时窗
        self.bandwidth = 20e6#np.random.uniform(-0.2, 0.2)*sample_rate
        
        self.center_frequency = np.random.uniform(-0.4, 0.4)*sample_rate
        self.lower_frequency = self.center_frequency-self.bandwidth/2
        self.upper_frequency = self.lower_frequency + self.bandwidth/2
        
        self.start = np.random.uniform(0.0, 0.8)*num_iq_samples*1/sample_rate # 绝对时间
        self.duration = np.random.uniform(0.1, 0.8)*num_iq_samples*1/sample_rate
        if self.start+self.duration>num_iq_samples/sample_rate:
            self.stop = num_iq_samples/sample_rate
        else:
            self.stop = self.start+self.duration
        
        if self.center_frequency+self.bandwidth/2>sample_rate/2:
            self.center_frequency = sample_rate/2-self.bandwidth/2
            self.lower_frequency = self.center_frequency-self.bandwidth/2
            self.upper_frequency = self.lower_frequency + self.bandwidth
        elif self.center_frequency-self.bandwidth/2<-sample_rate/2:
            self.center_frequency = -sample_rate/2+self.bandwidth/2
            self.lower_frequency = self.center_frequency-self.bandwidth/2
            self.upper_frequency = self.lower_frequency + self.bandwidth    
        
        # self.stop = num_iq_samples/sample_rate if (self.start+self.duration)>=num_iq_samples/sample_rate else (self.start+self.duration)

        # self.samples_per_symbol = np.random.randint(0, 4096)#np.random.uniform(0.0, 1.0)
        # self.excess_bandwidth = np.random.uniform(0.0, 1.0)
        self.num_symbol = np.random.randint(20, 40)
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.start+self.duration>self.num_iq_samples/self.sample_rate:
            self.stop = self.num_iq_samples/self.sample_rate
        else:
            self.stop = self.start+self.duration
        
        if self.center_frequency+self.bandwidth/2>self.sample_rate/2:
            self.center_frequency = self.sample_rate/2-self.bandwidth/2
            self.lower_frequency = self.center_frequency-self.bandwidth/2
            self.upper_frequency = self.lower_frequency + self.bandwidth
        elif self.center_frequency-self.bandwidth/2<-self.sample_rate/2:
            self.center_frequency = -self.sample_rate/2+self.bandwidth/2
            self.lower_frequency = self.center_frequency-self.bandwidth/2
            self.upper_frequency = self.lower_frequency + self.bandwidth 
        #依赖于计算的项单独拉出来搞
        self.upper_frequency = self.center_frequency + self.bandwidth/2
        self.lower_frequency = self.center_frequency - self.bandwidth/2
        self.stop = self.num_iq_samples/self.sample_rate if self.start+self.duration>=self.num_iq_samples/self.sample_rate else self.start+self.duration
        # self.stop = 1.0 if self.start+self.duration>=1 else self.start+self.duration

class BaseSignal:
    def __init__(self,signalparameter: SignalParameter,class_name) -> None:
        self.bandwidth = np.abs(signalparameter.bandwidth)
        self.center_frequency = signalparameter.center_frequency
        self.lower_frequency = signalparameter.lower_frequency
        self.sample_rate = signalparameter.sample_rate
        self.upper_frequency = signalparameter.upper_frequency
        self.num_iq_samples = signalparameter.num_iq_samples
        self.num_symbols = signalparameter.num_symbol
        self.start = signalparameter.start # 绝对时间
        self.stop = signalparameter.stop
        self.signal_description = SignalDescription(
            sample_rate=signalparameter.sample_rate,
            bandwidth=np.abs(signalparameter.bandwidth)/signalparameter.sample_rate,
            center_frequency=signalparameter.center_frequency/(signalparameter.sample_rate),
            upper_frequency=signalparameter.upper_frequency/(signalparameter.sample_rate),
            lower_frequency=signalparameter.lower_frequency/(signalparameter.sample_rate),
            start=signalparameter.start/(signalparameter.num_iq_samples/signalparameter.sample_rate),
            stop=signalparameter.stop/(signalparameter.num_iq_samples/signalparameter.sample_rate),
            class_name=class_name,
        )
class FixedSignal(BaseSignal):
    def __init__(self,signalparameter: SignalParameter) -> None:
        super().__init__(signalparameter,'fixed_signal')

    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        # time = np.linspace(0,self.num_iq_samples,self.num_iq_samples) ## 总时间
        
        # time_ = time[int(np.floor(self.signal_description.start*self.num_iq_samples)):int(np.floor(self.signal_description.stop*self.num_iq_samples))]
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        iq = np.exp(1j*np.pi*2*time_tmp*self.center_frequency) ##@audit 全部给成归一化的幅值
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq
        # return iq,self.signal_description

# 
# lower_frequency: 最小频率
# upper_frequency: 最大频率
# duration: 信号持续时间
# num_iq_samples: IQ采样点数
class LFM(BaseSignal):
    def __init__(self,signalparameter:SignalParameter) -> None:
        super().__init__(signalparameter,'LFM')

    # def LFM(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        skew = (self.upper_frequency-self.lower_frequency)/(self.stop-self.start)
        time_tmp = np.linspace(0,(self.stop-self.start),int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        iq = np.exp(1j*np.pi*time_tmp*(time_tmp*skew+2*self.lower_frequency)) ##@audit 全部给成归一化的幅值
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq
        del iq,time_tmp
        # return iq,self.signal_description

class NLFM(BaseSignal):
    ##@audit 信号框选不准
    def __init__(self,signalparameter:SignalParameter) -> None:
        super().__init__(signalparameter,'NLFM')

    # def NLFM(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        # time = np.linspace(0,self.num_iq_samples,self.num_iq_samples) ## 总时间
        
        skew = (self.upper_frequency-self.lower_frequency)/(1.5*np.power(-self.start+self.stop,2))
        
        # time_ = time[int(np.floor(self.signal_description.start*self.num_iq_samples)):int(np.floor(self.signal_description.stop*self.num_iq_samples))]
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        iq = np.exp(1j*2*np.pi*time_tmp*(np.power(time_tmp,2)*skew+2*self.lower_frequency)) ##@audit 全部给成归一化的幅值
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq
        # return iq,self.signal_description

class nFSK(BaseSignal):
    ## 2FSK和4FSK都可以用
    def __init__(self,signalparameter:SignalParameter,n:int=2) -> None:
        super().__init__(signalparameter,str(n)+'FSK')
        if n%2 != 0:
            raise ValueError('nFSK的n必须是偶数')
        self.status = n
        self.signal_description.class_name = str(n)+'FSK' if n == 2 else 'Frequency_HOP'

        
    # def nFSK(self):
    def __call__(self):
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        self.iq_samples_per_symbol = int(len(time_tmp)//self.num_symbols) ## start和stop的时间要重新划分
        
        self.stop = self.iq_samples_per_symbol*self.num_symbols/self.sample_rate+self.start
        self.signal_description.stop = self.iq_samples_per_symbol*self.num_symbols/self.num_iq_samples+self.signal_description.start
        
        
        
        bandwidth_div = self.bandwidth/(self.status-1)
        source = np.random.choice(np.arange(0,self.status),self.num_symbols,replace=True) #@todo 这里的随机抽取可能要改一下，不是很均匀
        # print(source)
        source = np.repeat(source,self.iq_samples_per_symbol) #@audit 不对
        time_tmp = np.linspace(0,self.stop-self.start,
                               len(source))
        # time_tmp = time_tmp[0:source.shape[0]]
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        iq = np.exp(1j*2*np.pi*(self.lower_frequency+bandwidth_div*source)*time_tmp)
        self.iq_data[int(np.floor(self.start*self.sample_rate)):int(np.floor(self.start*self.sample_rate))+len(source)] = iq

class AM:
    def __init__(self,signalparameter: SignalParameter,modulate_coefficient=1) -> None:
        self.a = modulate_coefficient
        self.lower_frequency = signalparameter.lower_frequency
        self.sample_rate = signalparameter.sample_rate
        self.start = signalparameter.start # 绝对时间
        self.stop = signalparameter.stop
        self.num_iq_samples = signalparameter.num_iq_samples
        self.center_frequency = signalparameter.center_frequency
        self.bandwidth = np.abs(signalparameter.bandwidth)
        self.signal_description = SignalDescription(
            bandwidth=(signalparameter.bandwidth),
            center_frequency=signalparameter.center_frequency,
            upper_frequency=(self.center_frequency+np.abs(signalparameter.bandwidth)/2)/(signalparameter.sample_rate),
            lower_frequency=(self.center_frequency-np.abs(signalparameter.bandwidth)/2)/(signalparameter.sample_rate),
            start=signalparameter.start/(signalparameter.num_iq_samples/signalparameter.sample_rate),
            stop=signalparameter.stop/(signalparameter.num_iq_samples/signalparameter.sample_rate),
            class_name='AM',
        )
    # def AM(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        time_tmp = np.linspace(0,(self.stop-self.start),
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        ## 把噪声调制上去
        ##@audit source得整个查找表
        tmp = np.random.randn(len(time_tmp))
        tmp = butter_lowpass_filter(tmp,self.bandwidth,self.sample_rate)
        source = tmp#np.exp(1j*2*np.pi*np.cumsum(tmp)*self.a)*np.sin(2*np.pi*time_tmp*self.bandwidth/2)#5*np.sin(time_tmp*2*np.pi)*np.sin(2*np.pi*time_tmp*self.bandwidth/2)#(np.random.randn(len(time_tmp))+1j*np.random.randn(len(time_tmp)))
        # source = np.random.randn(len(time_tmp))
        iq = np.exp(1j*2*np.pi*self.center_frequency*time_tmp)*source+self.a*np.exp(1j*2*np.pi*self.center_frequency*time_tmp) # 带宽没法确定
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq

class DSB(BaseSignal): 
    def __init__(self,signalparameter: SignalParameter,modulate_coefficient=0.0001) -> None:
        super().__init__(signalparameter,'DSB')
        
        self.a = modulate_coefficient
        self.bandwidth = 0.001*self.sample_rate/12.8

    # def DSB(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        time_tmp = np.linspace(0,(self.stop-self.start),
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        ## 把噪声调制上去
        ##@audit source得整个查找表
        tmp = np.random.randn(len(time_tmp))
        tmp = butter_lowpass_filter(tmp,2*self.bandwidth,self.sample_rate)
        source = tmp#*np.sin(2*np.pi*time_tmp*self.bandwidth/4)#np.exp(1j*2*np.pi*np.cumsum(tmp)*self.a)*np.sin(2*np.pi*time_tmp*self.bandwidth/4)#5*np.sin(time_tmp*2*np.pi)*np.sin(2*np.pi*time_tmp*self.bandwidth/2)#(np.random.randn(len(time_tmp))+1j*np.random.randn(len(time_tmp)))
        # source = np.random.randn(len(time_tmp))
        iq = np.exp(1j*2*np.pi*self.center_frequency*time_tmp)*source
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq*10

class FM(BaseSignal): 
    #@todo 需要整成窄带的
    def __init__(self,signalparameter: SignalParameter,modulate_coefficient=0.03) -> None:
        super().__init__(signalparameter,'FM')
        self.a = modulate_coefficient
        
    # def FM(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        time_tmp = np.linspace(0,(self.stop-self.start),
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        source = np.random.randn(len(time_tmp))#(np.random.randn(len(time_tmp))+1j*np.random.randn(len(time_tmp)))
        source = butter_lowpass_filter(source,self.bandwidth,self.sample_rate)
        #@audit 根据结果进行调节
        #+
        # source = np.random.randn(len(time_tmp))
        iq = np.exp(1j*2*np.pi*(self.center_frequency)*time_tmp)*np.exp(1j*2*np.pi*np.cumsum(source)*self.a)
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq


class OFDM(BaseSignal):
    #@audit 这个不知道怎么用
    def __init__(self,signalparameter: SignalParameter) -> None:
        super().__init__(signalparameter,'Otherwise')
        self.source_path = '20_QPSK_OFDM.mat'
        self.signal_description.lower_frequency = self.lower_frequency/self.sample_rate

        self.bandwidth = float(self.source_path.split('_')[0])*1e6
        self.signal_description.bandwidth = self.bandwidth/self.sample_rate
        self.center_frequency = np.random.uniform(-0.4, 0.4)*self.sample_rate
        self.signal_description.center_frequency = self.center_frequency/self.sample_rate
        self.upper_frequency = self.center_frequency+self.bandwidth/2
        self.lower_frequency = self.center_frequency-self.bandwidth/2
        self.signal_description.upper_frequency = self.upper_frequency/self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency/self.sample_rate
        self.data = sio.loadmat(self.source_path)
        self.data = self.data[list(self.data.keys())[-1]]
        self.start_point = np.random.randint(0,(self.data.shape[0]-8e6))
        
        pass
    def __call__(self):
        iq = self.data[self.start_point:self.start_point+int(np.floor(self.stop*self.sample_rate))-int(np.floor(self.start*self.sample_rate))].T*np.exp(1j*2*np.pi*self.center_frequency*np.linspace(0,self.stop-self.start,int(np.floor(self.stop*self.sample_rate))-int(np.floor(self.start*self.sample_rate))))
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+iq.shape[1])] = iq
        # self.iq
    
        pass
    # def OFDM(self):
    #     source = np.random.choice(np.arange(0,self.status),self.num_symbols,replace=True) #@todo 这里的随机抽取可能要改一下，不是很均匀
    #     print(source)
    #     source_ = decimal_to_binary(source,int(np.log2(self.status))) ##@todo 长度要再确定下
    #     del source
    #     # source = np.random.binomial(n=1,p=0.5,size=(128))
    #     nQAM_=cpy.QAMModem(self.status)
    #     source = nQAM_.modulate(source_)
    #     self.ofdm_transmitter(source)
        
    # def ofdm_transmitter(self,symbols, fft_size=512, cp_size=16, vc_size=16):
    #     N_used = (fft_size-vc_size)
    #     # 计算OFDM符号数
    #     num_symbols = len(symbols) // N_used
    #     # 将星座点重新分组为OFDM符号
    #     symbols_grouped = np.reshape(symbols[:num_symbols*N_used], (num_symbols, N_used))
    #     #添加VC
    #     symbols_grouped_vc = np.concatenate((np.zeros((num_symbols, vc_size)), symbols_grouped), axis=1)
    #     # 进行FFT变换
    #     freq_symbols = np.fft.ifft(symbols_grouped_vc, axis=1)
    #     # 添加循环前缀
    #     freq_symbols_cp = np.concatenate((freq_symbols[:, -cp_size:], freq_symbols), axis=1)
    #     # 将OFDM符号串联起来
    #     time_signal = freq_symbols_cp.flatten()
    #     return time_signal

class RADAR_Pulse(BaseSignal):
    ## 限制信号必须一直持续
    ## 信号等间隔产生
    ## 点频信号不需要考虑带宽
    ## 脉宽和PRF自己算吧
    def __init__(self,signalparameter: SignalParameter,DutyRate:float=0.2) -> None:
        super().__init__(signalparameter,'RadarPulse')
        self.DutyRate = DutyRate
        self.start = 0
        self.stop = self.num_iq_samples/self.sample_rate
        self.signal_description.bandwidth=0
        self.signal_description.start=0
        self.signal_description.upper_frequency = self.signal_description.center_frequency
        self.signal_description.lower_frequency = self.signal_description.center_frequency
        self.signal_description.stop = 1

    # def Radar_Pulse(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        cnt = 0
        ## 时间要重新划分
        time_chip_point = self.num_iq_samples//self.num_symbols/self.sample_rate
        time_tmp = np.linspace(0,time_chip_point*self.DutyRate,int(time_chip_point*self.DutyRate*self.sample_rate))
        source = np.exp(1j*2*np.pi*self.center_frequency*time_tmp)
        for i in range(self.num_symbols):
            self.iq_data[cnt:cnt+len(time_tmp)] = source
            cnt += int(time_chip_point*self.sample_rate)
        self.signal_description.stop = 1-(time_chip_point*(1-self.DutyRate))*self.sample_rate/len(self.iq_data)# 就是持续时间
        
        roll_ = np.random.randint(0,int((time_chip_point*(1-self.DutyRate))*self.sample_rate))
        self.iq_data = array_rotation(self.iq_data,roll_)
        self.signal_description.start+=roll_/len(self.iq_data)
        self.signal_description.stop+=roll_/len(self.iq_data)

class RADAR_Pulse_Compressed(BaseSignal):
    ## 限制信号必须一直持续
    ## 信号等间隔产生
    ## 点频信号不需要考虑带宽
    ## 脉宽和PRF自己算吧
    def __init__(self,signalparameter: SignalParameter,DutyRate:float=0.2) -> None:
        super().__init__(signalparameter,'RadarPulse_Compressed')
        self.start = 0
        self.stop = self.num_iq_samples/self.sample_rate
        self.signal_description.start=0
        self.signal_description.stop = 1
        self.DutyRate = DutyRate

    # def Radar_Pulse(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        cnt = 0
        ## 时间要重新划分
        time_chip_point = self.num_iq_samples//self.num_symbols*(1/self.signal_description.sample_rate) ## 绝对时间
        skew = (self.upper_frequency-self.lower_frequency)/(time_chip_point*self.DutyRate)
        ## 时间不对
        time_tmp = np.linspace(0,(time_chip_point*self.DutyRate),int(self.signal_description.sample_rate*time_chip_point*self.DutyRate))
        source = np.exp(1j*np.pi*time_tmp*(time_tmp*skew+2*self.lower_frequency))
        for i in range(self.num_symbols):
            self.iq_data[cnt:cnt+len(time_tmp)] = source
            cnt += int(time_chip_point*self.signal_description.sample_rate)
        self.signal_description.stop = 1-(time_chip_point*(1-self.DutyRate))*self.sample_rate/len(self.iq_data)# 就是持续时间
        
        roll_ = np.random.randint(0,int((time_chip_point*(1-self.DutyRate))*self.sample_rate))
        self.iq_data = array_rotation(self.iq_data,roll_)
        self.signal_description.start+=roll_/len(self.iq_data)
        self.signal_description.stop+=roll_/len(self.iq_data)
        
class nPSK(BaseSignal):
    ## 16QAM和64QAM应该都可以用
    def __init__(self,signalparameter:SignalParameter,n:int=2) -> None:
        super().__init__(signalparameter,'PSK')
        # 这个对n应该没那么多要求吧
        self.status = n
        self.bandwidth = 0
        self.signal_description.bandwidth = 0
        self.upper_frequency = self.center_frequency
        self.lower_frequency = self.center_frequency
        self.signal_description.upper_frequency = self.signal_description.center_frequency
        self.signal_description.lower_frequency = self.signal_description.center_frequency

        
    # def nPSK(self):
    def __call__(self):
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        self.iq_samples_per_symbol = int(len(time_tmp)//self.num_symbols)
        self.stop = self.iq_samples_per_symbol*self.num_symbols/self.sample_rate+self.start
        self.signal_description.stop = self.iq_samples_per_symbol*self.num_symbols/self.num_iq_samples+self.signal_description.start
        # bandwidth_div = self.bandwidth/(self.status-1)
        source = np.random.choice(np.arange(0,self.status),self.num_symbols,replace=True) #@todo 这里的随机抽取可能要改一下，不是很均匀
        # print(source)
        source_ = decimal_to_binary(source,int(np.log2(self.status))) ##@todo 长度要再确定下
        del source
        
        # source = np.random.binomial(n=1,p=0.5,size=(128))
        nPSK_=cpy.PSKModem(self.status)
        source = nPSK_.modulate(source_)
        source = np.repeat(source,self.iq_samples_per_symbol) # 时间上以这个长度为准
        time_tmp = time_tmp[0:source.shape[0]]
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        iq = np.exp(1j*2*np.pi*self.center_frequency*time_tmp)*source ##@audit QAM调制是不是这样的？
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq

class nPSK_(BaseSignal):
    ## 16QAM和64QAM应该都可以用
    def __init__(self,signalparameter:SignalParameter,n:int=2) -> None:
        super().__init__(signalparameter,'PSK')
        # 这个对n应该没那么多要求吧
        self.status = n
        self.bandwidth = 0
        self.signal_description.bandwidth = 0
        
        self.Rb = self.num_symbols/(self.stop-self.start) ## 比特速率
        self.Rs = self.Rb*np.log2(self.status) ## 符号速率
        self.Sps = int(self.sample_rate/self.Rs) ## 每个符号的采样点数
        
        self.upper_frequency = self.center_frequency+self.Rb/2
        self.lower_frequency = self.center_frequency-self.Rb/2
        self.signal_description.upper_frequency = self.signal_description.center_frequency+self.Rb/2/self.sample_rate
        self.signal_description.lower_frequency = self.signal_description.center_frequency-self.Rb/2/self.sample_rate
        
        
        
    # def nPSK(self):
    def __call__(self):
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        self.iq_samples_per_symbol = int(len(time_tmp)//self.num_symbols)
        self.stop = self.iq_samples_per_symbol*self.num_symbols/self.sample_rate+self.start
        self.signal_description.stop = self.iq_samples_per_symbol*self.num_symbols/self.num_iq_samples+self.signal_description.start
        # bandwidth_div = self.bandwidth/(self.status-1)
        source = np.random.choice(np.arange(0,self.status),self.num_symbols,replace=True) #@todo 这里的随机抽取可能要改一下，不是很均匀
        
        # print(source)
        source_ = decimal_to_binary(source,int(np.log2(self.status))) ##@todo 长度要再确定下
        del source
        
        # source = np.random.binomial(n=1,p=0.5,size=(128))
        nPSK__=cpy.PSKModem(self.status)
        source = nPSK__.modulate(source_)
        ## upsample
        tmp = np.zeros(len(source)*self.iq_samples_per_symbol,dtype = np.complex64)
        tmp[::self.iq_samples_per_symbol] = source
        
        sPSF = rcosfilter(len(source_), alpha=0.8, Ts=(self.stop-self.start)/len(source_), Fs=self.sample_rate)[1]
        source = np.convolve(sPSF,tmp, mode='same')
        # source = np.repeat(source,self.iq_samples_per_symbol) # 时间上以这个长度为准
        time_tmp = time_tmp[0:source.shape[0]]
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        iq = np.exp(1j*2*np.pi*self.center_frequency*time_tmp)*source ##@audit QAM调制是不是这样的？
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq
        pass
def decimal_to_binary(decimal_array, bit_length):## 将十进制的数组转为二进制数组
    binary_array = []
    array_ = []
    for decimal in decimal_array:
        binary = bin(decimal)[2:]
        binary = binary.zfill(bit_length)
        binary_array.append(binary)
    tmp = ''.join(binary_array)
    del binary_array
    array = tmp.split(',')
    for i in array[0]:
        if i != '':
            array_.append(int(i))

    return array_

class nQAM(BaseSignal):
    ## 16QAM和64QAM应该都可以用
    def __init__(self,signalparameter:SignalParameter,n:int=4) -> None:
        super().__init__(signalparameter,'QAM')
        if np.log2(n) != int(np.log2(n)):
            raise ValueError('nQAM的n必须是2的幂次')
        self.status = n
        self.bandwidth = 0
        self.signal_description.bandwidth = 0
        self.upper_frequency = self.center_frequency
        self.lower_frequency = self.center_frequency
        self.signal_description.upper_frequency = self.signal_description.center_frequency
        self.signal_description.lower_frequency = self.signal_description.center_frequency
        
    # def nQAM(self):
    def __call__(self):
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        self.iq_samples_per_symbol = int(len(time_tmp)//self.num_symbols)
        self.stop = self.iq_samples_per_symbol*self.num_symbols/self.sample_rate+self.start
        self.signal_description.stop = self.iq_samples_per_symbol*self.num_symbols/self.num_iq_samples+self.signal_description.start
        # bandwidth_div = self.bandwidth/(self.status-1)
        source = np.random.choice(np.arange(0,self.status),self.num_symbols,replace=True) #@todo 这里的随机抽取可能要改一下，不是很均匀
        # print(source)
        source_ = decimal_to_binary(source,int(np.log2(self.status))) ##@todo 长度要再确定下
        del source
        # source = np.random.binomial(n=1,p=0.5,size=(128))
        nQAM_=cpy.QAMModem(self.status)
        source = nQAM_.modulate(source_)
        source = np.repeat(source,self.iq_samples_per_symbol) # 时间上以这个长度为准
        time_tmp = time_tmp[0:source.shape[0]]
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        ## QAM调制能幅度直接乘吗？好像可以
        iq = np.exp(1j*2*np.pi*self.center_frequency*time_tmp)*source ##@audit QAM调制是不是这样的？
        ##@audit 这里的幅度是不是要归一化？
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq/np.sqrt(self.status/2)
        pass
    
class nQAM_(BaseSignal):
    ## 16QAM和64QAM应该都可以用
    def __init__(self,signalparameter:SignalParameter,n:int=4) -> None:
        super().__init__(signalparameter,'QAM')
        if np.log2(n) != int(np.log2(n)):
            raise ValueError('nQAM的n必须是2的幂次')
        self.status = n
        self.bandwidth = 0
        self.signal_description.bandwidth = 0
        
        self.Rb = self.num_symbols/(self.stop-self.start) ## 比特速率
        self.Rs = self.Rb*np.log2(self.status) ## 符号速率
        self.Sps = int(self.sample_rate/self.Rs) ## 每个符号的采样点数
        
        self.upper_frequency = self.center_frequency+self.Rb/2
        self.lower_frequency = self.center_frequency-self.Rb/2
        self.signal_description.upper_frequency = self.signal_description.center_frequency+self.Rb/2/self.sample_rate
        self.signal_description.lower_frequency = self.signal_description.center_frequency-self.Rb/2/self.sample_rate
        
    # def nQAM(self):
    def __call__(self):
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        self.iq_samples_per_symbol = int(len(time_tmp)//self.num_symbols)
        self.stop = self.iq_samples_per_symbol*self.num_symbols/self.sample_rate+self.start
        self.signal_description.stop = self.iq_samples_per_symbol*self.num_symbols/self.num_iq_samples+self.signal_description.start
        # bandwidth_div = self.bandwidth/(self.status-1)
        source = np.random.choice(np.arange(0,self.status),self.num_symbols,replace=True) #@todo 这里的随机抽取可能要改一下，不是很均匀
        # print(source)
        source_ = decimal_to_binary(source,int(np.log2(self.status))) ##@todo 长度要再确定下
        del source
        # source = np.random.binomial(n=1,p=0.5,size=(128))
        nQAM_=cpy.QAMModem(self.status)
        source = nQAM_.modulate(source_)/(np.sqrt(self.status)-1)
        ## upsample
        tmp = np.zeros(len(source)*self.iq_samples_per_symbol,dtype = np.complex64)
        tmp[::self.iq_samples_per_symbol] = source
        sPSF = rrcosfilter(len(source_), alpha=0.8, Ts=(self.stop-self.start)/len(source_), Fs=self.sample_rate)[1]
        source = np.convolve(sPSF,tmp, mode='same')
        
        time_tmp = time_tmp[0:source.shape[0]]
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        ## QAM调制能幅度直接乘吗？好像可以
        iq = np.exp(1j*2*np.pi*self.center_frequency*time_tmp)*source ##@audit QAM调制是不是这样的？
        ##@audit 这里的幅度是不是要归一化？
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq/np.sqrt(self.status/2)
        pass

class ADSB(BaseSignal): ###@audit 有问题
    OrderedDict = {
        'DF': 5,
        'CA': 3,
        'ICAO': 24,
        'TC': 5,
        'MSG': 51,
        'Interrogator': 24,
    }
    def __init__(self,signalparameter: SignalParameter) -> None:
        super().__init__(signalparameter,'ADSB')
        #@audit 载波频率必须为1090MHz，1ppm/bit        
        self.start = 0#signalparameter.start # 绝对时间
        self.center_frequency = 1090e6
        self.stop = self.start+120e-6
        self.samples_per_symbol = 1e-6/(1/signalparameter.sample_rate)/12.8#int(signalparameter.samples_per_symbol/2)*2
        self.total_samples = self.samples_per_symbol*120
        self.frame = np.zeros(112,dtype=np.int32)
        self.upper_frequency = self.center_frequency
        self.lower_frequency = self.center_frequency
        self.signal_description.center_frequency = self.center_frequency/self.sample_rate
        self.signal_description.upper_frequency = self.signal_description.center_frequency
        self.signal_description.lower_frequency = self.signal_description.center_frequency
        
        # self.signal_description.center_frequency = 1090e6/signalparameter.sample_rate
        # self.signal_description.upper_frequency = 1090e6/signalparameter.sample_rate
        # self.signal_description.lower_frequency = 1090e6/signalparameter.sample_rate
 
    def GenerateFrame(self):
        ##只产生数据位
        cnt = 0
        for i in self.OrderedDict.keys():
            if i=='DF':
                data = np.random.choice([17,18,18],1)
                self.frame[cnt:cnt+self.OrderedDict[i]] = decimal_to_binary(data,self.OrderedDict[i])
            elif i=='CA':
                data = np.random.choice([0,1,2,3,4,5,6,7],1)
                self.frame[cnt:cnt+self.OrderedDict[i]] = decimal_to_binary(data,self.OrderedDict[i])
            else:
                self.frame[cnt:cnt+self.OrderedDict[i]] = np.random.randint(0,2,self.OrderedDict[i])
            cnt+=self.OrderedDict[i]
    
    # def ADSB(self):
    def __call__(self,shift:bool=False):
        
        self.GenerateFrame()
        frame = self.reconstruct(self.frame)
        self.frame = np.repeat(frame,self.samples_per_symbol/2) ## ads-b包络情况       
        
        frame_ = np.zeros(self.num_iq_samples)## 所有的点数
        
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        # self.frame = np.repeat(self.frame,self.samples_per_symbol)
        
        if shift: # 是否有左移，然后随机左移
            left_shift_bit = np.random.randint(0,int(len(self.iq_data)/4))
        else:
            left_shift_bit = 0
        
            
        if self.signal_description.start-left_shift_bit/self.num_iq_samples<0: #左冒头
            diff = abs(self.signal_description.start-left_shift_bit/self.num_iq_samples)
            self.stop = self.stop-left_shift_bit/self.sample_rate
            self.signal_description.stop = self.signal_description.stop-left_shift_bit/self.num_iq_samples
            self.signal_description.start = 0
            self.start = 0
            if len(self.frame[-int(self.signal_description.stop*self.num_iq_samples):])>len(frame_): #同时右冒头
                frame_ = self.frame[-len(frame_):]
                self.stop = self.num_iq_samples/self.sample_rate
                self.signal_description.stop = 1
            else:
                frame_[0:int(self.signal_description.stop*self.num_iq_samples)] = self.frame[-int(self.signal_description.stop*self.num_iq_samples):]
            
        elif self.signal_description.stop>1:#右冒头
            diff = abs(1-self.signal_description.stop) #超出的部分
            self.signal_description.stop = 1
            self.stop = self.num_iq_samples/self.sample_rate
            frame_[-int((1-self.signal_description.start)*self.num_iq_samples):]=self.frame[:int((1-self.signal_description.start)*self.num_iq_samples)]
            
        elif len(self.frame)<len(frame_): #如果帧的长度小于总的点数
            if self.num_iq_samples-self.start*self.sample_rate>len(self.frame): #可以完全放下
                # frame_[int(self.signal_description.start*self.num_iq_samples):int(self.signal_description.start*self.num_iq_samples)+len(self.frame)] = self.frame
                frame_[int(self.signal_description.start*self.num_iq_samples):int(self.signal_description.start*self.num_iq_samples)+len(self.frame[:len(frame_[int(self.signal_description.start*self.num_iq_samples):])])] = self.frame[:len(frame_[int(self.signal_description.start*self.num_iq_samples):])]
                self.signal_description.stop = self.signal_description.start+len(self.frame[:len(frame_[int(self.signal_description.start*self.num_iq_samples):])])/self.num_iq_samples
                self.stop = self.start+len(self.frame[:len(frame_[int(self.signal_description.start*self.num_iq_samples):])])/self.sample_rate
            else: #放不下
                frame_[int(self.signal_description.start*self.num_iq_samples):] = self.frame[:len(frame_[int(self.signal_description.start*self.num_iq_samples):])]
                self.signal_description.stop = self.signal_description.start+len(self.frame[:len(frame_[int(self.signal_description.start*self.num_iq_samples):])])/self.num_iq_samples
                self.stop = self.start+len(self.frame[:len(frame_[int(self.signal_description.start*self.num_iq_samples):])])/self.sample_rate
            # frame_[int(self.signal_description.start*self.num_iq_samples):int(self.signal_description.start*self.num_iq_samples)+len(self.frame)] = self.frame
        else: ## 帧长度大于总长度
            self.stop = self.num_iq_samples/self.sample_rate
            self.signal_description.stop = 1
            self.signal_description.start = 0
            self.start = 0
            frame_ = self.frame[:self.num_iq_samples]
            
        time_tmp = np.linspace(
                0,
                self.num_iq_samples/self.sample_rate,
                self.num_iq_samples
                )    
        self.carrier = np.exp(1j*2*np.pi*time_tmp*self.center_frequency)
        # self.frame_ = self.frame[0:len(time_tmp)]
        self.iq_data = self.carrier*frame_
        ## 数组左移操作
        
    
    def reconstruct(self,a):
        ##组成帧
        c=[]
        w=len(a)
        for j in range(0,w):
            if a[j]==1:
                if c==[]:
                    c=[1,0]
                else:
                    c.extend([1,0])
            else:
                if c==[]:
                    c=[0,1]
                else:
                    c.extend([0,1])
        header=[1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0]
        header.extend(c)
        c = header
        return c

class MSK:#@audit 搞不定
    def __init__(self,signalparameter:SignalParameter) -> None:
        # 这个对n应该没那么多要求吧
        self.num_iq_samples = signalparameter.num_iq_samples # 总的点数
        if signalparameter.num_symbol%2 != 0:
            signalparameter.num_symbol += 1
        self.num_symbols = signalparameter.num_symbol
        
        self.center_frequency = signalparameter.center_frequency
        self.signal_description = SignalDescription(
            center_frequency=signalparameter.center_frequency,
            upper_frequency=signalparameter.center_frequency,
            lower_frequency=signalparameter.center_frequency,
            bandwidth=0, #点频
            start=signalparameter.start,
            stop=signalparameter.stop,
        )
        self.signal_description.class_name = 'MSK'
        
    # def nPSK(self):
    def __call__(self):
        time_tmp = np.linspace(0,int(np.floor(self.signal_description.stop*self.num_iq_samples))-int(np.floor(self.signal_description.start*self.num_iq_samples)),
                               int(np.floor(self.signal_description.stop*self.num_iq_samples))-int(np.floor(self.signal_description.start*self.num_iq_samples)))
        
        self.iq_samples_per_symbol = int(len(time_tmp)//self.num_symbols)
        if len(time_tmp)%self.num_symbols != 0: #需要更新一下点数
            self.signal_description.stop = (self.signal_description.stop-self.signal_description.start)/len(time_tmp)*self.iq_samples_per_symbol*self.num_symbols+self.signal_description.start
            time_tmp = np.linspace(0,int(np.floor(self.signal_description.stop*self.num_iq_samples))-int(np.floor(self.signal_description.start*self.num_iq_samples)),
                               int(np.floor(self.signal_description.stop*self.num_iq_samples))-int(np.floor(self.signal_description.start*self.num_iq_samples)))
            
        f = self.signal_description.lower_frequency
        time_tmp = np.linspace(0,int(np.floor(self.signal_description.stop*self.num_iq_samples))-int(np.floor(self.signal_description.start*self.num_iq_samples)),
                               int(np.floor(self.signal_description.stop*self.num_iq_samples))-int(np.floor(self.signal_description.start*self.num_iq_samples)))
        
        source = np.random.choice(np.array([-1,1]),self.num_symbols,replace=True)
        
        phi = np.zeros(self.num_symbols) ## 初始相位给到0
        for i in range(1,self.num_symbols):
            phi[i] = phi[i-1]+i*np.pi/2*(source[i-1]-source[i]) ## 还没扩充
        phi = np.repeat(phi,self.iq_samples_per_symbol) # 时间上以这个长度为准
        source = np.repeat(source,self.iq_samples_per_symbol) # 时间上以这个长度为准
        ## f1 
        # iq = np.cos(source*np.pi*time_tmp/2/T+phi+self.signal_description.lower_frequency*time_tmp)
        iq = np.exp(1j*(source*np.pi*time_tmp/2*f+phi+2*np.pi*self.signal_description.lower_frequency*time_tmp))
        self.center_frequency = self.signal_description.lower_frequency
        self.signal_description.lower_frequency = self.signal_description.lower_frequency*0.75
        self.signal_description.upper_frequency = self.signal_description.lower_frequency*1.65
        
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        self.iq_data[int(np.floor(self.signal_description.start*self.num_iq_samples)):int(np.floor(self.signal_description.stop*self.num_iq_samples))] = iq

class Frank(BaseSignal):
    def __init__(self,signalparameter:SignalParameter,n:int=2) -> None:
        super().__init__(signalparameter,'Frank')
        # 这个对n应该没那么多要求吧
        self.num_iq_samples = signalparameter.num_iq_samples # 总的点数
        self.num_symbols = np.power(int(np.sqrt(signalparameter.num_symbol)),2)
        self.status = n
        self.bandwidth=0
        self.upper_frequency = self.center_frequency
        self.lower_frequency = self.center_frequency
        self.signal_description.upper_frequency = self.signal_description.center_frequency
        self.signal_description.lower_frequency = self.signal_description.center_frequency
        
    def __call__(self):
        time_tmp = np.linspace(0,(self.stop-self.start),
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        self.iq_samples_per_symbol = int(len(time_tmp)//self.num_symbols)
        self.stop = self.iq_samples_per_symbol*self.num_symbols/self.sample_rate+self.start
        self.signal_description.stop = self.iq_samples_per_symbol*self.num_symbols/self.num_iq_samples+self.signal_description.start
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        # bandwidth_div = self.bandwidth/(self.status-1)
        phase = self.__frank__(int(np.sqrt(self.num_symbols)))
        # source = np.random.choice(np.arange(0,self.status),self.num_symbols,replace=True) #@todo 这里的随机抽取可能要改一下，不是很均匀
        # print(source)
        # source_ = decimal_to_binary(source,int(np.log2(self.status))) ##@todo 长度要再确定下
        # del source
        # source = np.random.binomial(n=1,p=0.5,size=(128))
        source = np.repeat(phase,self.iq_samples_per_symbol) # 时间上以这个长度为准
        time_tmp = time_tmp[0:source.shape[0]]
        self.iq_data = np.zeros(int(self.num_iq_samples),dtype=np.complex64)
        iq = np.exp(1j*2*np.pi*self.center_frequency*time_tmp)*source ##@audit QAM调制是不是这样的？
        self.iq_data[int(np.floor(self.start*self.sample_rate)):int(np.floor(self.stop*self.sample_rate))] = iq
        
    def __frank__(self,N):
        phi = np.zeros((N,N),dtype=complex)
        for i in range(N):
            for j in range(N):
                phi[i,j] = np.exp(1j*2*np.pi/N*i*j)
        phi = phi.reshape(1,-1)[0]
        return phi # 输出为相位矩阵

class Costas(BaseSignal):
    def __init__(self,signalparameter:SignalParameter) -> None:
        super().__init__(signalparameter,'Costas')

    def __call__(self):
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        self.iq_data = np.zeros(int(self.num_iq_samples),dtype=np.complex64) ## 总的点
        p = np.random.choice([11,13]) #就这几个 ## n=p-2
        freq = self.__Costas_Generate__(p,a=2)
        n=p-2
        bandwidth_div = self.bandwidth/(n-1)
        self.iq_samples_per_symbol = int(len(time_tmp)//n)
        
        self.signal_description.stop = self.iq_samples_per_symbol*n/len(time_tmp)*(self.signal_description.stop-self.signal_description.start)+self.signal_description.start
        self.stop = self.iq_samples_per_symbol*n/self.sample_rate+self.start
        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        source = np.linspace(1,n,n)
        source = np.repeat(source,self.iq_samples_per_symbol) # 时间上以这个长度为准
        for i,j in enumerate(source):
            source[i] = freq[int(j)-1][1]
        # time = np.linspace(0,self.num_iq_samples,self.num_iq_samples) ## 总时间
                
        # time_ = time[int(np.floor(self.signal_description.start*self.num_iq_samples)):int(np.floor(self.signal_description.stop*self.num_iq_samples))]
        
        iq = np.exp(1j*2*np.pi*(self.lower_frequency+bandwidth_div*(source-1))*time_tmp)
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq
        # return iq,self.signal_description
    
    def __Costas_Generate__(self,p=13,a=2): #prime,primitive element
        # p = 11 # prime
        # a = 2 # primitive element

        # verify a is a primitive element mod p
        s = {a**i % p for i in range(p)}
        assert( len(s) == p-1 )

        n = p-2
        dots = []

        for i in range(1, n+1):
            for j in range(1, n+1):
                if (np.power(a, i)%p + np.power(a, j)%p) % p == 1:
                    dots.append((i,j))
                    break
        return dots
    
def ConvertDescriptionToPatch(description: SignalDescription,zeroIF:bool=True,nfft:int=512) -> List[float]:
    ## 按照零中频的情况设计
    ## 默认长宽一样
    ## 方向：从上到下，从左到右
    if zeroIF:
        start = description.start*nfft
        stop = description.stop*nfft
        lower = (description.lower_frequency+0.5)*nfft
        higher = (description.upper_frequency+0.5)*nfft
        if stop>=nfft:
            width = stop-start-2 ## 微调的数参数
        else:
            width = stop-start-2
        # if abs(lower-higher)<(0.01*nfft):
        if lower>higher:
            return (start,lower+8),width,(higher-lower)-16 #@todo 看下是不是可以稍微放松几个像素
        else:
            return (start,lower-8),width,(higher-lower)+16 #@todo 看下是不是可以稍微放松几个像素
        # else:
        #     return (start-2,lower-1),stop-start,(higher-lower)+2 #@todo 看下是不是可以稍微放松几个像素
        # return (start-5,lower-5),stop-start+9,(higher-lower)+9 #@todo 看下是不是可以稍微放松几个像素
    else:
        return
    # return [description.start,description.stop,description.lower_frequency,description.upper_frequency]
    

def array_rotation(arr, k): ## 数据增强用
    # 用于数组循环移位
    return np.roll(arr, k)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def resample(input_signal,src_fs,tar_fs):
    '''
    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    '''
    dtype = input_signal.dtype
    audio_len = len(input_signal)
    audio_time_max = 1.0*(audio_len-1) / src_fs
    src_time = 1.0 * np.linspace(0,audio_len,audio_len) / src_fs
    tar_time = 1.0 * np.linspace(0,np.int(audio_time_max*tar_fs),np.int(audio_time_max*tar_fs)) / tar_fs
    output_signal = np.interp(tar_time,src_time,input_signal).astype(dtype)
    return output_signal

class signal(SignalData):
    def __init__(self, data: bytes | None, item_type: np.dtype, data_type: np.dtype, signal_description: List[SignalDescription] | SignalDescription | None = None) -> None:
        super().__init__(data, item_type, data_type, signal_description)

if __name__ == '__main__':
    a=SignalParameter()
    LFMs = nQAM(a,4)
    aa = ADSB(a)
    cnt = 0
    for i in aa.OrderedDict.keys():
        cnt+=aa.OrderedDict[i]
        print(aa.OrderedDict[i])
    print(cnt)
    aa()
    LFMs()
    print(LFMs.signal_description)
    # print(LFMs.iq_data)