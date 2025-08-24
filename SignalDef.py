from typing import Any, List, Optional, Union
from SignalDescription import SignalDescription
from SignalDescription import SignalData
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter, freqz
import commpy as cpy ## 省事写法
from commpy.modulation import Modem
# import h5py
import mat73

# 临时处理缺失的mat文件
try:
    zigbee_data = sio.loadmat('zigbee.mat')
except FileNotFoundError:
    print("Warning: zigbee.mat not found, using dummy data")
    zigbee_data = {'data': np.array([1, 2, 3])}

try:
    features = mat73.loadmat('LoRa.mat')
except FileNotFoundError:
    print("Warning: LoRa.mat not found, using dummy data")
    features = {'data': np.array([1, 2, 3])}

class SignalParameter:
## 定义信号参数，采样率归一化
## fs为1，频率就在-0.5到0.5之间
## 有正负频率
## 时间片长度定为1
    def __init__(self,num_iq_samples:int=512*512,sample_rate=None) -> None:
        self.sample_rate = sample_rate
        self.num_iq_samples = int(num_iq_samples)#np.random.randint(0, 512*512) ##@todo 要有torchsig的效果的话就要改stft的时窗
        self.bandwidth = np.random.uniform(-0.2, 0.2)*sample_rate
        
        self.center_frequency = np.random.uniform(-0.4, 0.4)*sample_rate
        self.lower_frequency = self.center_frequency-self.bandwidth/2
        self.upper_frequency = self.lower_frequency + self.bandwidth
        
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
        
        self.num_symbol = np.random.randint(10, 40)
        
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

class APSKModem(Modem):
    
    def __init__(self, constellation_type='16APSK'):
        configs = {
            '16APSK': {'radii': [1.0, 2.2], 'points': [4, 12]},
            '32APSK': {'radii': [1.0, 2.2, 3.5], 'points': [4, 12, 16]},
            '64APSK': {'radii': [1.0, 2.2, 3.5, 4.8], 'points': [4, 12, 20, 28]}
        }
        
        if constellation_type not in configs:
            raise ValueError(f"不支持的APSK类型: {constellation_type}")
        
        config = configs[constellation_type]
        constellation = self._generate_constellation(config['radii'], config['points'])
        super().__init__(constellation)
    
    def _generate_constellation(self, radii, points_per_ring):
        constellation = []
        
        for ring_idx, (radius, num_points) in enumerate(zip(radii, points_per_ring)):
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            
            # 优化环间相位
            if ring_idx == 1:
                angles += np.pi / 12
            elif ring_idx == 2:
                angles += np.pi / 16
            
            for angle in angles:
                point = radius * np.exp(1j * angle)
                constellation.append(point)
        
        return np.array(constellation)

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

class LFM(BaseSignal):
    def __init__(self,signalparameter:SignalParameter) -> None:
        super().__init__(signalparameter,'LFM')

    # def LFM(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        skew = (self.upper_frequency-self.lower_frequency)/(self.stop-self.start)
        time_tmp = np.linspace(0,(self.stop-self.start),int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        iq = np.exp(1j*np.pi*time_tmp*(time_tmp*skew+2*self.lower_frequency)) 
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq
        del iq,time_tmp
        # return iq,self.signal_description

class NLFM(BaseSignal):
    def __init__(self,signalparameter:SignalParameter) -> None:
        super().__init__(signalparameter,'NLFM')

    # def NLFM(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)

        time_tmp = np.linspace(0,self.stop-self.start,
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))

        T = self.stop - self.start  # 信号持续时间
        freq_deviation = self.upper_frequency - self.lower_frequency
        cosine_frequency_factor = np.random.uniform(1, 10.0)  # 随机选择0.5到2.0之间的值
        instant_freq = self.lower_frequency + freq_deviation * (1 - np.cos(cosine_frequency_factor * np.pi * time_tmp / T)) / 2

        phase = 2 * np.pi * np.cumsum(instant_freq) / self.sample_rate
        
        iq = np.exp(1j * phase)  ##@audit 全部给成归一化的幅值
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
        
        tmp = np.random.randn(len(time_tmp))
        tmp = butter_lowpass_filter(tmp,self.bandwidth,self.sample_rate)
        source = tmp#np.exp(1j*2*np.pi*np.cumsum(tmp)*self.a)*np.sin(2*np.pi*time_tmp*self.bandwidth/2)#5*np.sin(time_tmp*2*np.pi)*np.sin(2*np.pi*time_tmp*self.bandwidth/2)#(np.random.randn(len(time_tmp))+1j*np.random.randn(len(time_tmp)))
        # source = np.random.randn(len(time_tmp))
        iq = np.exp(1j*2*np.pi*self.center_frequency*time_tmp)*source+self.a*np.exp(1j*2*np.pi*self.center_frequency*time_tmp) # 带宽没法确定
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq

class DSB(BaseSignal): 
    def __init__(self,signalparameter: SignalParameter,modulate_coefficient=0.8) -> None:
        super().__init__(signalparameter,'DSB')
        self.a = modulate_coefficient
        # DSB的带宽应该是基带信号带宽的两倍
        self.msg_bandwidth = 0.005*self.sample_rate  # 基带信号带宽
        self.bandwidth = 2 * self.msg_bandwidth      # DSB带宽是基带的两倍
    # def DSB(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        time_tmp = np.linspace(0,(self.stop-self.start),
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        source = np.zeros(len(time_tmp))
        for i in range(5):  # 减少频率分量数量
            freq = np.random.uniform(0.1, 1.0) * self.msg_bandwidth / 5  # 基带频率分量
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.2, 1.0)
            source += amplitude * np.sin(2*np.pi*freq*time_tmp + phase)
        
        # 归一化基带信号
        source = source / np.max(np.abs(source)) * self.a
        
        # 低通滤波，限制基带信号带宽
        source = butter_lowpass_filter(source, self.msg_bandwidth, self.sample_rate)
        
        # 单边带调制（SSB-SC）：使用希尔伯特变换实现
        # 生成解析信号（复信号）
        from scipy.signal import hilbert
        
        # 对基带信号进行希尔伯特变换，得到解析信号
        analytic_signal = hilbert(source)
        
        # 选择上边带或下边带
        sideband_choice = np.random.choice(['USB', 'LSB'])  # 随机选择上边带或下边带
        
        if sideband_choice == 'USB':  # 上边带 (Upper Sideband)
            # 上边带：使用正频率分量
            ssb_baseband = analytic_signal
            # 更新频率范围（只有上边带）
            self.lower_frequency = self.center_frequency
            self.upper_frequency = self.center_frequency + self.msg_bandwidth
        else:  # 下边带 (Lower Sideband)
            # 下边带：使用负频率分量（共轭）
            ssb_baseband = np.conj(analytic_signal)
            # 更新频率范围（只有下边带）
            self.lower_frequency = self.center_frequency - self.msg_bandwidth
            self.upper_frequency = self.center_frequency
        
        # 载波调制：将SSB基带信号调制到载波频率
        carrier_complex = np.exp(1j * 2 * np.pi * self.center_frequency * time_tmp)
        ssb_signal = ssb_baseband * carrier_complex
        
        # 更新带宽为单边带
        self.bandwidth = self.msg_bandwidth  # SSB带宽等于基带带宽
        
        # 确保输出为复数IQ信号
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(ssb_signal))] = ssb_signal
        
        # 更新信号描述
        self.signal_description.upper_frequency = self.upper_frequency / self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency / self.sample_rate
        self.signal_description.bandwidth = self.bandwidth / self.sample_rate
        # 保持类名为'DSB'以兼容现有的标签系统，但在内部实现为SSB
        # self.signal_description.class_name = f'SSB-{sideband_choice}'  # 注释掉，保持'DSB'

class FM(BaseSignal): 
    #@todo 需要整成窄带的
    def __init__(self,signalparameter: SignalParameter,modulate_coefficient=0.05) -> None:
        super().__init__(signalparameter,'FM')
        self.a = modulate_coefficient
        
    # def FM(self):
    def __call__(self):
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        time_tmp = np.linspace(0,(self.stop-self.start),
                               int(self.stop*self.sample_rate)-int(self.start*self.sample_rate))
        
        source = np.zeros(len(time_tmp))
        for i in range(0,40):
            source = source+np.sin(2*np.pi*(time_tmp*9e5*np.random.randn(1)+np.random.randn(1)))
        source = source/40
        # source = np.random.randn(len(time_tmp))#(np.random.randn(len(time_tmp))+1j*np.random.randn(len(time_tmp)))
        source = butter_lowpass_filter(source,self.bandwidth,2.5e9)

        # source = np.random.randn(len(time_tmp))
        iq = np.exp(1j*2*np.pi*(self.center_frequency)*time_tmp)*np.exp(1j*2*np.pi*np.cumsum(source)*self.a)
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+len(iq))] = iq

class OFDM(BaseSignal):
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

class Zigbee(BaseSignal):
    def __init__(self,signalparameter: SignalParameter) -> None:
        super().__init__(signalparameter,'Zigbee')
        # self.source_path = 'zigbee.mat'
        self.signal_description.lower_frequency = self.lower_frequency/self.sample_rate

        self.bandwidth = 2e6
        self.signal_description.bandwidth = self.bandwidth/self.sample_rate
        self.center_frequency = 0.915e9
        self.signal_description.center_frequency = self.center_frequency/self.sample_rate
        self.upper_frequency = self.center_frequency+self.bandwidth/2
        self.lower_frequency = self.center_frequency-self.bandwidth/2
        self.signal_description.upper_frequency = self.upper_frequency/self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency/self.sample_rate
        self.data = zigbee_data #sio.loadmat(self.source_path)
        self.data = self.data[list(self.data.keys())[-1]]
        self.start_point = np.random.randint(0,(self.data.shape[0]-8e6))
        
        pass
    def __call__(self):
        iq = self.data[self.start_point:self.start_point+int(np.floor(self.stop*self.sample_rate))-int(np.floor(self.start*self.sample_rate))].T*np.exp(1j*2*np.pi*self.center_frequency*np.linspace(0,self.stop-self.start,int(np.floor(self.stop*self.sample_rate))-int(np.floor(self.start*self.sample_rate))))
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+iq.shape[1])] = iq
        # self.iq
        pass

class LoRa(BaseSignal):
    def __init__(self,signalparameter: SignalParameter) -> None:
        super().__init__(signalparameter,'LoRa')
        # self.source_path = 'LoRa.mat'
        self.signal_description.lower_frequency = self.lower_frequency/self.sample_rate

        self.bandwidth = 5e5
        self.signal_description.bandwidth = self.bandwidth/self.sample_rate
        self.center_frequency = 0.915e9
        self.signal_description.center_frequency = self.center_frequency/self.sample_rate
        self.upper_frequency = self.center_frequency+self.bandwidth/2
        self.lower_frequency = self.center_frequency-self.bandwidth/2
        self.signal_description.upper_frequency = self.upper_frequency/self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency/self.sample_rate
        # features = mat73.loadmat(self.source_path)
        self.data = features['LoRa']
        # self.data = sio.loadmat(self.source_path)
        # self.data = self.data[list(self.data.keys())[-1]]
        self.start_point = np.random.randint(0,(self.data.shape[0]-8e6))
        
        pass
    def __call__(self):
        iq = self.data[self.start_point:self.start_point+int(np.floor(self.stop*self.sample_rate))-int(np.floor(self.start*self.sample_rate))].T*np.exp(1j*2*np.pi*self.center_frequency*np.linspace(0,self.stop-self.start,int(np.floor(self.stop*self.sample_rate))-int(np.floor(self.start*self.sample_rate))))
        self.iq_data = np.zeros(self.num_iq_samples,dtype=np.complex64)
        self.iq_data[int(np.floor(self.start*self.sample_rate)):(int(np.floor(self.start*self.sample_rate))+iq.shape[0])] = iq
        # self.iq
        pass


class RADAR_Pulse(BaseSignal):

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
    ## 4PSK、8PSK、16PSK、32PSK
    def __init__(self,signalparameter:SignalParameter,n:int=2) -> None:
        # 根据n值设置类名
        if n == 4:
            class_name = '4PSK'
        elif n == 8:
            class_name = '8PSK'
        elif n == 16:
            class_name = '16PSK'
        elif n == 32:
            class_name = '32PSK'
        else:
            class_name = 'PSK'
        super().__init__(signalparameter, class_name)
        
        self.status = n
        
        # 设置合理的符号率和根升余弦滤波参数
        self.symbol_rate = np.random.uniform(1e6, 8e6)  # 1-8 Msps，适中的符号率
        self.rolloff_factor = np.random.uniform(0.2, 0.5)  # 滚降因子，典型值
        self.upsample_factor = np.random.randint(4, 12)  # 上采样因子，4-12倍
        
        # 根据奈奎斯特准则计算占用带宽
        self.bandwidth = self.symbol_rate * (1 + self.rolloff_factor)
        
        # 更新频率边界
        self.upper_frequency = self.center_frequency + self.bandwidth/2
        self.lower_frequency = self.center_frequency - self.bandwidth/2
        
        # 更新信号描述中的带宽信息（归一化）
        self.signal_description.bandwidth = self.bandwidth / self.sample_rate
        self.signal_description.upper_frequency = self.upper_frequency / self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency / self.sample_rate

    # def nPSK(self):
    def __call__(self):
        # 计算所需的时间长度
        duration = self.stop - self.start
        
        # 设置更合理的参数以确保足够长的信号，但保持原有的符号率范围
        min_duration_ratio = 0.15  # 最小15%时间
        max_duration_ratio = 0.65  # 最大65%时间
        total_time = self.num_iq_samples / self.sample_rate
        
        # 调整持续时间，确保信号足够长
        target_duration = np.random.uniform(min_duration_ratio, max_duration_ratio) * total_time
        duration = max(duration, target_duration)
        
        # 更新停止时间
        self.stop = min(self.start + duration, total_time * 0.9)  # 确保不超出边界
        duration = self.stop - self.start
        
        # 保持原有的符号率，但调整符号数量来适应新的持续时间
        target_symbols = max(50, int(duration * self.symbol_rate))  # 根据持续时间计算符号数
        self.num_symbols = min(target_symbols, target_symbols * 2)  # 允许更多符号
        
        # 动态调整上采样因子，确保合理的信号长度
        self.upsample_factor = np.random.randint(4, 8)  # 减少上采样因子避免信号过长
        
        # 重新计算带宽（保持原有计算方式）
        self.bandwidth = self.symbol_rate * (1 + self.rolloff_factor)
        self.upper_frequency = self.center_frequency + self.bandwidth/2
        self.lower_frequency = self.center_frequency - self.bandwidth/2
        
        # 更新信号描述中的带宽信息
        self.signal_description.bandwidth = self.bandwidth / self.sample_rate
        self.signal_description.upper_frequency = self.upper_frequency / self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency / self.sample_rate
        
        oversampled_sample_rate = self.symbol_rate * self.upsample_factor
        
        # 确保过采样率不超过系统采样率
        if oversampled_sample_rate > self.sample_rate:
            self.upsample_factor = max(2, int(self.sample_rate / self.symbol_rate * 0.8))
            oversampled_sample_rate = self.symbol_rate * self.upsample_factor
        
        # 生成随机符号序列
        source_symbols = np.random.choice(np.arange(0, self.status), self.num_symbols, replace=True)
        source_bits = decimal_to_binary(source_symbols, int(np.log2(self.status)))
        
        # PSK调制
        nPSK_modem = cpy.PSKModem(self.status)
        modulated_symbols = nPSK_modem.modulate(source_bits)
        
        # 上采样：在符号之间插入零
        upsampled_signal = np.zeros(len(modulated_symbols) * self.upsample_factor, dtype=np.complex64)
        upsampled_signal[::self.upsample_factor] = modulated_symbols
        
        # 设计根升余弦滤波器
        filter_span = 6  # 滤波器跨度
        filter_order = filter_span * self.upsample_factor
        
        # 生成根升余弦滤波器的时间轴
        t_filter = np.arange(-filter_order//2, filter_order//2 + 1) / oversampled_sample_rate
        
        # 根升余弦滤波器实现
        rrc_filter = self._root_raised_cosine_filter(t_filter, self.symbol_rate, self.rolloff_factor)
        
        # 应用滤波器（卷积）
        filtered_signal = np.convolve(upsampled_signal, rrc_filter, mode='same')
        
        # 归一化功率
        if np.std(filtered_signal) > 0:
            filtered_signal = filtered_signal / np.std(filtered_signal) * 0.7
        
        # 重采样到系统采样率
        if oversampled_sample_rate != self.sample_rate:
            # 计算重采样后的长度
            target_length = int(len(filtered_signal) * self.sample_rate / oversampled_sample_rate)
            # 简单的线性插值重采样
            old_indices = np.linspace(0, len(filtered_signal)-1, len(filtered_signal))
            new_indices = np.linspace(0, len(filtered_signal)-1, target_length)
            final_signal = np.interp(new_indices, old_indices, filtered_signal.real) + \
                          1j * np.interp(new_indices, old_indices, filtered_signal.imag)
        else:
            final_signal = filtered_signal
        
        # 确保信号长度不超过预期，但允许足够长的信号
        max_samples = int(duration * self.sample_rate)
        if len(final_signal) > max_samples:
            final_signal = final_signal[:max_samples]
        
        # 计算实际的时间长度
        actual_duration = len(final_signal) / self.sample_rate
        self.stop = self.start + actual_duration
        
        # PSK边界框优化：向右延长时间，增加频率高度
        time_extension_factor = 1.004  # 时间向右延长0.4%
        freq_extension_factor = 1.5   # 频率高度增加50%
        
        # 扩展时间边界
        extended_duration = actual_duration * time_extension_factor
        extended_stop = min(self.start + extended_duration, 
                          (self.num_iq_samples/self.sample_rate) * 0.95)
        
        # 扩展频率边界
        current_bandwidth = self.upper_frequency - self.lower_frequency
        extended_bandwidth = current_bandwidth * freq_extension_factor
        bandwidth_increase = extended_bandwidth - current_bandwidth
        
        # 对称扩展频率范围
        extended_lower_freq = self.lower_frequency - bandwidth_increase / 2
        extended_upper_freq = self.upper_frequency + bandwidth_increase / 2
        
        # 确保频率不超出奈奎斯特范围
        max_freq = self.sample_rate / 2 * 0.95
        min_freq = -self.sample_rate / 2 * 0.95
        extended_lower_freq = max(extended_lower_freq, min_freq)
        extended_upper_freq = min(extended_upper_freq, max_freq)
        
        # 更新信号描述的边界信息（使用扩展后的边界）
        self.signal_description.stop = self.start / (self.num_iq_samples/self.sample_rate) + \
                                      (extended_stop - self.start) / (self.num_iq_samples/self.sample_rate)
        self.signal_description.upper_frequency = extended_upper_freq / self.sample_rate
        self.signal_description.lower_frequency = extended_lower_freq / self.sample_rate
        self.signal_description.bandwidth = (extended_upper_freq - extended_lower_freq) / self.sample_rate
        # 生成载波调制后的时间轴
        time_axis = np.linspace(0, actual_duration, len(final_signal))
        
        # 应用载波调制
        carrier_modulated = final_signal * np.exp(1j * 2 * np.pi * self.center_frequency * time_axis)
        
        # 放入IQ数据数组
        self.iq_data = np.zeros(self.num_iq_samples, dtype=np.complex64)
        start_idx = int(np.floor(self.start * self.sample_rate))
        end_idx = min(start_idx + len(carrier_modulated), self.num_iq_samples)
        actual_length = end_idx - start_idx
        
        self.iq_data[start_idx:end_idx] = carrier_modulated[:actual_length]
    
    def _root_raised_cosine_filter(self, t, symbol_rate, rolloff):
        """
        生成根升余弦滤波器
        """
        T = 1.0 / symbol_rate  # 符号周期
        
        # 避免除零
        epsilon = 1e-10
        t = t + epsilon * (t == 0)
        
        # 根升余弦滤波器公式
        if rolloff == 0:
            h = np.sinc(t / T)
        else:
            # 处理特殊点 t = ±T/(4*rolloff)
            special_points = np.abs(np.abs(t) - T/(4*rolloff)) < epsilon
            
            h = np.zeros_like(t)
            
            # 一般情况
            normal_points = ~special_points
            t_norm = t[normal_points]
            
            numerator = np.sin(np.pi * t_norm / T * (1 - rolloff)) + \
                       4 * rolloff * t_norm / T * np.cos(np.pi * t_norm / T * (1 + rolloff))
            denominator = np.pi * t_norm / T * (1 - (4 * rolloff * t_norm / T) ** 2)
            
            h[normal_points] = numerator / denominator
            
            # 特殊点的值
            h[special_points] = rolloff / np.sqrt(2) * \
                              ((1 + 2/np.pi) * np.sin(np.pi/(4*rolloff)) + \
                               (1 - 2/np.pi) * np.cos(np.pi/(4*rolloff)))
        
        return h

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
    ## 16QAM、64QAM、256QAM
    def __init__(self,signalparameter:SignalParameter,n:int=4) -> None:
        # 根据n值设置类名
        if n == 16:
            class_name = '16QAM'
        elif n == 64:
            class_name = '64QAM'
        elif n == 256:
            class_name = '256QAM'
        else:
            class_name = 'QAM'
        super().__init__(signalparameter, class_name)
        if np.log2(n) != int(np.log2(n)):
            raise ValueError('nQAM的n必须是2的幂次')
        
        self.status = n
        
        # 设置合理的符号率和根升余弦滤波参数
        # QAM可以支持更高的符号率，因为频谱效率更高
        self.symbol_rate = np.random.uniform(2e6, 12e6)  # 2-12 Msps
        self.rolloff_factor = np.random.uniform(0.15, 0.4)  # 较小的滚降因子，提高频谱效率
        self.upsample_factor = np.random.randint(4, 10)  # 上采样因子
        
        # 根据奈奎斯特准则计算占用带宽
        self.bandwidth = self.symbol_rate * (1 + self.rolloff_factor)
        
        # 更新频率边界
        self.upper_frequency = self.center_frequency + self.bandwidth/2
        self.lower_frequency = self.center_frequency - self.bandwidth/2
        
        # 更新信号描述中的带宽信息（归一化）
        self.signal_description.bandwidth = self.bandwidth / self.sample_rate
        self.signal_description.upper_frequency = self.upper_frequency / self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency / self.sample_rate
    # def nQAM(self):
    def __call__(self):
        # 计算所需的时间长度
        duration = self.stop - self.start
        
        # 设置更合理的参数以确保足够长的信号，但保持原有的符号率范围
        min_duration_ratio = 0.15  # 最小15%时间
        max_duration_ratio = 0.65  # 最大65%时间
        total_time = self.num_iq_samples / self.sample_rate
        
        # 调整持续时间，确保信号足够长
        target_duration = np.random.uniform(min_duration_ratio, max_duration_ratio) * total_time
        duration = max(duration, target_duration)
        
        # 更新停止时间
        self.stop = min(self.start + duration, total_time * 0.9)  # 确保不超出边界
        duration = self.stop - self.start
        
        # 保持原有的符号率，但调整符号数量来适应新的持续时间
        target_symbols = max(50, int(duration * self.symbol_rate))  # 根据持续时间计算符号数
        self.num_symbols = min(target_symbols, target_symbols * 2)  # 允许更多符号
        
        # 动态调整上采样因子，确保合理的信号长度
        self.upsample_factor = np.random.randint(4, 8)  # 减少上采样因子避免信号过长
        
        # 重新计算带宽（保持原有计算方式）
        self.bandwidth = self.symbol_rate * (1 + self.rolloff_factor)
        self.upper_frequency = self.center_frequency + self.bandwidth/2
        self.lower_frequency = self.center_frequency - self.bandwidth/2
        
        # 更新信号描述中的带宽信息
        self.signal_description.bandwidth = self.bandwidth / self.sample_rate
        self.signal_description.upper_frequency = self.upper_frequency / self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency / self.sample_rate
        
        oversampled_sample_rate = self.symbol_rate * self.upsample_factor
        
        # 确保过采样率不超过系统采样率
        if oversampled_sample_rate > self.sample_rate:
            self.upsample_factor = max(2, int(self.sample_rate / self.symbol_rate * 0.8))
            oversampled_sample_rate = self.symbol_rate * self.upsample_factor
        
        # 生成随机符号序列
        source_symbols = np.random.choice(np.arange(0, self.status), self.num_symbols, replace=True)
        source_bits = decimal_to_binary(source_symbols, int(np.log2(self.status)))
        
        # QAM调制
        nQAM_modem = cpy.QAMModem(self.status)
        modulated_symbols = nQAM_modem.modulate(source_bits)
        
        # 上采样：在符号之间插入零
        upsampled_signal = np.zeros(len(modulated_symbols) * self.upsample_factor, dtype=np.complex64)
        upsampled_signal[::self.upsample_factor] = modulated_symbols
        
        # 设计根升余弦滤波器
        filter_span = 6  # 滤波器跨度
        filter_order = filter_span * self.upsample_factor
        
        # 生成根升余弦滤波器的时间轴
        t_filter = np.arange(-filter_order//2, filter_order//2 + 1) / oversampled_sample_rate
        
        # 根升余弦滤波器实现
        rrc_filter = self._root_raised_cosine_filter(t_filter, self.symbol_rate, self.rolloff_factor)
        
        # 应用滤波器（卷积）
        filtered_signal = np.convolve(upsampled_signal, rrc_filter, mode='same')
        
        # QAM信号的功率归一化 - 考虑星座图特性
        if np.std(filtered_signal) > 0:
            # 对于QAM，不同星座点有不同的功率，需要适当的归一化
            constellation_power = np.sqrt((self.status - 1) / 3.0)  # QAM星座图的理论功率
            filtered_signal = filtered_signal / np.std(filtered_signal) * 0.7 / constellation_power
        
        # 重采样到系统采样率
        if oversampled_sample_rate != self.sample_rate:
            # 计算重采样后的长度
            target_length = int(len(filtered_signal) * self.sample_rate / oversampled_sample_rate)
            # 简单的线性插值重采样
            old_indices = np.linspace(0, len(filtered_signal)-1, len(filtered_signal))
            new_indices = np.linspace(0, len(filtered_signal)-1, target_length)
            final_signal = np.interp(new_indices, old_indices, filtered_signal.real) + \
                          1j * np.interp(new_indices, old_indices, filtered_signal.imag)
        else:
            final_signal = filtered_signal
        
        # 确保信号长度不超过预期，但允许足够长的信号
        max_samples = int(duration * self.sample_rate)
        if len(final_signal) > max_samples:
            final_signal = final_signal[:max_samples]
        
        # 计算实际的时间长度
        actual_duration = len(final_signal) / self.sample_rate
        self.stop = self.start + actual_duration
        
        # QAM边界框优化：向右延长时间，增加频率高度
        time_extension_factor = 1.004  # 时间向右延长25%
        freq_extension_factor = 1.5   # 频率高度增加40%
        
        # 扩展时间边界
        extended_duration = actual_duration * time_extension_factor
        extended_stop = min(self.start + extended_duration, 
                          (self.num_iq_samples/self.sample_rate) * 0.95)
        
        # 扩展频率边界
        current_bandwidth = self.upper_frequency - self.lower_frequency
        extended_bandwidth = current_bandwidth * freq_extension_factor
        bandwidth_increase = extended_bandwidth - current_bandwidth
        
        # 对称扩展频率范围
        extended_lower_freq = self.lower_frequency - bandwidth_increase / 2
        extended_upper_freq = self.upper_frequency + bandwidth_increase / 2
        
        # 确保频率不超出奈奎斯特范围
        max_freq = self.sample_rate / 2 * 0.95
        min_freq = -self.sample_rate / 2 * 0.95
        extended_lower_freq = max(extended_lower_freq, min_freq)
        extended_upper_freq = min(extended_upper_freq, max_freq)
        
        # 更新信号描述的边界信息（使用扩展后的边界）
        self.signal_description.stop = self.start / (self.num_iq_samples/self.sample_rate) + \
                                      (extended_stop - self.start) / (self.num_iq_samples/self.sample_rate)
        self.signal_description.upper_frequency = extended_upper_freq / self.sample_rate
        self.signal_description.lower_frequency = extended_lower_freq / self.sample_rate
        self.signal_description.bandwidth = (extended_upper_freq - extended_lower_freq) / self.sample_rate
        # 生成载波调制后的时间轴
        time_axis = np.linspace(0, actual_duration, len(final_signal))
        
        # 应用载波调制
        carrier_modulated = final_signal * np.exp(1j * 2 * np.pi * self.center_frequency * time_axis)
        
        # 放入IQ数据数组
        self.iq_data = np.zeros(self.num_iq_samples, dtype=np.complex64)
        start_idx = int(np.floor(self.start * self.sample_rate))
        end_idx = min(start_idx + len(carrier_modulated), self.num_iq_samples)
        actual_length = end_idx - start_idx
        
        self.iq_data[start_idx:end_idx] = carrier_modulated[:actual_length]
    
    def _root_raised_cosine_filter(self, t, symbol_rate, rolloff):
        """
        生成根升余弦滤波器 (与PSK使用相同的实现)
        """
        T = 1.0 / symbol_rate  # 符号周期
        
        # 避免除零
        epsilon = 1e-10
        t = t + epsilon * (t == 0)
        
        # 根升余弦滤波器公式
        if rolloff == 0:
            h = np.sinc(t / T)
        else:
            # 处理特殊点 t = ±T/(4*rolloff)
            special_points = np.abs(np.abs(t) - T/(4*rolloff)) < epsilon
            
            h = np.zeros_like(t)
            
            # 一般情况
            normal_points = ~special_points
            t_norm = t[normal_points]
            
            numerator = np.sin(np.pi * t_norm / T * (1 - rolloff)) + \
                       4 * rolloff * t_norm / T * np.cos(np.pi * t_norm / T * (1 + rolloff))
            denominator = np.pi * t_norm / T * (1 - (4 * rolloff * t_norm / T) ** 2)
            
            h[normal_points] = numerator / denominator
            
            # 特殊点的值
            h[special_points] = rolloff / np.sqrt(2) * \
                              ((1 + 2/np.pi) * np.sin(np.pi/(4*rolloff)) + \
                               (1 - 2/np.pi) * np.cos(np.pi/(4*rolloff)))
        
        return h

class nAPSK(BaseSignal):
    """APSK信号类 - 支持16APSK, 32APSK, 64APSK"""
    
    def __init__(self, signalparameter: SignalParameter, n: int = 16) -> None:
        # 根据n值设置类名
        if n == 16:
            class_name = '16APSK'
        elif n == 32:
            class_name = '32APSK'
        elif n == 64:
            class_name = '64APSK'
        else:
            class_name = 'APSK'
        
        super().__init__(signalparameter, class_name)
        self.status = n
        
        # 设置APSK参数
        self.symbol_rate = np.random.uniform(2e6, 10e6)  # 2-10 Msps
        self.rolloff_factor = np.random.uniform(0.15, 0.4)
        self.upsample_factor = np.random.randint(4, 12)
        
        # 计算占用带宽
        self.bandwidth = self.symbol_rate * (1 + self.rolloff_factor)
        
        # 设置频率边界（假设中心频率为0，会在signal_gen.py中重新设置）
        self.center_frequency = 0  # 临时值，会被signal_gen.py覆盖
        self.upper_frequency = self.center_frequency + self.bandwidth/2
        self.lower_frequency = self.center_frequency - self.bandwidth/2
        
        # 更新信号描述中的带宽信息（归一化）
        self.signal_description.bandwidth = self.bandwidth / self.sample_rate
        self.signal_description.upper_frequency = self.upper_frequency / self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency / self.sample_rate
        
        # 创建APSK调制器
        try:
            self.modem = APSKModem(f'{n}APSK')
            self.apsk_modem = self.modem  # 兼容性别名
        except:
            # 如果不支持，回退到16APSK
            self.modem = APSKModem('16APSK')
            self.apsk_modem = self.modem
            self.status = 16
    
    def __call__(self):
        """生成APSK信号"""
        # 初始化iq_data
        self.iq_data = np.zeros(self.num_iq_samples, dtype=np.complex64)
        
        # 计算所需的时间长度（类似QAM的处理方式）
        duration = self.stop - self.start
        
        # 设置合理的参数以确保足够长的APSK信号
        min_duration_ratio = 0.18  # 最小18%时间（比QAM稍大）
        max_duration_ratio = 0.68  # 最大68%时间（比QAM稍大）
        total_time = self.num_iq_samples / self.sample_rate
        
        # 调整持续时间，确保信号足够长
        target_duration = np.random.uniform(min_duration_ratio, max_duration_ratio) * total_time
        duration = max(duration, target_duration)
        
        # 更新停止时间
        self.stop = min(self.start + duration, total_time * 0.9)  # 确保不超出边界
        duration = self.stop - self.start
        
        # 保持原有的符号率，但调整符号数量来适应新的持续时间
        target_symbols = max(50, int(duration * self.symbol_rate))  # 根据持续时间计算符号数
        num_symbols = min(target_symbols, target_symbols * 2)  # 允许更多符号
        
        # 重新计算带宽和频率边界（更新可能在signal_gen.py中修改的center_frequency）
        self.bandwidth = self.symbol_rate * (1 + self.rolloff_factor)
        self.upper_frequency = self.center_frequency + self.bandwidth/2
        self.lower_frequency = self.center_frequency - self.bandwidth/2
        
        # 更新信号描述中的带宽信息
        self.signal_description.bandwidth = self.bandwidth / self.sample_rate
        self.signal_description.upper_frequency = self.upper_frequency / self.sample_rate
        self.signal_description.lower_frequency = self.lower_frequency / self.sample_rate
        
        # 计算实际时间参数
        oversampled_sample_rate = self.symbol_rate * self.upsample_factor
        
        # 生成随机数据
        bits_per_symbol = int(np.log2(self.status))
        data_bits = np.random.randint(0, 2, num_symbols * bits_per_symbol)
        
        # APSK调制
        modulated_symbols = self.apsk_modem.modulate(data_bits)
        
        # 上采样：在符号之间插入零
        upsampled_signal = np.zeros(len(modulated_symbols) * self.upsample_factor, dtype=np.complex64)
        upsampled_signal[::self.upsample_factor] = modulated_symbols
        
        # 设计根升余弦滤波器
        filter_span = 6  # 滤波器跨度
        filter_order = filter_span * self.upsample_factor
        
        # 生成根升余弦滤波器的时间轴
        t_filter = np.arange(-filter_order//2, filter_order//2 + 1) / oversampled_sample_rate
        
        # 根升余弦滤波器实现
        rrc_filter = self._root_raised_cosine_filter(t_filter, self.symbol_rate, self.rolloff_factor)
        
        # 应用滤波器（卷积）
        filtered_signal = np.convolve(upsampled_signal, rrc_filter, mode='same')
        
        # APSK信号的功率归一化
        if np.std(filtered_signal) > 0:
            # APSK具有多环结构，需要适当的归一化
            filtered_signal = filtered_signal / np.std(filtered_signal) * 0.7
        
        # 重采样到系统采样率
        if oversampled_sample_rate != self.sample_rate:
            # 计算重采样后的长度
            target_length = int(len(filtered_signal) * self.sample_rate / oversampled_sample_rate)
            # 简单的线性插值重采样
            old_indices = np.linspace(0, len(filtered_signal)-1, len(filtered_signal))
            new_indices = np.linspace(0, len(filtered_signal)-1, target_length)
            final_signal = np.interp(new_indices, old_indices, filtered_signal.real) + \
                          1j * np.interp(new_indices, old_indices, filtered_signal.imag)
        else:
            final_signal = filtered_signal
        
        # 确保信号长度不超过预期
        max_samples = int(duration * self.sample_rate)
        if len(final_signal) > max_samples:
            final_signal = final_signal[:max_samples]
        
        # 计算实际的时间长度
        actual_duration = len(final_signal) / self.sample_rate
        self.stop = self.start + actual_duration
        
        # APSK边界框优化：类似QAM的处理方式，但针对APSK多环结构调整
        time_extension_factor = 1.006  # 时间向右延长0.6%（比QAM稍大）
        freq_extension_factor = 1.6   # 频率高度增加60%（比QAM更大，因为APSK多环结构）
        
        # 扩展时间边界
        extended_duration = actual_duration * time_extension_factor
        extended_stop = min(self.start + extended_duration, 
                          (self.num_iq_samples/self.sample_rate) * 0.95)
        
        # 扩展频率边界
        current_bandwidth = self.upper_frequency - self.lower_frequency
        extended_bandwidth = current_bandwidth * freq_extension_factor
        bandwidth_increase = extended_bandwidth - current_bandwidth
        
        # 对称扩展频率范围
        extended_lower_freq = self.lower_frequency - bandwidth_increase / 2
        extended_upper_freq = self.upper_frequency + bandwidth_increase / 2
        
        # 确保频率不超出奈奎斯特范围
        max_freq = self.sample_rate / 2 * 0.95
        min_freq = -self.sample_rate / 2 * 0.95
        extended_lower_freq = max(extended_lower_freq, min_freq)
        extended_upper_freq = min(extended_upper_freq, max_freq)
        
        # 更新信号描述的边界信息（使用扩展后的边界）
        self.signal_description.stop = self.start / (self.num_iq_samples/self.sample_rate) + \
                                      (extended_stop - self.start) / (self.num_iq_samples/self.sample_rate)
        self.signal_description.upper_frequency = extended_upper_freq / self.sample_rate
        self.signal_description.lower_frequency = extended_lower_freq / self.sample_rate
        self.signal_description.bandwidth = (extended_upper_freq - extended_lower_freq) / self.sample_rate
        
        # 生成载波调制后的时间轴
        time_axis = np.linspace(0, actual_duration, len(final_signal))
        
        # 应用载波调制
        carrier_modulated = final_signal * np.exp(1j * 2 * np.pi * self.center_frequency * time_axis)
        
        # 放入IQ数据数组
        start_idx = int(np.floor(self.start * self.sample_rate))
        end_idx = min(start_idx + len(carrier_modulated), self.num_iq_samples)
        actual_length = end_idx - start_idx
        
        self.iq_data[start_idx:end_idx] = carrier_modulated[:actual_length]
    
    def _root_raised_cosine_filter(self, t, symbol_rate, rolloff):
        """
        生成根升余弦滤波器 (与PSK/QAM使用相同的实现)
        """
        T = 1.0 / symbol_rate  # 符号周期
        
        # 避免除零
        epsilon = 1e-10
        t = t + epsilon * (t == 0)
        
        # 根升余弦滤波器公式
        if rolloff == 0:
            h = np.sinc(t / T)
        else:
            # 处理特殊点 t = ±T/(4*rolloff)
            special_points = np.abs(np.abs(t) - T/(4*rolloff)) < epsilon
            
            h = np.zeros_like(t)
            
            # 一般情况
            normal_points = ~special_points
            t_norm = t[normal_points]
            
            numerator = np.sin(np.pi * t_norm / T * (1 - rolloff)) + \
                       4 * rolloff * t_norm / T * np.cos(np.pi * t_norm / T * (1 + rolloff))
            denominator = np.pi * t_norm / T * (1 - (4 * rolloff * t_norm / T) ** 2)
            
            h[normal_points] = numerator / denominator
            
            # 特殊点的值
            h[special_points] = rolloff / np.sqrt(2) * \
                              ((1 + 2/np.pi) * np.sin(np.pi/(4*rolloff)) + \
                               (1 - 2/np.pi) * np.cos(np.pi/(4*rolloff)))
        
        return h

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
        self.stop = self.start+120e-6
        self.samples_per_symbol = 1e-6/(1/signalparameter.sample_rate)#int(signalparameter.samples_per_symbol/2)*2
        self.total_samples = self.samples_per_symbol*120
        self.frame = np.zeros(112,dtype=np.int32)
        self.upper_frequency = self.center_frequency
        self.lower_frequency = self.center_frequency
        self.signal_description.upper_frequency = self.signal_description.center_frequency
        self.signal_description.lower_frequency = self.signal_description.center_frequency
        # self.center_frequency = 1090e6
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