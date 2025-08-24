#!/usr/bin/env python3
"""
APSK信号类 - 可以集成到SignalDef.py中
"""

import numpy as np
import commpy as cpy
from commpy.modulation import Modem
from SignalDescription import SignalDescription, SignalData
from SignalDef import SignalParameter, BaseSignal

class APSKModem(Modem):
    """APSK调制器 - 精简版，适用于信号生成"""
    
    def __init__(self, constellation_type='16APSK'):
        # APSK配置
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
        """生成APSK星座图"""
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

class nAPSK(BaseSignal):
    """APSK信号类"""
    
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
        self.signal_description.bandwidth = self.symbol_rate * (1 + self.rolloff_factor)
        
        # 创建APSK调制器
        try:
            self.apsk_modem = APSKModem(f'{n}APSK')
        except:
            # 如果不支持，回退到16APSK
            self.apsk_modem = APSKModem('16APSK')
            self.status = 16
    
    def __call__(self):
        """生成APSK信号"""
        # 初始化iq_data
        self.iq_data = np.zeros(self.num_iq_samples, dtype=np.complex64)
        
        # 计算符号数
        symbol_duration = 1.0 / self.symbol_rate
        total_duration = self.signal_description.stop - self.signal_description.start
        num_symbols = int(total_duration / symbol_duration)
        num_symbols = max(num_symbols, 10)  # 至少10个符号
        
        # 生成随机数据
        bits_per_symbol = int(np.log2(self.status))
        data_bits = np.random.randint(0, 2, num_symbols * bits_per_symbol)
        
        # APSK调制
        modulated_symbols = self.apsk_modem.modulate(data_bits)
        
        # 上采样和根升余弦滤波
        upsampled_signal = self._apply_rrc_filter(modulated_symbols)
        
        # 调整信号长度
        target_length = int(total_duration * self.signal_description.sample_rate)
        if len(upsampled_signal) > target_length:
            upsampled_signal = upsampled_signal[:target_length]
        elif len(upsampled_signal) < target_length:
            # 用零填充
            padding = target_length - len(upsampled_signal)
            upsampled_signal = np.concatenate([upsampled_signal, np.zeros(padding)])
        
        # 频率调制
        t = np.arange(len(upsampled_signal)) / self.signal_description.sample_rate
        carrier = np.exp(1j * 2 * np.pi * self.signal_description.center_frequency * t)
        modulated_signal = upsampled_signal * carrier
        
        # 插入到完整信号中
        start_sample = int(self.signal_description.start * self.num_iq_samples)
        self.iq_data[start_sample:start_sample + len(modulated_signal)] = modulated_signal
        
        return self.iq_data
    
    def _apply_rrc_filter(self, symbols):
        """应用根升余弦滤波器"""
        try:
            # 使用commpy的根升余弦滤波器
            filter_length = 101
            _, rrc_filter = cpy.rrcosfilter(filter_length, self.rolloff_factor, 
                                          1.0/self.upsample_factor, 1/self.symbol_rate)
            
            # 上采样
            upsampled = np.zeros(len(symbols) * self.upsample_factor, dtype=complex)
            upsampled[::self.upsample_factor] = symbols
            
            # 滤波
            filtered_signal = np.convolve(upsampled, rrc_filter, mode='same')
            
            return filtered_signal
            
        except Exception as e:
            print(f"RRC滤波失败，使用简单上采样: {e}")
            # 简单上采样作为后备
            upsampled = np.repeat(symbols, self.upsample_factor)
            return upsampled

def test_apsk_signal():
    """测试APSK信号生成"""
    print("🧪 测试APSK信号生成")
    print("=" * 40)
    
    # 创建信号参数
    from SignalDef import SignalParameter
    
    signal_param = SignalParameter(num_iq_samples=512*512, sample_rate=2.5e9)
    signal_param.start = 0.1
    signal_param.stop = 0.9
    signal_param.center_frequency = 0
    signal_param()
    
    # 测试不同APSK类型
    apsk_types = [16, 32, 64]
    
    for apsk_type in apsk_types:
        print(f"\n--- 测试 {apsk_type}APSK ---")
        
        try:
            # 创建APSK信号
            apsk_signal = nAPSK(signal_param, apsk_type)
            
            print(f"信号类型: {apsk_signal.signal_description.class_name}")
            print(f"符号率: {apsk_signal.symbol_rate/1e6:.2f} Msps")
            print(f"带宽: {apsk_signal.signal_description.bandwidth/1e6:.2f} MHz")
            print(f"滚降因子: {apsk_signal.rolloff_factor:.3f}")
            
            # 生成信号
            iq_data = apsk_signal()
            
            print(f"生成信号长度: {len(iq_data)}")
            print(f"信号功率: {np.mean(np.abs(iq_data)**2):.6f}")
            print("✅ 信号生成成功")
            
        except Exception as e:
            print(f"❌ {apsk_type}APSK 生成失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_apsk_signal()
    
    print(f"\n💡 集成说明:")
    print("=" * 30)
    print("1. 将APSKModem和nAPSK类复制到SignalDef.py中")
    print("2. 在signal_gen.py的default_class中添加APSK类型")
    print("3. 在genFrame方法中添加APSK处理逻辑")
    print("4. APSK支持16/32/64阶调制")
    print("5. 使用DVB-S2标准的环配置")
