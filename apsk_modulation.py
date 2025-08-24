#!/usr/bin/env python3
"""
APSK调制实现
使用commpy的基础类来实现APSK (Amplitude Phase Shift Keying)
"""

import numpy as np
import commpy as cpy
import matplotlib.pyplot as plt
from commpy.modulation import Modem

class APSKModem(Modem):
    """
    APSK (Amplitude Phase Shift Keying) 调制器/解调器
    
    APSK是一种结合了幅度和相位调制的方案，
    星座点分布在不同半径的同心圆上。
    """
    
    def __init__(self, constellation_type='16APSK', radii=None, points_per_ring=None):
        """
        初始化APSK调制器
        
        Parameters:
        -----------
        constellation_type : str
            APSK类型，如 '16APSK', '32APSK', '64APSK'
        radii : list
            每个环的半径，如果为None则使用默认值
        points_per_ring : list
            每个环上的点数，如果为None则使用默认值
        """
        
        self.constellation_type = constellation_type
        
        # 根据APSK类型设置默认参数
        if constellation_type == '16APSK':
            # DVB-S2标准16APSK: 4+12结构
            if radii is None:
                radii = [1.0, 2.2]  # 内环半径1.0，外环半径2.2
            if points_per_ring is None:
                points_per_ring = [4, 12]  # 内环4个点，外环12个点
                
        elif constellation_type == '32APSK':
            # DVB-S2标准32APSK: 4+12+16结构
            if radii is None:
                radii = [1.0, 2.2, 3.5]
            if points_per_ring is None:
                points_per_ring = [4, 12, 16]
                
        elif constellation_type == '64APSK':
            # 64APSK: 4+12+20+28结构
            if radii is None:
                radii = [1.0, 2.2, 3.5, 4.8]
            if points_per_ring is None:
                points_per_ring = [4, 12, 20, 28]
                
        else:
            # 对于自定义类型，使用传入的参数
            if radii is None or points_per_ring is None:
                raise ValueError(f"自定义APSK类型需要提供radii和points_per_ring参数")
        
        self.radii = radii
        self.points_per_ring = points_per_ring
        self.num_rings = len(radii)
        self.total_points = sum(points_per_ring)
        
        # 生成星座图
        constellation = self._generate_constellation()
        
        # 调用父类初始化
        super().__init__(constellation)
        
    def _generate_constellation(self):
        """生成APSK星座图"""
        constellation = []
        
        for ring_idx, (radius, num_points) in enumerate(zip(self.radii, self.points_per_ring)):
            # 每个环上均匀分布点
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            
            # 为了优化性能，可以旋转某些环
            if ring_idx == 1:  # 第二个环旋转π/12
                angles += np.pi / 12
            elif ring_idx == 2:  # 第三个环旋转π/16
                angles += np.pi / 16
                
            # 生成复数星座点
            for angle in angles:
                point = radius * np.exp(1j * angle)
                constellation.append(point)
        
        return np.array(constellation)
    
    def plot_constellation(self, title=None):
        """绘制星座图"""
        plt.figure(figsize=(8, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        start_idx = 0
        
        for ring_idx, num_points in enumerate(self.points_per_ring):
            end_idx = start_idx + num_points
            ring_points = self.constellation[start_idx:end_idx]
            
            plt.scatter(ring_points.real, ring_points.imag, 
                       c=colors[ring_idx % len(colors)], 
                       s=50, 
                       label=f'环{ring_idx+1} (r={self.radii[ring_idx]:.1f})',
                       alpha=0.7)
            
            # 画圆环
            circle = plt.Circle((0, 0), self.radii[ring_idx], 
                              fill=False, linestyle='--', 
                              color=colors[ring_idx % len(colors)], alpha=0.3)
            plt.gca().add_patch(circle)
            
            start_idx = end_idx
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlabel('同相分量 (I)')
        plt.ylabel('正交分量 (Q)')
        
        if title is None:
            title = f'{self.constellation_type} 星座图'
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def get_info(self):
        """获取APSK调制器信息"""
        info = {
            'type': self.constellation_type,
            'num_rings': self.num_rings,
            'radii': self.radii,
            'points_per_ring': self.points_per_ring,
            'total_points': self.total_points,
            'bits_per_symbol': int(np.log2(self.total_points))
        }
        return info

def test_apsk():
    """测试APSK调制"""
    print("🚀 测试APSK调制功能")
    print("=" * 50)
    
    # 测试不同的APSK类型
    apsk_types = ['16APSK', '32APSK', '64APSK']
    
    for apsk_type in apsk_types:
        print(f"\n--- {apsk_type} ---")
        
        try:
            # 创建APSK调制器
            modem = APSKModem(apsk_type)
            info = modem.get_info()
            
            print(f"环数: {info['num_rings']}")
            print(f"总星座点数: {info['total_points']}")
            print(f"每符号比特数: {info['bits_per_symbol']}")
            print(f"环半径: {info['radii']}")
            print(f"每环点数: {info['points_per_ring']}")
            
            # 生成测试数据
            num_symbols = 100
            data_bits = np.random.randint(0, 2, 
                                        num_symbols * info['bits_per_symbol'])
            
            # 调制
            modulated_symbols = modem.modulate(data_bits)
            print(f"调制了 {len(modulated_symbols)} 个符号")
            
            # 解调
            demodulated_bits = modem.demodulate(modulated_symbols, 'hard')
            
            # 计算误差
            bit_errors = np.sum(data_bits != demodulated_bits)
            ber = bit_errors / len(data_bits)
            
            print(f"误码率 (无噪声): {ber:.6f}")
            
            if ber == 0:
                print("✅ 调制/解调测试通过")
            else:
                print("⚠️  调制/解调存在误差")
                
        except Exception as e:
            print(f"❌ {apsk_type} 测试失败: {e}")
    
    # 绘制星座图示例
    print(f"\n📊 绘制16APSK星座图...")
    try:
        modem_16apsk = APSKModem('16APSK')
        modem_16apsk.plot_constellation()
    except Exception as e:
        print(f"绘图失败: {e}")

def create_custom_apsk():
    """创建自定义APSK配置"""
    print("\n🛠️  创建自定义APSK配置")
    print("=" * 40)
    
    # 自定义8APSK: 4+4结构
    custom_radii = [1.0, 2.0]
    custom_points = [4, 4]
    
    try:
        custom_modem = APSKModem('8APSK', radii=custom_radii, 
                               points_per_ring=custom_points)
        print("✅ 成功创建8APSK调制器")
        
        info = custom_modem.get_info()
        print(f"自定义配置: {info}")
        
        return custom_modem
        
    except Exception as e:
        print(f"❌ 自定义APSK创建失败: {e}")
        return None

if __name__ == "__main__":
    test_apsk()
    create_custom_apsk()
    
    print(f"\n💡 使用说明:")
    print("=" * 30)
    print("1. APSK可以用于DVB-S2等标准")
    print("2. 支持16APSK, 32APSK, 64APSK")
    print("3. 可以自定义环半径和每环点数")
    print("4. 基于commpy的Modem基类实现")
    print("5. 支持硬判决和软判决解调")
