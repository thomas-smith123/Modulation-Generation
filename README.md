<!--
 * @Date: 2025-08-17 10:33:03
 * @LastEditors: thomas-smith123 thomas-smith@live.cn
 * @LastEditTime: 2025-08-30 15:31:42
 * @FilePath: \MG-orphan\README.md
-->

## 文件说明
支持的信号样式
- "LFM",# 0
- "2FSK",# 1
- "Costas",# 2
- "Frequency_HOP",# 4
- "16QAM",# 5
- "64QAM",# 6
- "256QAM",# 7
- "4PSK",# 8
- "8PSK",# 9
- "16PSK",# 10
- "32PSK",# 11
- "16APSK",# 12
- "32APSK",# 13
- "64APSK",# 14
- "RadarPulse_Compressed",# 15
- "NLFM",# 16
- "ADSB",# 17
- "OFDM", #18
- "Zigbee", #19
- "LoRa", #20

## 说明
OFDM、Zigbee、LoRa调制方式的信号是由Matlab产生，分别保存为名为20_QPSK_OFDM.mat，zigbee.mat，LoRa.mat的文件，然后在产生样本的时候进行随机截取就行。

## 样例
我们产生了如下所示的一系列样本，输出的图像是不同colormap下的时频图。
<div>
    <img width=150 height=150 src=example_pics/1839_-5.png></a>
    <img width=150 height=150 src=example_pics/1840_-5.png></a>
    <img width=150 height=150 src=example_pics/1841_-5.png></a>
    <img width=150 height=150 src=example_pics/1842_-5.png></a>
</div>
<div>
    <img width=150 height=150 src=example_pics/1843_-5.png></a>
    <img width=150 height=150 src=example_pics/1844_-5.png></a>
    <img width=150 height=150 src=example_pics/1845_-5.png></a>
    <img width=150 height=150 src=example_pics/1846_-5.png></a>
</div>
