## 仅用于测试，输出信号的时域和时频，以及频谱图和标签框
from SignalDef import NLFM,Costas,Frank,LFM,nFSK,nQAM,AM,DSB,FM,nPSK,RADAR_Pulse,RADAR_Pulse_Compressed,MSK,ADSB,SignalParameter,ConvertDescriptionToPatch
from utils.transforms import Spectrogram,Normalize
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import cv2,csv




fft_size = 512 #定义图像为正方形
# num_iq_samples = fft_size * fft_size
noverlap = fft_size*0.75 #重叠点数
nperseg = fft_size #窗长
# 时域长度：窗长+（点数-1）*（窗长-重叠）
num_iq_samples = (fft_size-1)*(fft_size-noverlap)+fft_size
stft = Spectrogram(nperseg=nperseg, noverlap=noverlap, nfft=fft_size,mode='magnitude') # 默认布莱克曼窗

tmp = np.zeros(int(num_iq_samples),dtype=np.complex64)
f = open('ofdm.csv','r')
for j,i in enumerate(f):
    # if j<20000:
    #     continue
    if j>int(num_iq_samples)-1:
        break
    tmp[j] = complex(i.replace('i','j'))
    # tmp[j] = complex(i.replace('i','j'))
    pass

a=SignalParameter(num_iq_samples=num_iq_samples,sample_rate=2.5e9)
# a.samples_per_symbol = np.random.randint(2000,5000)
# a.bandwidth = 0.02
# a()
# a.bandwidth=0.001*a.sample_rate
# LFMs = NLFM(a)
# LFMs()


normalize = Normalize(np.inf,flatten=True)

# a,b,c = ConvertDescriptionToPatch(LFMs.signal_description,nfft=fft_size)

# test_Rectangle1 = plt.Rectangle(a, width=b, height=c, angle=0, fill=False,edgecolor='r')
# plt.plot(np.real(LFMs.iq_data))
# plt.text(111,111, "hello", va='top')
# plt.savefig('timeDomain.png')
tf=stft(tmp)

# tf=normalize(tf)
# result = np.zeros((512,512),dtype=np.int16)
# result = tf*255
# result = result.astype(np.int16)
# fig = np.expand_dims(result,0)
# fig = np.repeat(fig,3,0)

# cv2.imwrite('tmp.png',fig.transpose(1,2,0))
# plt.figure(figsize=(5.12,5.12),dpi=100)
heatmapshow = cv2.normalize(tf, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# x_sobel = cv2.Sobel(heatmapshow,cv2.CV_64F,1,0,ksize=19)
# y_sobel = cv2.Sobel(heatmapshow,cv2.CV_64F,0,1,ksize=19)
# heatmapshow = cv2.addWeighted(x_sobel, 0.5, y_sobel, 0.5, 0)
# heatmapshow = np.uint8(heatmapshow)
# heatmapshow = heatmapshow.astype(cv2.CV_8U)
        # heatmapshow = cv2.normalize(tf, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
cv2.imwrite('tmp.png',heatmapshow)

# px = 1/plt.rcParams['figure.dpi']  # pixel in inches
# fig, ax = plt.subplots(figsize=(1024*px,1024*px))

# # fig, ax = plt.subplots(figsize = (10,10))
# ax.add_patch(test_Rectangle1)
# font={
#     'style': "italic",
#     'weight': "normal",
#     'fontsize':8
# }
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.margins(0,0)
# ax.text(a[0],a[1],LFMs.signal_description.class_name,fontdict=font)
# plt.axis('off')
# plt.imshow(np.abs(tf))
# # plt.figure(figsize=(5.12,5.12))
# plt.show()
# plt.savefig('tmp.png',pad_inches = 0, bbox_inches='tight')

# pass