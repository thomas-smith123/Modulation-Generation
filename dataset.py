#
# @Date: 2024-09-30 15:38:32
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-11-01 19:40:24
# @FilePath: /hy_bak_test_delete_after_used/complex_gru_single_radioml/dataset.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import torch
from torch.utils.data import Dataset, DataLoader
import random

class RandomShiftDataset(Dataset):
    def __init__(self, data,labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取样本
        sample = self.data[idx]
        label = self.labels[idx]
        operate = random.randint(0, 2)  # 随机生成0或1,2
        
        if operate == 0:
            return sample,label
        elif operate == 1:
            l = sample.size(-1)
            # 随机生成移位步数
            shift = random.randint(-l//4, l//4)  # 例如，-2到2之间的随机整数
            shifted_sample = torch.zeros_like(sample,dtype=sample.dtype)
            # 执行移位操作
            if shift > 0:
                shifted_sample[..., shift:] = sample[..., shift:]
                # shifted_sample = torch.cat((sample[shift:], torch.zeros(shift, dtype=sample.dtype)))
            elif shift < 0:
                shift = abs(shift)
                shifted_sample[..., :-shift] = sample[..., :-shift]
                
                # shifted_sample = torch.cat((torch.zeros(shift, dtype=sample.dtype), sample[:-shift]))
            else:
                shifted_sample = sample
            
            return shifted_sample,label
        else: ##直接从中间截取
            l = sample.size(-1)
            start = random.randint(0, l//5)
            duration = random.randint(l//5, int(l*3/5))
            end = start + duration
            sample_ = torch.zeros_like(sample,dtype=sample.dtype)
            sample_[..., start:end] = sample[..., start:end]
            return sample_,label
            
            
if __name__ == '__main__':
    # 示例数据：多个数组
    data = torch.tensor([[1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15]])

    # 创建数据集和数据加载器
    dataset = RandomShiftDataset(data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 测试 DataLoader
    for batch in dataloader:
        print(batch)

