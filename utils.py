import torch


def padding(data0, data1, data2, data3, data4, pad_idx=0):
    len0, len1, len2, len3, len4 = data0.shape[1], data1.shape[1], data2.shape[1], data3.shape[1], data4.shape[1]
    max_len = max(len0, len1, len2, len3, len4)
    data0 = torch.cat([data0, torch.ones(data0.shape[0], max_len - len0).long() * pad_idx], dim=1)
    data1 = torch.cat([data1, torch.ones(data1.shape[0], max_len - len1).long() * pad_idx], dim=1)
    data2 = torch.cat([data2, torch.ones(data2.shape[0], max_len - len2).long() * pad_idx], dim=1)
    data3 = torch.cat([data3, torch.ones(data3.shape[0], max_len - len3).long() * pad_idx], dim=1)
    data4 = torch.cat([data4, torch.ones(data4.shape[0], max_len - len4).long() * pad_idx], dim=1)
    return data0, data1, data2, data3, data4


def batch_padding(data, pad_idx=0):
    max_len = 0
    for text in data:
        max_len = max(max_len, len(text))
    for i in range(len(data)):
        data[i] += [pad_idx] * (max_len - len(data[i]))
    return torch.tensor(data)

def padding_2(data1, data2, pad_idx=0):
    len1, len2 = data1.shape[1], data2.shape[1]
    if len1 > len2:
        data2 = torch.cat([data2, torch.ones(data2.shape[0], len1 - len2).long() * pad_idx], dim=1)
    elif len2 > len1:
        data1 = torch.cat([data1, torch.ones(data1.shape[0], len2 - len1).long() * pad_idx], dim=1)
    return data1, data2

