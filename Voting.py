import torch


def hard_voting(model_outputs):
    stacked_outputs = torch.stack(model_outputs,dim=0) #5 1 3 4800
    #每个位置输出索引
    class_indices = torch.argmax(stacked_outputs,dim=2) # 5 1 4800
    #众数作为最终预测
    value,index = torch.mode(class_indices,dim=0)
    return value
