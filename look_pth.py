import torch

model = torch.load('/home/sherry/lyh/mae/logs/checkpoint-2.pth')  # 加载已保存的模型
model.state_dict()
print(model.state_dict())  # 打印所有键
# print(model)  # 打印模型的结构

