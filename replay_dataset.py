import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class replay_dataset(Dataset):
    def __init__(self, replay_buffer, view):
        self.xs = {}
        self.replay_buffer = replay_buffer

        self.view = view
        for i in range(view):
            x_name = f"x{i}"
            self.xs[x_name] = replay_buffer[i]


        self.y = torch.ones_like(replay_buffer[0])

    def __len__(self):
        return len(self.replay_buffer[0])

    def __getitem__(self, idx):
        x_ = [self.xs[f"x{i}"][idx] for i in range(len(self.xs))]

        return x_,self.y[idx],torch.from_numpy(np.array(idx)).long(),

# 假设有样本和标签数据
def test():
    samples = [[1,2,3],[2,2,2],[1,3,4]]
    labels = [1,3,5]

    # 创建自定义数据集对象
    dataset = replay_dataset(samples, 0,3)

    # 创建数据加载器
    batch_size = 2
    shuffle = True
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # 使用数据加载器迭代数据
    for batch_samples, batch_labels, idx in data_loader:
        print("Batch samples:", batch_samples)
        print("Batch labels:", batch_labels)

if __name__ == '__main__':
    test()