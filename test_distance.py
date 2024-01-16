import torch

# 假设你有一个点集合，每个元素是（x, y）坐标。
# 这里使用一个示例点集合，你可以替换为实际的数据。
points = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0],
                       [7.0, 8.0]])

# 使用广播操作计算每个点到其他所有点的二维欧氏距离之和。
# 首先，计算坐标差值。
x_diff = points[:, 0][:, None] - points[:, 0]
y_diff = points[:, 1][:, None] - points[:, 1]
print(x_diff)
print(y_diff)
# 计算欧氏距离。
distances = torch.sqrt(x_diff**2 + y_diff**2)

# 计算每个点到其他所有点的欧氏距离之和。
distance_sums = distances.sum(dim=1)

# 'distance_sums' 张量将包含每个点到其他所有点的二维欧氏距离之和。
print(distance_sums)