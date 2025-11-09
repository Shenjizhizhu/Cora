import os
import pickle
import numpy as np

# --------------------------
# 配置路径（无需修改，确保和步骤2一致）
# --------------------------
# 原始CIFAR-10数据路径（pickle格式）
original_data_dir = "./data/cifar-10-batches-py"
# 转换后numpy格式数据的保存路径（目标路径）
npy_data_dir = "./data/cifar-10-npy"

# 创建保存numpy数据的目录（如果不存在）
os.makedirs(npy_data_dir, exist_ok=True)

# --------------------------
# 检查原始数据是否存在
# --------------------------
if not os.path.exists(original_data_dir):
    raise FileNotFoundError(
        f"未找到原始CIFAR-10数据，请确认路径：{original_data_dir}\n"
        "请按照步骤2解压原始数据到该目录"
    )

# 检查训练集批次文件是否齐全
for i in range(1, 6):
    batch_file = os.path.join(original_data_dir, f"data_batch_{i}")
    if not os.path.exists(batch_file):
        raise FileNotFoundError(f"缺失原始训练集文件：{batch_file}")

# --------------------------
# 转换训练集数据为 train_data.npy
# --------------------------
print("开始转换训练集数据...")
train_data_list = []  # 存储所有训练图像数据

# 读取5个训练批次文件并合并
for i in range(1, 6):
    batch_path = os.path.join(original_data_dir, f"data_batch_{i}")
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")  # 读取pickle格式数据
    # 提取当前批次的图像数据（形状：(10000, 3072)，3072=32×32×3）
    train_data_list.append(batch["data"])
    print(f"已读取第{i}/5个训练批次")

# 合并所有训练数据（总样本数：50000，形状：(50000, 3072)）
train_data = np.concatenate(train_data_list, axis=0)

# 保存为 train_data.npy（目标文件）
train_data_path = os.path.join(npy_data_dir, "train_data.npy")
np.save(train_data_path, train_data)
print(f"训练集数据已保存至：{train_data_path}")

# --------------------------
# 同时生成训练标签文件（可选，但主程序需要）
# --------------------------
train_labels_list = []
for i in range(1, 6):
    batch_path = os.path.join(original_data_dir, f"data_batch_{i}")
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    train_labels_list.extend(batch["labels"])  # 提取标签

train_labels = np.array(train_labels_list, dtype="int32")
train_labels_path = os.path.join(npy_data_dir, "train_labels.npy")
np.save(train_labels_path, train_labels)
print(f"训练集标签已保存至：{train_labels_path}")

# --------------------------
# 转换测试集数据（主程序也需要，一并生成）
# --------------------------
test_batch_path = os.path.join(original_data_dir, "test_batch")
with open(test_batch_path, "rb") as f:
    test_batch = pickle.load(f, encoding="latin1")

test_data = test_batch["data"]  # 形状：(10000, 3072)
test_labels = np.array(test_batch["labels"], dtype="int32")

# 保存测试集
np.save(os.path.join(npy_data_dir, "test_data.npy"), test_data)
np.save(os.path.join(npy_data_dir, "test_labels.npy"), test_labels)
print("测试集数据和标签已生成")

print("所有转换完成！")