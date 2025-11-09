import os
import pickle
import numpy as np

original_data_dir = "./data/cifar-10-batches-py"
npy_data_dir = "./data/cifar-10-npy"

os.makedirs(npy_data_dir, exist_ok=True)

if not os.path.exists(original_data_dir):
    raise FileNotFoundError(
        f"未找到原始CIFAR-10数据，请确认路径：{original_data_dir}\n"
        "请按照步骤2解压原始数据到该目录"
    )

for i in range(1, 6):
    batch_file = os.path.join(original_data_dir, f"data_batch_{i}")
    if not os.path.exists(batch_file):
        raise FileNotFoundError(f"缺失原始训练集文件：{batch_file}")

print("开始转换训练集数据...")
train_data_list = []  

for i in range(1, 6):
    batch_path = os.path.join(original_data_dir, f"data_batch_{i}")
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    train_data_list.append(batch["data"])
    print(f"已读取第{i}/5个训练批次")

train_data = np.concatenate(train_data_list, axis=0)

train_data_path = os.path.join(npy_data_dir, "train_data.npy")
np.save(train_data_path, train_data)
print(f"训练集数据已保存至：{train_data_path}")

train_labels_list = []
for i in range(1, 6):
    batch_path = os.path.join(original_data_dir, f"data_batch_{i}")
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    train_labels_list.extend(batch["labels"])

train_labels = np.array(train_labels_list, dtype="int32")
train_labels_path = os.path.join(npy_data_dir, "train_labels.npy")
np.save(train_labels_path, train_labels)
print(f"训练集标签已保存至：{train_labels_path}")

test_batch_path = os.path.join(original_data_dir, "test_batch")
with open(test_batch_path, "rb") as f:
    test_batch = pickle.load(f, encoding="latin1")

test_data = test_batch["data"]
test_labels = np.array(test_batch["labels"], dtype="int32")

np.save(os.path.join(npy_data_dir, "test_data.npy"), test_data)
np.save(os.path.join(npy_data_dir, "test_labels.npy"), test_labels)
print("测试集数据和标签已生成")

print("所有转换完成！")