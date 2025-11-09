import jax
import jax.numpy as jnp
import numpy as np
import os
from flax import nnx
import optax
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    data_dir = "cifar-10-npy" 

    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.data, self.labels = self._load_data()

    def _load_data(self):
        full_data_dir = os.path.join(self.root, self.data_dir)
        if self.train:
            data = np.load(os.path.join(full_data_dir, "train_data.npy"))
            labels = np.load(os.path.join(full_data_dir, "train_labels.npy"))
        else:
            data = np.load(os.path.join(full_data_dir, "test_data.npy"))
            labels = np.load(os.path.join(full_data_dir, "test_labels.npy"))

        return data.astype("float32"), labels.astype("int32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class Block(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = jax.nn.relu(x)
        return x


class Model(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, rngs: nnx.Rngs):
        self.block = Block(din, dmid, rngs=rngs)
        self.linear = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        x = self.block(x)
        x = self.linear(x)
        return x


@nnx.jit
def train_step(model, inputs, labels):
    def loss_fn(model):
        logits = model(inputs)
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(preds == labels)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean(), accuracy

    (loss, accuracy), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    _, params, rest = nnx.split(model, nnx.Param, ...)
    params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
    nnx.update(model, nnx.merge_state(params, rest))
    
    return loss, accuracy  


def load_cifar10():
    def transform(x):
        return x / 255.0

    train_dataset = CIFAR10Dataset(root="./data", train=True, transform=transform)
    test_dataset = CIFAR10Dataset(root="./data", train=False, transform=transform)

    x_train = np.array([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))]).ravel()
    x_test = np.array([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))]).ravel()

    return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    print(f"训练数据形状: {x_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试数据形状: {x_test.shape}, 标签形状: {y_test.shape}")
    
    model = Model(3072, 256, 10, rngs=nnx.Rngs(0))
    
    assert model.linear.bias.value.shape == (10,)
    assert model.block.linear.kernel.value.shape == (3072, 256)
    
    model.train()
    
    batch_size = 128
    num_epochs = 500

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        rng = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(rng, len(x_train))
        
        for i in range(0, len(x_train), batch_size):
            batch_indices = perm[i:i+batch_size]
            batch_inputs = x_train[batch_indices]
            batch_labels = y_train[batch_indices]
            
            batch_loss, batch_accuracy = train_step(model, batch_inputs, batch_labels)
            
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {num_batches} | "
                      f"Loss: {batch_loss:.4f} | Accuracy: {batch_accuracy:.2%}")
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        print(f"\nEpoch {epoch+1} 完成 | 平均损失: {avg_loss:.4f} | 平均准确率: {avg_accuracy:.2%}\n")
    
    model.eval()
    test_logits = model(x_test)
    test_preds = jnp.argmax(test_logits, axis=-1)

if __name__ == "__main__":
    main()