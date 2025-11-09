import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from tensorflow.keras.datasets import cifar10

# 数据加载与预处理
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = jnp.float32(x_train)/255.0
    x_test = jnp.float32(x_test)/255.0
    y_train = jnp.int32(y_train).flatten()
    y_test = jnp.int32(y_test).flatten()
    return (x_train, y_train), (x_test, y_test)

# 简化CNN模型
class SimpleCNN(nn.Module):
    @nn.compact
  def __call__(self, x, training=True):  # 新增training参数用于Dropout
        # 第1组卷积
        x = nn.Conv(64, (3,3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3,3), padding='SAME')(x)  # 增加1层卷积
        x = nn.relu(x)
        x = nn.max_pool(x, (2,2), (2,2))  # (16,16,64)
        
        # 第2组卷积
        x = nn.Conv(128, (3,3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(128, (3,3), padding='SAME')(x)  # 增加1层卷积
        x = nn.relu(x)
        x = nn.max_pool(x, (2,2), (2,2))  # (8,8,128)
        
        # 全局平均池化（手动实现）
        x = jnp.mean(x, axis=(1, 2))  # (batch_size, 128)
        
        # 增加全连接层+Dropout防过拟合
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5)(x, deterministic=not training)  # 训练时启用Dropout
        
        x = nn.Dense(10)(x)
        return x

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = SimpleCNN()
    
    # 初始化训练状态
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, jnp.ones((1,32,32,3)))['params']
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(0.001)
    )

    # 训练步函数
    @jax.jit
    def train_step(state, batch):
        x, y = batch
        def loss_fn(p):
            logits = model.apply({'params': p}, x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    # 评估函数
    @jax.jit
    def evaluate(state, x, y):
        preds = jnp.argmax(model.apply({'params': state.params}, x), axis=-1)
        return jnp.mean(preds == y)

    # 训练循环
    batch_size, num_epochs = 128, 3
    for epoch in range(num_epochs):
        rng_epoch = jax.random.fold_in(rng, epoch)
        shuffled_idx = jax.random.permutation(rng_epoch, len(x_train))
        
        total_loss = 0.0
        for i in range(0, len(x_train), batch_size):
            batch = (x_train[shuffled_idx[i:i+batch_size]], y_train[shuffled_idx[i:i+batch_size]])
            state, loss = train_step(state, batch)
            total_loss += loss
        
        train_loss = total_loss / (len(x_train)//batch_size)
        test_acc = evaluate(state, x_test, y_test)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2%}")

if __name__ == "__main__":
    main()