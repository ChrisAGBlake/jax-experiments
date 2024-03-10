import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

class ResNetBlock(nn.Module):
    features: int

    def __call__(self, x):
        pass

class MLPBlock(nn.Module):
    features: tuple

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

class ResNetBlock(nn.Module):
    features: tuple

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            if i == 0:
                y = nn.Dense(feat)(x)
            else:
                y = nn.Dense(feat)(y)
            y = nn.relu(x)
        return y + x

class ResNet(nn.Module):
    resnet_blocks: tuple
    mlp_features: tuple

    @nn.compact
    def __call__(self, x):
        for block in self.resnet_blocks:
            x = ResNetBlock(features=block)(x)
        x = MLPBlock(features=self.mlp_features)(x)
        return x

def train():

    # set hyperparameters
    key = jax.random.PRNGKey(0)
    lr = 1e-4
    batch_size = 128
    n_epochs = 10
    n_in = 5
    n_out = 1

    # initialise model and optimiser
    resnet_blocks = ((32, 32),(32, 32))
    mlp_features = (32, 32, n_out)
    model = ResNet(resnet_blocks=resnet_blocks, mlp_features=mlp_features)
    optimiser = optax.adam(lr)
    params = model.init(key, jnp.empty((1, n_in)))
    opt_state = optimiser.init(params)
    
    # generate random data
    data_sz = 100000
    x = jax.random.uniform(key, (data_sz, n_in), minval=0.0, maxval=1.0)
    y = jnp.expand_dims(jnp.sum(x, axis=1), axis=1)

    # define the loss function
    def loss_fn(params, inputs, labels):
        preds = model.apply(params, inputs)
        return jnp.mean(jnp.square(labels - preds))
    
    # define the update step
    @jax.jit
    def step(params, opt_state, inputs, labels):
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, labels)
        updates, opt_state = optimiser.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # train for n epochs
    for i in range(n_epochs):
        j = 0
        l = 0
        while j < data_sz:
            e = min(data_sz, j+batch_size)
            params, opt_state, loss = step(params, opt_state, x[j:e, :], y[j:e, :])
            l += loss
            j = e
        l /= int(data_sz / batch_size)
        print(f'epoch {i}, loss: {l}')

if __name__ == '__main__':
    train()