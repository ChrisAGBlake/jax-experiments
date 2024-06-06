import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import pickle
import random

# load tiny shakespere dataset
with open('data/tiny_shakespere.txt', 'r') as f:
    text = f.read()
    text = text.lower()

# tokenise the text, 1 token per character to keep it simple
unique_tokens = sorted(list(set(text)))
token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}
tokens = jnp.array([token_to_idx[token] for token in text], dtype=jnp.int32)

# set hyperparameters
key = jax.random.PRNGKey(0)
lr = 1e-3
batch_size = 1
n_epochs = 1
context_length = 32
stride = 10
num_layers = 2
features = 32
vocab_size = len(unique_tokens)
num_heads = 8

class TransformerBlock(nn.Module):
    features: int
    dropout_rate: float = 0.1
    num_heads: int = 8

    @nn.compact
    def __call__(self, x, train, mask=None):
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.features)(x, mask=mask)
        x = nn.LayerNorm()(x + y)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        y = nn.Dense(features=self.features)(x)
        y = nn.gelu(y)  
        x = nn.LayerNorm()(x + y)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)  
        return x
    
class DecoderOnlyTransformer(nn.Module):
    num_layers: int
    features: int
    vocab_size: int
    num_heads: int
    dropout_rate: float = 0.1
    

    def setup(self):
        self.token_embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.features)
        self.transformer_blocks = [TransformerBlock(features=self.features, dropout_rate=self.dropout_rate, num_heads=self.num_heads) for _ in range(self.num_layers)]
        self.mlp = nn.Dense(features=self.vocab_size)

    def __call__(self, inputs, train=False):
        x = self.token_embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, train)
        x = self.mlp(x)
        x = nn.softmax(x)
        return x[:, -1, :]
    
def inference():

    # setup model
    model = DecoderOnlyTransformer(num_layers=num_layers, features=features, vocab_size=vocab_size, num_heads=num_heads)

    # load the model params
    with open('models/model.pkl', 'rb') as f:
        params = pickle.load(f)

    # generate the prompt
    i = random.randint(0, len(text) - context_length)
    prompt = text[i:i+context_length]
    prompt_tokens = jnp.array([[token_to_idx[token] for token in prompt]], dtype=jnp.int32)

    # generate text response
    gen_text = ''
    for _ in range(100):
        output = model.apply(params, prompt_tokens)
        predicted_token = jnp.argmax(output[0, -1]).astype(jnp.int32) 
        gen_text += idx_to_token[predicted_token.item()]
        prompt_tokens = jnp.append(prompt_tokens, predicted_token)
        prompt_tokens = prompt_tokens[1:]
    print('prompt:', prompt)
    print('response:', gen_text)

def train():

    # setup model
    model = DecoderOnlyTransformer(num_layers=num_layers, features=features, vocab_size=vocab_size, num_heads=num_heads)

    # generate training data
    x = []
    y = []
    for i in range(0, len(tokens) - context_length - 1, stride):
        x.append(tokens[i:i+context_length])
        y.append(tokens[i+context_length:i+context_length+1])
    x = jnp.array(x)
    y = jnp.array(y)
    data_sz = x.shape[0]
    print('data size:', data_sz)

    # setup the optimiser
    optimiser = optax.adam(lr)
    params = model.init(key, jnp.empty((1, context_length), dtype=jnp.int32))
    params = jax.device_put(params)
    opt_state = optimiser.init(params)

    # define the loss function - cross entropy
    def loss_fn(params, inputs, labels):
        # get predicted logits
        logits = model.apply(params, inputs)

        # convert labels to one-hot
        labels_one_hot = jax.nn.one_hot(labels, vocab_size)

        # calculate cross entropy loss
        ce_loss = -jnp.mean(jnp.sum(labels_one_hot * jax.nn.log_softmax(logits), axis=-1))
        return ce_loss

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
            inputs = jax.device_put(x[j:e, :])
            labels = jax.device_put(y[j:e, :])
            params, opt_state, loss = step(params, opt_state, inputs, labels)
            l += loss
            j = e
        l /= int(data_sz / batch_size)
        print(f'epoch {i}, loss: {l}')

    # save the model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(params, f)


if __name__ == '__main__':
    train()
    inference()
    