import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

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
    dropout_rate: float = 0.1
    num_heads: int = 8

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
        return x
    
def inference():

    # setup model
    num_layers = 2
    features = 32
    vocab_size = 100
    model = DecoderOnlyTransformer(num_layers=num_layers, features=features, vocab_size=vocab_size)

    # generate random data
    key = jax.random.PRNGKey(0)
    tokens = jax.random.randint(key, (1, 20), 0, vocab_size)
    print('input:', tokens)

    # initialise model parameters
    params = model.init(key, tokens)

    # run model
    output = model.apply(params, tokens)
    print('output:', output.shape)

    # get predicted tokens
    predicted_tokens = jnp.argmax(output, axis=-1)
    print('predicted tokens:', predicted_tokens)

def train():

    # set hyperparameters
    key = jax.random.PRNGKey(0)
    lr = 1e-4
    batch_size = 1
    n_epochs = 10
    context_length = 128

    # setup model
    num_layers = 2
    features = 32
    vocab_size = 100
    model = DecoderOnlyTransformer(num_layers=num_layers, features=features, vocab_size=vocab_size)

    # load tiny shakespere dataset
    with open('data/tiny_shakespeare.txt', 'r') as f:
        text = f.read()
        text = text.lower()

    # tokenise the text, 1 token per character to keep it simple
    all_tokens = sorted(list(set(text)))
    token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    tokens = jnp.array([token_to_idx[token] for token in text], dtype=jnp.int32)

    # generate training data
    x = []
    y = []
    for i in range(0, len(tokens) - context_length - 1):
        x.append(tokens[i:i+context_length])
        y.append(tokens[i+1:i+context_length+1])
    x = jnp.array(x)
    y = jnp.array(y)

    # define the loss function

    # define the update step

    # train for n epochs
    
if __name__ == '__main__':
    inference()