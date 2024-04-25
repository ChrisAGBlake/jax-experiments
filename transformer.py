import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

class TransformerBlock(nn.Module):
    features: int
    dropout_rate: float = 0.1
    num_heads: int = 8

    @nn.compact
    def __call__(self, x, mask=None):
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.features)(x, mask=mask)
        x = nn.LayerNorm()(x + y)
        x = nn.Dropout(rate=self.dropout_rate)(x)
        y = nn.Dense(features=self.features)(x)
        y = nn.gelu(y)  
        x = nn.LayerNorm()(x + y)
        x = nn.Dropout(rate=self.dropout_rate)(x)  
        return x
    
class DecoderOnlyTransformer(nn.Module):
    num_layers: int
    features: int
    dropout_rate: float = 0.1
    num_heads: int = 8
    vocab_size: int = 100

    def setup(self):
        self.token_embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.features)
        self.transformer_blocks = [TransformerBlock(features=self.features, dropout_rate=self.dropout_rate, num_heads=self.num_heads) for _ in range(self.num_layers)]

    def __call__(self, inputs):
        x = self.token_embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = nn.Dense(features=self.vocab_size)(x)
        x = nn.softmax(x)
        return x