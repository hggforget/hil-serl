from flax import linen as nn
from jax import numpy as jnp
from typing import Optional
import numpy as np
import jax
from jax import lax
from jax import random
from jax import jit
from flax import struct

@struct.dataclass
class Config:
    # Number of encoder-decoder layer pairs
    layers: int = 6
        
    # Number of units in dense layer
    mlp_dim: int = 2048
        
    # Tokens length
    length: int = 18
        
    # Number of embdedding dim
    features: int = 512
        
    # Batch size
    batch: int = 16
        
    # Number of heads in multihead attention
    num_heads: int = 8
        
    # Number of dims per head = features/num_heads
    head_dim: int = 64
        
    # Bias
    use_bias: bool = False
        
    # Droput rate
    dropout_rate: float = 0.2
        
    # Dropout or not
    training: bool = False
        
    # Random seed
    seed: int = 0

def reparametrize(mu, logvar):
    std = jnp.exp(logvar / 2)
    eps = random.normal(random.PRNGKey(0), shape=std.shape)
    return mu + std * eps

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / jnp.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = jnp.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table = sinusoid_table.at[:, 0::2].set(jnp.sin(sinusoid_table[:, 0::2]))  # dim 2i
    sinusoid_table = sinusoid_table.at[:, 1::2].set(jnp.cos(sinusoid_table[:, 1::2]))  # dim 2i+1

    return jnp.expand_dims(sinusoid_table, axis=0)

class MultiheadAttention(nn.Module):
    
    config: Config
    decode: bool = False
    dtype: jnp.dtype = jnp.float32
        
    @nn.compact
    def __call__(self, kv, q, mask):
        # Initialize config
        cfg = self.config
        # Initialize key, query, value through Dense layer [batch=1, length=10, features=10]
        query = nn.Dense(features=cfg.features, use_bias=cfg.use_bias, name='query')(q)
        key = nn.Dense(features=cfg.features, use_bias=cfg.use_bias, name='key')(kv)
        value = nn.Dense(features=cfg.features, use_bias=cfg.use_bias, name='value')(kv)
        
        # Layer norm
        query = nn.LayerNorm()(query)
        key = nn.LayerNorm()(key)
        value = nn.LayerNorm()(value)

        # Split head [batch=1, length=10, features=10] -> [batch=1, length=10, num_heads=2, depth_per_head=5]
        query = query.reshape(cfg.batch, cfg.length, cfg.num_heads, cfg.head_dim)
        key = key.reshape(cfg.batch, cfg.length, cfg.num_heads, cfg.head_dim)
        value = value.reshape(cfg.batch, cfg.length, cfg.num_heads, cfg.head_dim)
        
        # Scaled dot-product attention [batch=1, length=10, num_heads=2, depth_per_head=5]
        logits = self.scaled_dot_product_attention(key, query, value, mask)
        
        # Concat [batch=1, length=10, num_heads=2, depth_per_head=5] -> [batch=1, length=10, features=10]
        logits = logits.reshape(cfg.batch, cfg.length, cfg.features)
        
        # Linear
        logits = nn.Dense(features=cfg.features, 
                          use_bias=cfg.use_bias, 
                          name='attention_weights'
                         )(logits)
        logits = nn.Dropout(rate=cfg.dropout_rate)(logits, deterministic=not cfg.training)
        
        return logits
    

    def scaled_dot_product_attention(self, key, query, value, mask):
        """ Matmul
            query: [batch, q_length, num_heads, qk_depth_per_head]
            key: [batch, kv_length, num_heads, qk_depth_per_head]
            -> qk: [batch, num_heads, q_length, kv_length]
        """
        attention_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)
        # Scale
        d_k = query.shape[-1]
        attention_weights = attention_weights/jnp.sqrt(d_k)
        # Mask
        attention_weights = attention_weights + mask
        # Softmax
        attention_weights = nn.softmax(attention_weights).astype(self.dtype)
        """ Matmul
            qk: [batch, num_heads, q_length, kv_length]
            value: [batch, kv_length, num_heads, v_depth_per_head]
            -> Return: [batch, length, num_heads, v_depth_per_head]
        """
        
        return jnp.einsum('bhqk,bkhd->bqhd', attention_weights, value)
    
    
    
    
    # Attention mask 
def attention_mask(input_tokens):
    """Mask-making helper for attention weights (mask for padding)
    Args:
        input_tokens: [batch_size, tokens_length]
    return:
        mask: [batch_size, num_heads=1, query_length, key_value_length]
    """
    mask = jnp.multiply(jnp.expand_dims(input_tokens, axis=-1), jnp.expand_dims(input_tokens, axis=-2))
    mask = jnp.expand_dims(mask, axis=-3)
    mask = lax.select(
        mask > 0,
        jnp.full(mask.shape, 0.).astype(jnp.float32),
        jnp.full(mask.shape, -jnp.inf).astype(jnp.float32)
    )
    return mask


# Causal mask for Decoder
def causal_mask(input_tokens):
    mask = attention_mask(input_tokens)
    mask = jnp.full(mask.shape, -jnp.inf)
    mask = jnp.triu(mask, k=1)
    return mask


class MLP(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        # Initialize config
        cfg = self.config
        # mlp_dim*4 to increase knowledge by 4 times
        x = nn.Dense(features=cfg.mlp_dim*4, use_bias=cfg.use_bias)(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not cfg.training)
        x = nn.relu(x)
        x = nn.Dense(features=cfg.features, use_bias=cfg.use_bias)(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not cfg.training)
        return x
    
    
class Encoder(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, input_embeding, encoder_mask):
        cfg = self.config
        # Encoder layer
        x = input_embeding
        for i in range(cfg.layers):
            # Multihead Attention
            y = MultiheadAttention(config=cfg)(
                kv=x,
                q=x,
                mask=encoder_mask
            )
            # Add & Norm
            y = y + x
            x = nn.LayerNorm()(y)
            # MLP
            y = MLP(cfg)(x)
            # Add & Norm
            y = y + x
            x = nn.LayerNorm()(y)
        
        return x
    
    
class Decoder(nn.Module):
    config: Config
        
    @nn.compact
    def __call__(self, input_embeding, encoder_kv, decoder_mask, encoder_decoder_mask):
        cfg = self.config
        # Decoder layer
        x = input_embeding
        for i in range(cfg.layers):
            # Multihead Attention
            y = MultiheadAttention(config=cfg)(
                kv=x,
                q=x,
                mask=decoder_mask,
            )
            # Add & Norm
            y = y + x
            x = nn.LayerNorm()(y)
            # Cross Multihead-Attention
            y = MultiheadAttention(config=cfg)(
                kv=encoder_kv,
                q=x,
                mask=encoder_decoder_mask,
            ) 
            # Add & Norm
            y = y + x
            x = nn.LayerNorm()(y)
            # MLP
            y = MLP(cfg)(x)
            # Add & Norm
            y = y + x
            x = nn.LayerNorm()(y)
        
        return x
    

class Transformer(nn.Module):
    config: Config
        
    @nn.compact
    def __call__(self, encoder_input_tokens, decoder_target_tokens):
        cfg = self.config
        # Initialize mask
        padding_mask = attention_mask(encoder_input_tokens)
        decoder_mask = causal_mask(decoder_target_tokens)
        # Embeding: [batch, length] -> [batch, length, feature]
        input_embeding = nn.Embed(num_embeddings=cfg.length, features=cfg.features)(
            encoder_input_tokens.astype('int32')
        )
        
        # Encoder layer
        x = Encoder(cfg)(input_embeding=input_embeding, encoder_mask=padding_mask)
        
        # Decoder layer
        x = Decoder(cfg)(
            input_embeding=input_embeding,
            encoder_kv=x,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=padding_mask
        )
        
        # Linear
        x = nn.Dense(features=cfg.length, use_bias=cfg.use_bias)(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not cfg.training)
        
        return x
    
    
    
if __name__ == '__main__':
    config = Config()
    random_seed = jax.random.PRNGKey(config.seed)

    # TODO: Position embeding
    #................................................

    # Tokenizer
    encoder_input_tokens = random.normal(random_seed, (config.batch, config.length))
    decoder_target_tokens = random.normal(random_seed, (config.batch, config.length))

    # Init model
    model = Transformer(config)
    params = model.init(random_seed, encoder_input_tokens, decoder_target_tokens)

    # Generate output
    x = model.apply(params, encoder_input_tokens, decoder_target_tokens)
    print('Shape of logits:', x.shape) 
    
    
class PositionEmbeddingSine(nn.Module):
    num_pos_feats: int = 64
    temperature: float = 10000.0
    normalize: bool = False
    scale: float = None
    

    def setup(self):
        if self.scale is None:
            self.scale = 2 * jnp.pi

    def __call__(self, tensor):
        x = tensor
        not_mask = jnp.ones_like(x[:, 0, 0])
        y_embed = jnp.cumsum(not_mask, axis=1, dtype=jnp.float32)
        x_embed = jnp.cumsum(not_mask, axis=2, dtype=jnp.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = jnp.arange(self.num_pos_feats, dtype=jnp.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = jnp.stack((jnp.sin(pos_x[:, :, :, 0::2]), jnp.cos(pos_x[:, :, :, 1::2])), axis=4).reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        pos_y = jnp.stack((jnp.sin(pos_y[:, :, :, 0::2]), jnp.cos(pos_y[:, :, :, 1::2])), axis=4).reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        pos = jnp.concatenate((pos_y, pos_x), axis=3).transpose((0, 3, 1, 2))
        return pos

class DETRVAE(nn.Module):
    num_queries: int
    camera_names: list
    state_dim: int
    latent_dim: int = 32
    encoder: Optional[nn.Module]
    transformer: Optional[nn.Module]

    def setup(self):
        self.action_head = nn.Dense(self.state_dim)
        self.is_pad_head = nn.Dense(1)
        self.query_embed = nn.Embed(self.num_queries, self.transformer.d_model)
        if self.backbones is not None:
            self.input_proj = nn.Conv(self.backbones[0].num_channels, self.transformer.d_model, kernel_size=(1, 1))
            self.input_proj_robot_state = nn.Dense(self.transformer.d_model)
        else:
            self.input_proj_robot_state = nn.Dense(self.transformer.d_model)
            self.input_proj_env_state = nn.Dense(self.transformer.d_model)
            self.pos = nn.Embed(2, self.transformer.d_model)
            self.backbones = None

        # encoder extra parameters
        self.cls_embed = nn.Embed(1, self.transformer.d_model)
        self.encoder_action_proj = nn.Dense(self.transformer.d_model)
        self.encoder_joint_proj = nn.Dense(self.transformer.d_model)
        self.latent_proj = nn.Dense(self.latent_dim * 2)
        self.pos_table = get_sinusoid_encoding_table(1 + 1 + self.num_queries, self.transformer.d_model)

        # decoder extra parameters
        self.latent_out_proj = nn.Dense(self.transformer.d_model)
        self.additional_pos_embed = nn.Embed(2, self.transformer.d_model)

    def __call__(self, qpos, image, env_state, actions=None, is_pad=None):
        is_training = actions is not None
        bs, _ = qpos.shape

        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos)
            qpos_embed = jnp.expand_dims(qpos_embed, axis=1)
            cls_embed = self.cls_embed.weight
            cls_embed = jnp.expand_dims(cls_embed, axis=0).tile((bs, 1, 1))
            encoder_input = jnp.concatenate([cls_embed, qpos_embed, action_embed], axis=1)
            encoder_input = jnp.transpose(encoder_input, (1, 0, 2))
            cls_joint_is_pad = jnp.full((bs, 2), False)
            is_pad = jnp.concatenate([cls_joint_is_pad, is_pad], axis=1)
            pos_embed = self.pos_table.clone().detach()
            pos_embed = jnp.transpose(pos_embed, (1, 0, 2))
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = jnp.zeros((bs, self.latent_dim), dtype=jnp.float32)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            proprio_input = self.input_proj_robot_state(qpos)
            src = jnp.concatenate(all_cam_features, axis=3)
            pos = jnp.concatenate(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = jnp.concatenate([qpos, env_state], axis=1)
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]
