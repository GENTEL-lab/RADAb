import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

from diffab.utils.protein.constants import MSA_PAD, MASK, MSA_ALPHABET
from esm.modules import TransformerLayer, LearnedPositionalEmbedding, RobertaLMHead, ESM1bLayerNorm, AxialTransformerLayer
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        Used for encoding timestep in diffusion models

        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) * -(np.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        device = x.device
        pe = pe.to(device)
        return pe[x] # .to(x.device)

class MSATransformerTime(nn.Module):
    """
    Based on implementation described by Rao et al. in "MSA Transformer"  ESM1-b
    https://doi.org/10.1101/2021.02.12.430858
    Args:
        d_model: int,
            embedding dimension of model
        d_hidden: int,
            embedding dimension of feed forward network
       n_layers: int,
           number of layers
       n_heads: int,
           number of attention heads
   """

    def __init__(self, d_model, d_hidden, n_layers, n_heads, n_tokens=22,
                 padding_idx=21, mask_idx=20,
                 max_positions=1024, timesteps=None):
        super(MSATransformerTime, self).__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_model, timesteps) 
        self.embed_tokens = nn.Embedding(
            n_tokens, d_model, padding_idx=mask_idx
        )
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer( 
                    d_model, d_hidden, n_heads
                )
                for _ in range(n_layers)
            ]
        )
        self.padding_idx = padding_idx

        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, padding_idx)
        self.emb_layer_norm_before = nn.LayerNorm(d_model)
        self.emb_layer_norm_after = nn.LayerNorm(d_model)
        self.lm_head = RobertaLMHead(
            embed_dim=d_model,
            output_dim=n_tokens,
            weight=self.embed_tokens.weight
        )


    
    def forward(self, tokens, timesteps):
        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C
        x = self.embed_tokens(tokens)
        x = x + self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(x.size())
        x = self.emb_layer_norm_before(x)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        #print("x", x.shape) # B, D, L, E
        y = self.time_encoding(timesteps)
        y = y.unsqueeze(1).unsqueeze(1)
        y = y.expand(y.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x += y

        q = torch.zeros(x.shape)
        q = q.to(x.device)
        q[:,0,:,0] += 1 
        x += q
        #

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)
        
        for layer_idx, layer in enumerate(self.layers):
            x = checkpoint(layer, x, None, padding_mask, False)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        x = self.lm_head(x)
        return x
    

    
class MSATransformer(nn.Module):
    """
    Based on implementation described by Rao et al. in "MSA Transformer"
    https://doi.org/10.1101/2021.02.12.430858

    Args:
        d_model: int,
            embedding dimension of model
        d_hidden: int,
            embedding dimension of feed forward network
       n_layers: int,
           number of layers
       n_heads: int,
           number of attention heads
   """

    def __init__(self, d_model, d_hidden, n_layers, n_heads, use_ckpt=False, n_tokens=29,
                 padding_idx=21, mask_idx=20,
                 max_positions=1024):
        super(MSATransformer, self).__init__()
        self.embed_tokens = nn.Embedding(
            n_tokens, d_model, padding_idx=mask_idx
        )
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    d_model, d_hidden, n_heads
                )
                for _ in range(n_layers)
            ]
        )
        self.padding_idx = padding_idx

        # self.contact_head = ContactPredictionHead()
        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, padding_idx)
        self.emb_layer_norm_before = nn.LayerNorm(d_model)
        self.emb_layer_norm_after = nn.LayerNorm(d_model)
        self.lm_head = RobertaLMHead(
            embed_dim=d_model,
            output_dim=n_tokens,
            weight=self.embed_tokens.weight
        )

        self.use_ckpt = use_ckpt

    def forward(self, tokens):
        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C

        x = self.embed_tokens(tokens)
        x = x + self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(x.size())

        x = self.emb_layer_norm_before(x)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = checkpoint(layer, x, None, padding_mask, False)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        x = self.lm_head(x)
        return x
    
