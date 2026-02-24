import torch

from einops import rearrange
from torch import nn



class LoraLayer(nn.Module):
  def __init__(self, cross_att_matrix, rank = 8, alpha = 8):
    super().__init__()
    self.cross_att_matrix = cross_att_matrix
    # self.cross_att_matrix.weight.requires_grad = False
    # self.cross_att_matrix.bias.requires_grad = False
    self.rank = rank
    self.alpha = alpha
    self.A = nn.Linear(self.cross_att_matrix.in_features, rank, bias=False)
    self.B = nn.Linear(rank, self.cross_att_matrix.out_features, bias=False)
    nn.init.kaiming_uniform_(self.A.weight)
    nn.init.zeros_(self.B.weight)
  
  def forward(self, x):

    ans = self.cross_att_matrix(x) + self.B(self.A(x)) * (self.alpha / self.rank)

    return ans


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### YOUR CODE HERE
    # query: b h t d + key: b h d t
    # attention b h t t 
    attention = torch.matmul(query, key.transpose(-2, -1)) / (self.attention_head_size ** 0.5)
    seq_len = attention.size(-1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention.device))
    after_mask = attention + attention_mask 
    after_mask = after_mask.masked_fill(causal_mask == 0, float('-inf'))

    att_weight = torch.softmax(after_mask, dim = -1)
    # att_weight b h t t + value b h t d
    attn_value = torch.matmul(att_weight, value)
    # final_weight b h t d
    attn_value = rearrange(attn_value, 'b h t d -> b t (h d)')
    return attn_value


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
