import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x, mask=None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        batch_size, seq_len, d_model = q.size()
        n_d = self.d_model // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, n_d).permute(0, 2, 1, 3) # 等价于q.transpose(1, 2) ，torch.einsum('abcd->acbd', q) 
        k = k.view(batch_size, seq_len, self.num_heads, n_d).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, n_d).permute(0, 2, 1, 3)
        score = q @ k.transpose(2, 3) / math.sqrt(n_d) 

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e10)
        
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)
        out = self.w_o(score)
        return out
    
if __name__ == '__main__':
    x = torch.randn((3,10,200))
    mhattention = MultiHeadAttention(200, 5)
    y = mhattention(x)
    print(y.shape)

