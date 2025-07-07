"""
Embedding层将Tokenizer得到的词索引转化为嵌入向量,嵌入转化的原理实际就是一个查找表.
[batch_size, seq_len] -> [batch_size, seq_len, dim]
此代码实现了添加位置编码的embedding模块，
"""

import torch
import torch.nn as nn


# 将输入的词汇表索引转换为指定维度的Embedding
# 每个token的词索引升维到d_model维，padding_idx=1表示填充词的索引为1
# 继承nn.Embedding在训练中前向传播，反向传播，更新参数
class TokenEmbedding(nn.Embedding):
    # Token_Embedding 词嵌入处理
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionEmbedding(nn.Module):
    # Position_Embedding 位置编码
    def __init__(self, d_model, max_len, device="cuda"):
        super(PositionEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        # vocab_size 词表大小, d_model Embedding转换的维度, max_len 位置编码的最长长度, drop_prob dropout层的比例
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model).to(device)
        self.positional_embedding = PositionEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding(x)
        return self.dropout(token_embedding + positional_embedding)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.Tensor([[1, 2, 3, 4], [2, 3, 4, 5]]).long()
    vocab_size = 100
    d_model = 256
    max_len = 1000
    drop_prob = 0.1
    print(device)
    print("x", x)
    model = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)
    y = model(x)
    print("y", y)
    print("shape of y", y.shape)

    embedding = TokenEmbedding(vocab_size, d_model)
    print(embedding(x))
    print(embedding(x).shape)
