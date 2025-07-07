import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Module):
    def __init__(
        self, in_features, out_features, merge, rank=16, alpha=16, dropout=0.5
    ):
        """
        in_features：输入维度，比如词嵌入维度是768，这里就是768；
        out_features：输出维度;
        merge：控制是否合并 LoRA 和原始权重;
        rank：LoRA 的秩，越小压缩程度越大，控制 LoRA 矩阵参数量的主要因素;
        alpha：缩放系数，用于调节 LoRA 矩阵参数的影响程度;
        dropout：防止过拟合的 dropout 率;
        """
        super(LoraLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features)

        if rank > 0:
            # 构建权重参数
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
            self.scale = self.alpha / rank
            self.linear.weight.requires_grad = False

        if dropout > 0:
            self.dropout = nn.Dropout(self.dropout)
        else:
            self.dropout = nn.Identity()

        self.initial_weights()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        if self.rank > 0 and self.merge:
            output = F.linear(
                x,
                weight=self.linear.weight + self.lora_b @ self.lora_a,
                bias=self.linear.bias,
            )
            output = self.dropout(output)
            return output

        else:
            return self.linear(x)


if __name__ == "__main__":
    x = torch.randn(3, 2, 10)
    rola_linear = LoraLinear(10, 20, True)
    output = rola_linear(x)
    print("output", output)
    print("shape of output", output.shape)
