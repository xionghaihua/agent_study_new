"""
什么是自注意力
我们观察事物时，之所以能够快速判断一种事物，是因为我们大脑能够很快把注意力放到事物具有辨识度的部分从而做出判断
而并非是从头到尾的观察一遍事物后，才能有判断结果。

什么是注意力计算规则：

需要三个指定的输入Q，K，V，然后通过公式得到主力已的计算结果。，这个结果代表query在key和value作用下的表示

什么是注意力机制
注意力机制是注意力计算规则能够作用的深度学习网络的载体，除了注意力计算规则外，还包括一些必要的全连接层以及张量处理。



"""
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        """
        :param d_model: 代表词嵌入维度
        :param dropout: 代表Dropout层的置零比率
        :param max_len: 代表每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        #实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)
        #初始化一个位置编码矩阵，矩阵大小max_len * d_model
        pe = torch.zeros(max_len,d_model)
        #初始化一个绝对位置矩阵
        position = torch.arange(0,max_len).unsqueeze(1)
        #定义一个变换矩阵div_term，跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        #将前面定义的变化矩阵进行奇数，偶数分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)  #正玄波，偶数
        pe[:, 1::2] = torch.cos(position * div_term)  #余玄波，奇数
        #将二维张量扩充为三维张量
        pe = pe.unsqueeze(0)

        #将位置编码矩阵注册成模型的buffer，这个buffer不是模型中的参数，不跟随优化器同步更新
        #注册成buffer后，我们就可以在模型保存后重新加载后，将这个位置编码器和模型参数加载进来
        self.register_buffer('pe',pe)

    def forward(self,x):
        #x：代表文本序列的词嵌入表示
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        #最后使用self.dropout对象进行丢弃操作，并返回结果
        return self.dropout(x)
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        """类的初始化函数，有2个参数
        d_model：词嵌入的维度
        vocab： 此表的大小
        """
        super(Embeddings,self).__init__()
        #定义Embedding层
        self.lut = nn.Embedding(vocab,d_model)
        #将参数传入类中
        self.d_model = d_model
    def forward(self,x):
        """
         可以将其理解为该层的前向传播逻辑，所有层中都会由此函数，当传给该类的实例化对象参数时
         自动调用该类函数
        :param x: 输入给模型的文本，通过词汇映射后的数字张量
        :return:
        """
        #将x传给self.lut并与根号下self.d_model相乘
        return self.lut(x) * math.sqrt(self.d_model)

def attention(query,key,value,mask=None,dropout=None):
    """注意力机制的实现，输入分别为query,key,value,mask掩码张量
    dropout: nn.Dropout层的实例化对象
    """

    #在函数中，首先取query的最后一堆的大小，代表的是词嵌入的维度
    d_k = query.size(-1)
    #按照注意力公式，将query与key的转置相乘，这里面的key是将最后两个维度进行转置，在除以缩放系数
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    #判断是否使用掩码张量
    if mask is not None:
        #使用masked_fill方法，将掩码张量和0进行位置的一一比较，如果等于0，替换成一个非常小的数值
        scores = scores.masked_fill(mask == 0, -1e9)
    #对scores的最后一个维度进行softmax操作
    p_attn = F.softmax(scores, dim=-1)
    #判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    #最后一步完成p_atten和value张量的乘法，并返回query注意力表示
    return torch.matmul(p_attn, value), p_attn


d_model = 512
dropout = 0.1
max_len = 60
vocab = 1000
x=Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
emb = Embeddings(d_model,vocab)
embr = emb(x)
#位置编码
pe = PositionalEncoding(d_model,dropout,max_len)
pe_result = pe(embr)

#自注意力
query=key=value=pe_result
attn,p_attn = attention(query,key,value)
print('attn:',attn)
print(attn.shape)
print('p_attn:',p_attn)
print(p_attn.shape)

mask = Variable(torch.zeros(2,4,4))
attn1,p_attn1 = attention(query,key,value,mask)
print('attn1:',attn1)
print(attn1.shape)
print('p_attn1:',p_attn1)

"""
attn: tensor([[[-26.7933,  37.2366,   0.0000,  ..., -40.1239,   4.5494, -21.9429],
         [-20.5188,  19.3238, -24.0036,  ...,  12.6795,  26.8266,  15.0491],
         [-10.5331,  -1.5147, -31.2705,  ...,  41.6160,  40.7642,  40.8303],
         [-18.2339, -53.8416,  31.4027,  ..., -26.2250,  22.7011,   7.5014]],

        [[ 28.7383, -33.3948,  10.1932,  ...,  21.6507,  19.5721, -10.9990],
         [ 13.5468,   0.0000, -14.5661,  ...,  52.5496, -10.7250,  17.5659],
         [ 20.9401, -10.0718, -14.5619,  ..., -19.1807,  24.1714,  -3.1682],
         [-10.6525, -30.7589,  -3.9749,  ...,  11.8437,   0.0000, -33.8177]]],
       grad_fn=<UnsafeViewBackward0>)
torch.Size([2, 4, 512])
p_attn: tensor([[[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]],

        [[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]]], grad_fn=<SoftmaxBackward0>)
torch.Size([2, 4, 4])
attn1: tensor([[[-19.0198,   0.3010,  -5.9678,  ...,  -3.0134,  23.7103,  10.3595],
         [-19.0198,   0.3010,  -5.9678,  ...,  -3.0134,  23.7103,  10.3595],
         [-19.0198,   0.3010,  -5.9678,  ...,  -3.0134,  23.7103,  10.3595],
         [-19.0198,   0.3010,  -5.9678,  ...,  -3.0134,  23.7103,  10.3595]],

        [[ 13.1432, -18.5564,  -5.7274,  ...,  16.7158,   8.2547,  -7.6048],
         [ 13.1432, -18.5564,  -5.7274,  ...,  16.7158,   8.2547,  -7.6048],
         [ 13.1432, -18.5564,  -5.7274,  ...,  16.7158,   8.2547,  -7.6048],
         [ 13.1432, -18.5564,  -5.7274,  ...,  16.7158,   8.2547,  -7.6048]]],
       grad_fn=<UnsafeViewBackward0>)
torch.Size([2, 4, 512])
p_attn1: tensor([[[0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500]],

        [[0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500]]], grad_fn=<SoftmaxBackward0>)

"""