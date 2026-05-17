"""
位置编码的作用：因为在transformer的编码结构中，并没有针对词汇位置信息的处理，因此需要再
Embedding层后加入位置编码器，将词汇位置不同的可能会产生不同语义的信息加入到词嵌入张量中，
"""

#位置编码器的代码分析
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
#构建位置编码器的类
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
"""
m = nn.Dropout(p=0.2)
input = torch.randn(4,5)
output = m(input)
print(output)
"""
"""
tensor([[-0.0000,  1.2073, -0.0000,  1.4977, -0.8622],
        [ 0.8457,  0.0647, -2.3360,  0.0000, -1.3821],
        [ 0.1791,  1.9195, -0.1534, -0.6049, -0.4185],
        [ 1.2247,  0.2208,  0.7979,  1.2558, -0.2980]])
"""
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
print(pe_result)
print(pe_result.shape)
"""
tensor([[[  2.9577,  -7.6124,  -0.0000,  ..., -29.9418,   5.8156, -27.1105],
         [  7.2757,  -3.0374,  62.7219,  ..., -41.3684, -17.8006,   9.9326],
         [ 22.5290,  42.4456, -27.7315,  ...,  10.1303,  -1.0908, -36.2931],
         [ 25.4460,  -7.6727,   7.2530,  ...,  18.9337, -25.3299, -12.3543]],

        [[  9.2815,  10.3302,  17.9495,  ..., -24.1487, -17.8344,  -7.0681],
         [ 11.9284,  -0.0000,  -5.3875,  ..., -18.3815,   0.0000,  -2.4167],
         [ 37.2720,  16.7200,  -0.0000,  ...,  -1.8779,   4.3047,   5.8277],
         [ -1.4356,  -9.8609,  16.0787,  ...,   0.0000, -14.3855,  -5.6584]]],
       grad_fn=<MulBackward0>)
torch.Size([2, 4, 512])
"""

