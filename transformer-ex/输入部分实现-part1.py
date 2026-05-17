"""
文本嵌入层的作用：无论是源文本嵌入还是目标文本嵌入，都是为了将文本中词汇的数字表示转变为向量表示
希望在这样的高维空间捕捉词汇间的关系

#按照torch，numpy,matplotlib,seaborn
"""
import torch
#预定义的网络层torch.nn,工具开发者已经帮助我们开发好的一些常用层
#比如 卷积层，lstm层，embedding层等，不需要我们再造轮子
import torch.nn as nn

#数据计算工具包
import math

#torch中变量封装函数Variable
from torch.autograd import Variable

#定义Embeddings类实现文本嵌入层，这里s说明代表两个一模一样的嵌入层，他们共享参数
#该类继承nn.Module,这样就有标准层的一些功能
#构建Embedding类来实现文本嵌入层
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

embedding = nn.Embedding(10,3)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(embedding(input))
"""
tensor([[[ 1.3499, -0.0862, -1.9520],
         [ 0.2244,  0.3669,  0.1404],
         [ 1.3372, -2.6520, -1.6524],
         [ 0.4546, -0.6353, -0.8617]],

        [[ 1.3372, -2.6520, -1.6524],
         [ 1.0867, -1.6906,  0.6300],
         [ 0.2244,  0.3669,  0.1404],
         [-0.6019,  0.1517, -0.4892]]], grad_fn=<EmbeddingBackward0>)
"""

embedding1 = nn.Embedding(10,3,padding_idx=0)
input = torch.LongTensor([0,2,0,5])
print(embedding1(input))
"""
tensor([[ 0.0000,  0.0000,  0.0000],
        [-1.6227, -0.1606, -1.4749],
        [ 0.0000,  0.0000,  0.0000],
        [-1.1297,  1.3681,  1.1692]], grad_fn=<EmbeddingBackward0>)
"""

d_model=512
vocab=1000
x=Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
emb = Embeddings(d_model,vocab)
embr = emb(x)
print("embr:",embr)
print(embr.shape)  #torch.Size([2, 4, 512])