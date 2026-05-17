"""
#编码器部分
由N个编码器堆叠而成
每个编码器层有两个子层连接结构组成
第一个子层连接结构包含一个多头自注意力层和规范化层以及一个残差连接
第二个子层链接结构包括一个前馈全连接子层和规范化层以及一个残差连接


#掩码张量
什么是？
掩代码遮掩，码就是我们张量中的数值，它的尺寸不定，里面一般只有1和0的元素。代表位置被遮掩或者不被遮掩。


在transformer，掩码张量主要应用在attention中
"""
import numpy as np
import torch
#构建掩码张量的函数
def subsequent_mask(size):
    #size: 代码掩码张量后两个维度，形成一个矩阵
    attn_shape = (1,size, size)
    #使用np.ones先构建一个全1的张量，然后利用np.triu形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    #使得这个三角矩阵反转
    return torch.from_numpy(1 - subsequent_mask)

size=5
sm = subsequent_mask(size)
print(sm)
"""
tensor([[[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1]]], dtype=torch.uint8)
"""