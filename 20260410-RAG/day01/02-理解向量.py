import numpy as np
v = np.array([3,4])
magnitude = np.linalg.norm(v) #计算大小
unit_vector = v/magnitude #计算单位向量
angle = np.arctan2(v[1],v[0]) *100/np.pi  #角度
print(f"大小:{magnitude}")
print(f"单位向量:{unit_vector}")
print(f"角度:{angle:.1f}")

"""
大小:5.0
单位向量:[0.6 0.8]
角度:29.5
"""