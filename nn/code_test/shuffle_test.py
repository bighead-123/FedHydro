import numpy as np
import torch
import tensorflow as tf

if  __name__ == '__main__':
    # a = np.array([1, 2, 3, 4, 5, 6])
    # np.random.shuffle(a)
    # print(a)
    A = torch.ones(2, 3)
    B = torch.ones(4, 3)
    C = tf.ones(2, 3)
    D = tf.ones(4, 3)
    list = []
    list2 = []
    list.append(A)
    list.append(B)
    list2.append(C)
    list2.append(D)
    print(torch.cat((A, B)))
    print(torch.cat(list))
    print(tf.concat(list2, axis=1))

