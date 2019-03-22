from __future__ import print_function

import argparse
from random import shuffle
import random
import os
import sys
import math
import tensorflow as tf
import glob
from nets import *

if __name__ == "__main__":
    # noise = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 3],name='normal_noise') #输入的训练图像
    # train_block = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 3],name='train_block') #输入与训练图像匹配标签

    a = tf.random_normal([1, 32, 32, 1])

    gen_label = generator(noise=a, gf_dim=1, reuse=False, name='generator') #得到生成器的输出
    d_tensor = discriminator(gen_label,16)
    # soft_tensor = tf.nn.softmax(d_tensor,axis=[-1])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 设定显存不超量使用
    sess = tf.Session(config=config)  # 新建会话层
    init = tf.global_variables_initializer()  # 参数初始化器

    sess.run(init)

    print(gen_label.get_shape())
    print(d_tensor.get_shape())
    # print(d_tensor.get_shape())
