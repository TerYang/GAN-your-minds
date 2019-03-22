import numpy as np
import tensorflow as tf
import math

"""训练参数"""

dropoutrate = 0.5
tf.set_random_seed(1234)#设置流图中的随机种子
def make_var(name, shape, trainable=True):
    return tf.get_variable(name, shape, trainable=trainable)

def maxout(inputs, bind_num):
    """数据格式参数的变化，增加维度，倍选维度，
    如inputs.shape = [2,8],bind_num=4,则shape最终变为[-1,2,4],
    reshape之后，tensor变形，reduce_max"""
    shape = inputs.get_shape().as_list()
    #get_shape(),inputs must be a tensor return tupple
    # tf.shape(a),param a could be a array,list or tensor
    in_chan = shape[-1]
    shape[-1] = in_chan//bind_num#除法取下限
    shape += [bind_num]#列表增加一个长度
    shape[0] = -1
    return tf.reduce_max(tf.reshape(inputs,shape), -1,keepdims= False)


def conv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="conv2d", biased=False):
    """定义卷积层"""
    """
    input_:输入tensor
    output_dim：输出tensor shape
    kernel_size：卷积核尺寸weight width
    stride：步长
    padding：类型
    biased：偏置
    """
    input_dim = input_.get_shape()[-1]

    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output



# def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding="SAME", name="atrous_conv2d", biased=False):
#     """空洞卷积层"""
#     input_dim = input_.get_shape()[-1]
#     with tf.variable_scope(name):
#         kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
#         output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding=padding)
#         if biased:
#             biases = make_var(name='biases', shape=[output_dim])
#             output = tf.nn.bias_add(output, biases)
#         return output


def deconv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="deconv2d"):
    """定义反卷积层"""
    input_dim = input_.get_shape()[-1]  # 输入tensor列数，即维度

    input_height = int(input_.get_shape()[1])  # 图片的长宽
    input_width = int(input_.get_shape()[2])  # 图片的长宽
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, output_dim, input_dim])  # 卷积个数为
        output = tf.nn.conv2d_transpose(input_, kernel,
                                        [1, input_height * 2, input_width * 2, output_dim], [1, stride, stride, 1],
                                        padding=padding)
        return output


def batch_norm(input_, name="batch_norm"):
    """定义batchnorm(批次归一化)层"""
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32),)
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))

        mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_ - mean) * inv
        output = scale * normalized + offset
        return output


def lrelu(x, leak=0.2, name="lrelu"):
    """lrelu激活层"""
    return tf.maximum(x, leak * x)


def generator(noise, gf_dim=1, reuse=False, name="generator"):
    """定义生成器，主要由4个反卷积层组成"""
    # batch_picture = np.expand_dims(np.array(noise).astype(np.float32), axis=0)  # 填充维度

    input_dim = int(noise.get_shape()[-1])  # 获取输入通道
    dropout_rate = 0.5  # 定义dropout的比例

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # 第一个反卷积层，输入尺寸（1,32,32,1）输出尺度[1, 64, 64, 1
        d1 = deconv2d(input_=tf.nn.relu(noise), output_dim=gf_dim, kernel_size=4, stride=2, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)  # 随机扔掉一般的输出
        # d1 = tf.concat([batch_norm(d1, name='g_bn_d1'), e7], 3)

        # 第二个反卷积层，输出尺度[1, 128, 128 1]
        d2 = deconv2d(input_=tf.nn.relu(d1), output_dim=gf_dim, kernel_size=4, stride=2, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)  # 随机扔掉一般的输出
        # d2 = tf.concat([batch_norm(d2, name='g_bn_d2'), e6], 3)

        # 第三个反卷积层，输出尺度[1, 256 256, 1]
        d3 = deconv2d(input_=tf.nn.relu(d2), output_dim=gf_dim, kernel_size=4, stride=2, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)  # 随机扔掉一般的输出
        # d3 = tf.concat([batch_norm(d3, name='g_bn_d3'), e5], 3)

        # 第四个反卷积层，输出尺度[1, 512, 512, 1]
        d8 = deconv2d(input_=tf.nn.relu(d3), output_dim=1, kernel_size=4, stride=2, name='g_d8')
        return tf.nn.tanh(d8)


def discriminator(image, df_dim=16, reuse=False, name="discriminator"):
    """定义判别器"""
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        # dis_input = tf.concat([image, targets], 3)  # 在第三维度拼接

        """第1个卷积模块，输入尺寸：1,512,512,1,输出尺度: 1*256*256,16"""
        h0 = tf.nn.relu(conv2d(input_=image, output_dim=df_dim,kernel_size=4, stride=2, name='con1'),name='d1')
        print("h0.shape",h0.get_shape())
        #"""第2个卷积模块，输入尺寸：1*256*256,16,输出尺度: 1*128*128*32"""
        h1 = tf.nn.relu(batch_norm(conv2d(input_=h0, output_dim=df_dim * 2, kernel_size=4, stride=2, name='conv2'),name='bn1'))
        print("h1.shape", h1.get_shape())
        """pooling模块，输入尺寸：1*128*128*32,输出尺度: 1*64*64*32"""
        p0 = lrelu(batch_norm(tf.nn.max_pool(h1,[1,2,2,1],[1,2,2,1],padding='SAME',name='dp0'),name='bn2'))
        print("p0.shape", p0.get_shape())
        #"""第3个卷积模块，输入尺寸：1*64*64*32,输出尺度: 1*32*32*64"""
        h2 = tf.nn.relu(batch_norm(conv2d(input_=p0, output_dim=df_dim * 4, kernel_size=4, stride=2, name='conv3'),name='bn3'))
        print("h2.shape", h2.get_shape())
        #"""第4个卷积模块，输入尺寸：1*32*32*64,输出尺度: 1*16*16*128"""
        h3 =tf.nn.relu(batch_norm(conv2d(input_=h2, output_dim=df_dim * 8, kernel_size=4, stride=2, name='conv4'),name='bn4'))
        print("h3.shape", h3.get_shape())
        #"""第5个卷积模块，输入尺寸：: 1*16*16*128,ouput size:1*8*8*256"""
        h4 = tf.nn.relu(batch_norm(conv2d(input_=h3, output_dim=df_dim *16, kernel_size=4, stride=2, name='conv5'),name='bn5'))
        print("h4.shape", h4.get_shape())
        """第2个pooling模块，输入size:1*8*8*256,输出尺度: 1*2*2*256"""
        p1 = lrelu(batch_norm(tf.nn.max_pool(h4,[1,6,6,1],[1,2,2,1],padding='SAME',name='dp1'),name='bn7'))
        print("p1.shape", p1.get_shape())

        # """sigmoid模块，输入尺寸：: 1*16*16*128,ouput size:1*8*8*256"""
        dis_out = tf.sigmoid(p1,name='sigmoid0')  # 在输出之前经过sigmoid层，因为需要进行log运算
        print("dis_out.shape", dis_out.get_shape())
        return dis_out
