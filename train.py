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


parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--snapshot_dir", default='./snapshots', help="path of snapshots")
#保存模型的路径./snapshots
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs")
#训练时保存可视化输出的路径./train_out
parser.add_argument("--batch_size", type=int, default=512, help="train batch size")
#网络输入的尺度 256
parser.add_argument("--random_seed", type=int, default=1234, help="random seed")
#随机数种子 1234
parser.add_argument('--learnr', type=float, default=0.0002, help='initial learning rate for adam')
#学习率learn rate = 0.0002
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
#训练的epoch=200
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
#adam优化器的beta1=0.5
parser.add_argument("--summary_pred_every", type=int, default=100, help="times to summary.")
#训练中每过200step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=100, help="times to write.")
#训练中每过100step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=5000, help="times to save.")
#训练中每过5000step保存模型(可训练参数)
parser.add_argument("--lamda_gan_weight", type=float, default=1.0, help="GAN lamda")
#训练中GAN_Loss前的乘数1.0
parser.add_argument("--train_picture_format", default='.png', help="format of training datas.")
#网络训练输入的图片的格式(图片在CGAN中被当做条件)
parser.add_argument("--train_label_format", default='.jpg', help="format of training labels.")
#网络训练输入的标签的格式(标签在CGAN中被当做真样本)
parser.add_argument("--train_picture_path", default='./dataset/train_picture/', help="path of training datas.")
#网络训练输入的图片路径./dataset/train_picture/

args = parser.parse_args() #用来解析命令行参数
EPS = 1e-12 #EPS用于保证log函数里面的参数大于零
 
def save(saver, sess, logdir, step): #保存模型的save函数
   model_name = 'model' #保存的模型名前缀
   checkpoint_path = os.path.join(logdir, model_name) #模型的保存路径与名称
   if not os.path.exists(logdir): #如果路径不存在即创建
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step) #保存模型
   print('The checkpoint has been created.')
 
def l1_loss(src, dst): #定义l1_loss
    return tf.reduce_mean(tf.abs(src - dst))
 
def main(): #训练程序的主函数
    # greate the path where the out param saved
    if not os.path.exists(args.snapshot_dir): #如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): #如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)

    train_picture_list = glob.glob(os.path.join(args.train_picture_path, "*")) #得到训练输入图像路径名称列表
    tf.set_random_seed(args.random_seed) #初始一下随机数

    """输入容器"""
    prac_data = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 3],name='discrimination_holder') #输入的训练图像
    noise_data = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 3],name='generator_holder') #输入与训练图像匹配标签
    #
    # print(prac_data.get_shape())
    # print(noise_data.get_shape())
    # exit()

    """网络生成器输出模拟，判决器分别对真实标签和模拟标签进行判决"""
    genera_data = generator(noise=noise_data) #得到生成器的输出[1,512,512,1]
    discr_prac = discriminator(image=prac_data) #判别器返回的对真实标签的判别结果

    discr_fake = discriminator(image=genera_data) #判别器返回的对生成(虚假的)标签判别结果

    """损失函数"""
    fake_loss = tf.reduce_mean(-tf.log(discr_fake + EPS)) #计算生成器在判决器中的判决
    g_diff_d = tf.reduce_mean(l1_loss(genera_data, prac_data)) #判决器训练前，计算生成数据和实际数据差距
    prac_loss = tf.reduce_mean(-tf.log(discr_prac)) #计算训练实际数据的损害

    genera_loss = fake_loss+g_diff_d# 计算生成器的loss
    discr_loss = tf.reduce_mean(-(tf.log(discr_prac + EPS) + tf.log(1 - discr_fake + EPS))) #计算判别器的loss

    """tensorboard内容"""
    gen_loss_sum = tf.summary.scalar("gen_loss", genera_loss) #记录生成器loss的日志
    dis_loss_sum = tf.summary.scalar("dis_loss", discr_loss) #记录判别器loss的日志

    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph()) #日志记录器

    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name] #所有生成器的可训练参数
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name] #所有判别器的可训练参数

    d_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1) #判别器训练器
    g_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1) #生成器训练器

    """梯度计算"""
    d_grads_and_vars = d_optim.compute_gradients(discr_loss, var_list=d_vars) #计算判别器参数梯度
    d_train = d_optim.apply_gradients(d_grads_and_vars) #更新判别器参数

    g_grads_and_vars = g_optim.compute_gradients(genera_loss, var_list=g_vars) #计算生成器参数梯度
    g_train = g_optim.apply_gradients(g_grads_and_vars) #更新生成器参数

    """参数更新操作"""
    train_op = tf.group(d_train, g_train) #train_op表示了参数更新操作
    """系统设置"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #设定显存不超量使用
    sess = tf.Session(config=config) #新建会话层
    init = tf.global_variables_initializer() #参数初始化器
    sess.run(init) #初始化所有可训练参数

    # 模型保存器
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50)

    counter = 0 #counter记录训练步数

    for epoch in range(args.epoch): #训练epoch数
        shuffle(train_picture_list) #每训练一个epoch，就打乱一下输入的顺序
        for step in range(len(train_picture_list)): #每个训练epoch中的训练step数
            counter += 1
            picture_name, _ = os.path.splitext(os.path.basename(train_picture_list[step])) #获取图片名称

	        #读取一张训练图片，一张训练标签，以及相应的高和宽
            picture_resize, label_resize, picture_height, picture_width = ImageReader(file_name=picture_name, picture_path=args.train_picture_path, label_path=args.train_label_path, picture_format = args.train_picture_format, label_format = args.train_label_format, size = args.image_size)

            batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis = 0) #填充维度
            batch_label = np.expand_dims(np.array(label_resize).astype(np.float32), axis = 0) #填充维度

            feed_dict = { train_picture : batch_picture, train_label : batch_label } #构造feed_dict

            gen_loss_value, dis_loss_value, _ = sess.run([gen_loss, dis_loss, train_op], feed_dict=feed_dict) #得到每个step中的生成器和判别器loss

            if counter % args.save_pred_every == 0: #每过save_pred_every次保存模型
                save(saver, sess, args.snapshot_dir, counter)
            if counter % args.summary_pred_every == 0: #每过summary_pred_every次保存训练日志
                gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss_sum, dis_loss_sum], feed_dict=feed_dict)
                summary_writer.add_summary(gen_loss_sum_value, counter)
                summary_writer.add_summary(discriminator_sum_value, counter)
            if counter % args.write_pred_every == 0: #每过write_pred_every次写一下训练的可视化结果

                gen_label_value = sess.run(gen_label, feed_dict=feed_dict) #run出生成器的输出

                write_image = get_write_picture(picture_resize, gen_label_value, label_resize, picture_height, picture_width) #得到训练的可视化结果
                write_image_name = args.out_dir + "/out"+ str(counter) + ".png" #待保存的训练可视化结果路径与名称
                cv2.imwrite(write_image_name, write_image) #保存训练的可视化结果
            print('epoch {:d} step {:d} \t gen_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, gen_loss_value, dis_loss_value))

if __name__ == '__main__':
    # train_picture_list = glob.glob(os.path.join(args.train_picture_path, "*"))
    # print(train_picture_list)
    main()
