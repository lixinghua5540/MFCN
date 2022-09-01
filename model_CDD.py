# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:28:28 2020
@author: Think
"""

import numpy as np
import cv2
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from sklearn.metrics import classification_report
import argparse

class Model():
    def __init__(self, arg):
        self.arg = arg
        self.trainingmode = tf.constant(True,dtype=tf.bool)
        self.testingmode = tf.constant(False,dtype=tf.bool)
        self.global_step = tf.Variable(0, trainable=False)
        self.BATCH_SIZE=arg.BATCH_SIZE
        self.IMG_W=arg.IMG_W
        self.IMG_H=arg.IMG_H
        
        self.dropoutrate=0.02

    def conv_k3_s2(self,x,filters_num):
        with tf.variable_scope('conv_k3_s2',reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(2,2),padding='same',use_bias=True,kernel_initializer=tf.random_normal_initializer() ,bias_initializer=tf.random_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
            return x

    def conv_k1_s1(self,x,filters_num):
        with tf.variable_scope('conv_k1_s1',reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(inputs=x,filters=filters_num,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.random_normal_initializer() ,bias_initializer=tf.random_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
            return x

    def conv_k3_s1(self,x,filters_num):
        with tf.variable_scope('conv_k3_s1',reuse=tf.AUTO_REUSE):
            in_chennel = x.get_shape()[-1]
            x = tf.layers.conv2d(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.random_normal_initializer() ,bias_initializer=tf.random_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
            return x
        
    def conv_k5_s1(self,x,filters_num):
        with tf.variable_scope('conv_k5_s1',reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(inputs=x,filters=filters_num,kernel_size=(5,5),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.random_normal_initializer() ,bias_initializer=tf.random_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
            return x
    
    def conv_k7_s1(self,x,filters_num):
        with tf.variable_scope('conv_k7_s1',reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(inputs=x,filters=filters_num,kernel_size=(7,7),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.random_normal_initializer() ,bias_initializer=tf.random_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
            return x
    
    def deconv_k3_s2(self,x,filters_num):
        with tf.variable_scope('deconv_k3_s2',reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d_transpose(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(2,2),padding='same',use_bias=True,kernel_initializer=tf.random_normal_initializer() ,bias_initializer=tf.random_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
            return x

    def deconv_k1_s1(self,x,filters_num):
        with tf.variable_scope('deconv_k1_s1',reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d_transpose(inputs=x,filters=filters_num,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.random_normal_initializer() ,bias_initializer=tf.random_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
            return x

    def deconv_k3_s1(self,x,filters_num):
        with tf.variable_scope('deconv_k3_s1',reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d_transpose(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.random_normal_initializer() ,bias_initializer=tf.random_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
            return x
        
    def Relu(self,x):
        return tf.nn.relu(x)

    def muti_scale_ef(self,img1,in_channel,is_training):
        with tf.variable_scope('muti_feature',reuse=tf.AUTO_REUSE) as scope0:
            with tf.variable_scope('block1',reuse=tf.AUTO_REUSE) as scope0:
                f1=self.conv_k1_s1(img1,in_channel//2)
            
            with tf.variable_scope('block2',reuse=tf.AUTO_REUSE) as scope0:
                f3=self.conv_k1_s1(img1,in_channel//2)
                f3=self.conv_k3_s1(f3,in_channel//2)
            
            with tf.variable_scope('block3',reuse=tf.AUTO_REUSE) as scope1:
                f5=self.conv_k1_s1(img1,in_channel//2)
                f5=self.conv_k5_s1(f5,in_channel//2)
           
            with tf.variable_scope('block4',reuse=tf.AUTO_REUSE) as scope2:
                fp=tf.layers.max_pooling2d(img1,pool_size=(3,3),strides=1,padding='same')
                fp=self.conv_k1_s1(fp,in_channel) 
                fc1=tf.concat([f1,f3,f5,fp],axis=3)
                
            with tf.variable_scope('block5',reuse=tf.AUTO_REUSE) as scope2:                
                fc1=self.conv_k1_s1(fc1,in_channel)
                fc1=tf.layers.batch_normalization(fc1,training=is_training,name='bn1',reuse=tf.AUTO_REUSE)
                fc1=tf.nn.relu(fc1)
                return fc1

    def FC_EF_SE(self,img1,img2,is_training,rate1,rate2):
        with_dropout=False
        shape = img1.get_shape().as_list()
        in_channel = shape[3]
        H=shape[1]
        W=shape[2]
        
        mutiscale=False
        attention=False
        SE=False
        Res_block=False
        
        ratio=4
        
        EF=tf.concat([img1,img2],axis=3)
        
        with tf.variable_scope('layer1',reuse=tf.AUTO_REUSE) as scope1:
            x11=self.conv_k3_s1(EF,16)
            x11 = tf.layers.batch_normalization(x11,training = is_training,name='b1',reuse=tf.AUTO_REUSE)
            x11 = tf.nn.relu(x11)

        with tf.variable_scope('layer1_1',reuse=tf.AUTO_REUSE) as scope11:
            x111=self.conv_k3_s1(x11,16)
            x111 = tf.layers.batch_normalization(x111,training = is_training,name='b12',reuse=tf.AUTO_REUSE)
            x111 = tf.nn.relu(x111)

        with tf.variable_scope('substract1',reuse=tf.AUTO_REUSE) as s1: 
            sub1=x111
            
        with tf.variable_scope('maxpooling1',reuse=tf.AUTO_REUSE) as scope111:
            x111=self.muti_scale_ef(x111,16,is_training)
            x11p=tf.layers.max_pooling2d(x111,pool_size=(2,2),strides=2,padding='same')#(200, 16, 16, 16)

        with tf.variable_scope('layer2',reuse=tf.AUTO_REUSE) as scope2:
            x21=self.conv_k3_s1(x11p,32)
            x21 = tf.layers.batch_normalization(x21,training = is_training,name='b2',reuse=tf.AUTO_REUSE)
            x21 = tf.nn.relu(x21)

        with tf.variable_scope('layer2_2',reuse=tf.AUTO_REUSE) as scope22:
            x211=self.conv_k3_s1(x21,32)
            x211 = tf.layers.batch_normalization(x211,training = is_training,name='b22',reuse=tf.AUTO_REUSE)
            x211 = tf.nn.relu(x211)

        with tf.variable_scope('substract2',reuse=tf.AUTO_REUSE) as s2:
            sub2=x211
            
        with tf.variable_scope('maxpooling2',reuse=tf.AUTO_REUSE) as scope222:
            x21p=tf.layers.max_pooling2d(x211,pool_size=(2,2),strides=2,padding='same')#(200, 8, 8, 32)

        with tf.variable_scope('layer3',reuse=tf.AUTO_REUSE) as scope3:
            x31=self.conv_k3_s1(x21p,64)
            x31 = tf.layers.batch_normalization(x31,training = is_training,name='b3',reuse=tf.AUTO_REUSE)
            x31 = tf.nn.relu(x31)

        with tf.variable_scope('layer3_2',reuse=tf.AUTO_REUSE) as scope33:
            x311=self.conv_k3_s1(x31,64)
            x311 = tf.layers.batch_normalization(x311,training = is_training,name='b32',reuse=tf.AUTO_REUSE)
            x311 = tf.nn.relu(x311)

        with tf.variable_scope('layer3_3',reuse=tf.AUTO_REUSE) as scope333:
            x3111=self.conv_k3_s1(x311,64)
            x3111 = tf.layers.batch_normalization(x3111,training = is_training,name='b33',reuse=tf.AUTO_REUSE)
            x3111 = tf.nn.relu(x3111)

        with tf.variable_scope('substract3',reuse=tf.AUTO_REUSE) as s2:
            sub3=x3111
            
        with tf.variable_scope('maxpooling3',reuse=tf.AUTO_REUSE) as scope3333:
            x3111=self.muti_scale_ef(x3111,64,is_training)
            x31p=tf.layers.max_pooling2d(x3111,pool_size=(2,2),strides=2,padding='same')#(200, 4, 4, 64)
             
            
        with tf.variable_scope('layer4',reuse=tf.AUTO_REUSE) as scope4:
            x41=self.conv_k3_s1(x31p,128)
            x41 = tf.layers.batch_normalization(x41,training = is_training,name='b4',reuse=tf.AUTO_REUSE)
            x41 = tf.nn.relu(x41)
            
        with tf.variable_scope('layer4_1',reuse=tf.AUTO_REUSE) as scope44:
            x411=self.conv_k3_s1(x41,128)
            x411 = tf.layers.batch_normalization(x411,training = is_training,name='b42',reuse=tf.AUTO_REUSE)
            x411 = tf.nn.relu(x411)

        with tf.variable_scope('layer4_2',reuse=tf.AUTO_REUSE) as scope444:
            x4111=self.conv_k3_s1(x411,128)
            x4111 = tf.layers.batch_normalization(x4111,training = is_training,name='b43',reuse=tf.AUTO_REUSE)
            x4111 =tf.nn.relu(x4111)

        with tf.variable_scope('substract4',reuse=tf.AUTO_REUSE) as s2:
            sub4=x4111
            
        with tf.variable_scope('maxpooling4',reuse=tf.AUTO_REUSE) as scope4444:
            x4111=self.muti_scale_ef(x4111,128,is_training)
            x41p=tf.layers.max_pooling2d(x4111,pool_size=(2,2),strides=2,padding='same')#(200, 2, 2, 128)
        
        with tf.variable_scope('deconv1',reuse=tf.AUTO_REUSE):
            x51=self.conv_k3_s1(x41p,128)
            x51 = tf.layers.batch_normalization(x51,training = is_training,name='b5',reuse=tf.AUTO_REUSE)
            x51 = tf.nn.relu(x51)

        with tf.variable_scope('deconv1_1',reuse=tf.AUTO_REUSE):
            x511=self.deconv_k3_s2(x51,128)
            x511 = tf.layers.batch_normalization(x511,training = is_training,name='b51',reuse=tf.AUTO_REUSE)
            x511 = tf.nn.relu(x511)
            x511 = tf.layers.dropout(x511, rate=self.dropoutrate, training=is_training)
            shape = sub4.get_shape().as_list()
            x511=tf.image.resize_images(x511,(shape[1],shape[2]),0)   
            x511=tf.concat([x511,sub4],axis=3)
            x511=self.conv_k1_s1(x511,128)

        with tf.variable_scope('deconv1_2',reuse=tf.AUTO_REUSE):
            x512=self.conv_k3_s1(x511,128)
            x512 = tf.layers.batch_normalization(x512,training = is_training,name='b52',reuse=tf.AUTO_REUSE)
            x512 = tf.nn.relu(x512)


        with tf.variable_scope('deconv1_3',reuse=tf.AUTO_REUSE):
            x513=self.conv_k3_s1(x512,128)
            x513 = tf.layers.batch_normalization(x513,training = is_training,name='b53',reuse=tf.AUTO_REUSE)
            x513 = tf.nn.relu(x513)

        with tf.variable_scope('deconv1_4',reuse=tf.AUTO_REUSE):
            x514=self.conv_k3_s1(x513,64)
            x514 = tf.layers.batch_normalization(x514,training = is_training,name='b54',reuse=tf.AUTO_REUSE)
            x514 = tf.nn.relu(x514)

        with tf.variable_scope('deconv2',reuse=tf.AUTO_REUSE):
            x61=self.conv_k3_s1(x514,64)
            x61 = tf.layers.batch_normalization(x61,training = is_training,name='b6',reuse=tf.AUTO_REUSE)
            x61 = tf.nn.relu(x61)
            
        with tf.variable_scope('deconv2_1',reuse=tf.AUTO_REUSE):
            x611=self.deconv_k3_s2(x61,64)
            x611 = tf.layers.batch_normalization(x611,training = is_training,name='b61',reuse=tf.AUTO_REUSE)
            x611 = tf.nn.relu(x611)
            x611 = tf.layers.dropout(x611, rate=self.dropoutrate, training=is_training)
            shape = sub3.get_shape().as_list()
            x611=tf.image.resize_images(x611,(shape[1],shape[2]),0)
            x611=tf.concat([x611,sub3],axis=3)
            x611=self.conv_k1_s1(x611,64)
            x611=self.muti_scale_ef(x611,64,is_training)

        with tf.variable_scope('deconv2_2',reuse=tf.AUTO_REUSE):
            x612=self.conv_k3_s1(x611,64)
            x612 = tf.layers.batch_normalization(x612,training = is_training,name='b62',reuse=tf.AUTO_REUSE)
            x612 = tf.nn.relu(x612)

        with tf.variable_scope('deconv2_3',reuse=tf.AUTO_REUSE):
            x613=self.conv_k3_s1(x612,64)
            x613 = tf.layers.batch_normalization(x613,training = is_training,name='b63',reuse=tf.AUTO_REUSE)
            x613 = tf.nn.relu(x613)
            
        with tf.variable_scope('deconv2_4',reuse=tf.AUTO_REUSE):
            x614=self.conv_k3_s1(x613,32)
            x614 = tf.layers.batch_normalization(x614,training = is_training,name='b64',reuse=tf.AUTO_REUSE)
            x614 = tf.nn.relu(x614)

        with tf.variable_scope('deconv3',reuse=tf.AUTO_REUSE):
            x71=self.deconv_k3_s2(x614,32)
            x71 = tf.layers.batch_normalization(x71,training = is_training,name='b7',reuse=tf.AUTO_REUSE)
            x71 = tf.nn.relu(x71)
            x71 = tf.layers.dropout(x71, rate=self.dropoutrate, training=is_training)
            shape = sub2.get_shape().as_list()
            x71=tf.image.resize_images(x71,(shape[1],shape[2]),0)
            x71=tf.concat([x71,sub2],axis=3)
            x71=self.conv_k1_s1(x71,32)
            x71=self.muti_scale_ef(x71,32,is_training)
            
        with tf.variable_scope('deconv3_1',reuse=tf.AUTO_REUSE):
            x711=self.conv_k3_s1(x71,32)
            x711 = tf.layers.batch_normalization(x711,training = is_training,name='b71',reuse=tf.AUTO_REUSE)
            x711 = tf.nn.relu(x711)

        with tf.variable_scope('deconv3_2',reuse=tf.AUTO_REUSE):
            x712=self.conv_k3_s1(x711,16)
            x712 = tf.layers.batch_normalization(x712,training = is_training,name='b72',reuse=tf.AUTO_REUSE)
            x712 = tf.nn.relu(x712)

        with tf.variable_scope('deconv3_3',reuse=tf.AUTO_REUSE):
            x713=self.deconv_k3_s2(x712,16)
            x713 = tf.layers.batch_normalization(x713,training = is_training,name='b73',reuse=tf.AUTO_REUSE)
            x713 = tf.nn.relu(x713)
            shape = sub1.get_shape().as_list()
            x713=tf.image.resize_images(x713,(shape[1],shape[2]),0)
            x713=tf.concat([x713,sub1],axis=3)
            x713=self.conv_k1_s1(x713,16)
            
        with tf.variable_scope('predict',reuse=tf.AUTO_REUSE):
            output=self.conv_k3_s1(x713,16)
            output = tf.layers.batch_normalization(output,training = is_training,name='o1',reuse=tf.AUTO_REUSE)
            output = tf.nn.relu(output)
            
        with tf.variable_scope('predict2',reuse=tf.AUTO_REUSE):
            output=self.conv_k3_s1(output,2)
            output = tf.layers.batch_normalization(output,training = is_training,name='o2',reuse=tf.AUTO_REUSE)
            output = tf.nn.relu(output)           
            output=tf.image.resize_images(output,(H,W),0)
            print("predict2 x shape2 %s"%(str(output.shape)))
            output=tf.nn.softmax(output)
        return output
    
    def focal_loss(self,prediction,labels,alpha=0.1,gamma=3):
        zeros=tf.zeros_like(prediction,dtype=prediction.dtype)
        pos_corr=tf.where(labels>zeros,labels-prediction,zeros)
        neg_corr=tf.where(labels>zeros,zeros,prediction)
        fl_loss=-alpha*(pos_corr**gamma)*tf.log(tf.clip_by_value(prediction,1e-8, 1.0))-(1-alpha)*(neg_corr**gamma)*tf.log(tf.clip_by_value(1.0-prediction,1e-8, 1.0))
        fl_loss=tf.reduce_sum(fl_loss)
        self.loss_cross_entropy=fl_loss
        return fl_loss
     
    def train_loss(self,predict,gt):
        zeros=tf.zeros_like(predict,dtype=predict.dtype)
        #predict=tf.where(gt>zeros,predict*0.8,predict*0.2)
        self.loss_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict,labels=gt)
        self.loss_cross_entropy = tf.reduce_mean(self.loss_cross_entropy)
        return self.loss_cross_entropy
    
    def new_loss(self,predict,gt,weight):
        predict = tf.cast(predict,dtype=tf.float32)
        gt = tf.cast(gt,dtype=tf.float32)
        self.loss_cross_entropy = tf.nn.weighted_cross_entropy_with_logits (predict, gt, weight)
        self.loss_cross_entropy = tf.reduce_sum(self.loss_cross_entropy)
        self.loss2=self.dice_coef(gt,predict)
        self.loss_cross_entropy=self.loss_cross_entropy+0.5*self.loss2
        return self.loss_cross_entropy
    
    
    def dice_coef(self,y_true, y_pred, smooth=1e-8, weight=0.5):
        intersection = tf.reduce_sum(y_true*y_pred)
        union1=tf.reduce_sum(y_true)+weight*tf.reduce_sum(y_pred)
        coef=(2.*intersection+smooth)/(union1+smooth)
        return 1-coef
        
    
    def optimize(self,learning_rate):
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss=regularization_loss+self.loss_cross_entropy
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_opt = tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step=self.global_step)
        return train_opt,regularization_loss

    def show_classification_report(self,predict,gt):
        target_name=['unchanged','changeed']
        print("check for a while")
        print(type(predict))
        print(type(gt))
        p=classification_report(gt,predict,target_name)
        print(type(p))