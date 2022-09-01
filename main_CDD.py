# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:42:28 2020

@author: Think
"""
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
import argparse
from model_CDD import Model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, auc
import sklearn.decomposition as dp
import scipy.io as sio
import scipy
from data_CDD import Data_CDD
import math
import random

class Main():
    def __init__(self):
    		self.trainingmode = tf.constant(True,dtype=tf.bool)#训练模式
    		self.testingmode = tf.constant(False,dtype=tf.bool)#测试模式
    		self.learning_rate=0.004#学习率
    		self.testnum=3000#测试样本总数
    		self.test_batch_size=1#单次测试样本数量
    		self.n1=1000#多少次显示一次loss，多少次存储一次模型
    		self.n2=100000#第多少次训练后开始测试  
    		self.losssave=[]#保存loss的值
    		self.networkstye="FC_EF_SE"
    		self.parser = argparse.ArgumentParser(description='depth completion')
    		self.parser.add_argument('--BATCH_SIZE',dest='BATCH_SIZE',default=40)
    		self.parser.add_argument('--epoch',dest='epoch',default=100002) 
    		self.parser.add_argument('--IMG_H',dest='IMG_H',default=256)
    		self.parser.add_argument('--IMG_W',dest='IMG_W',default=256)
    		self.args = self.parser.parse_args()
    		self.index=1
              
    def train(self):
        #占位符img1
        img1 = tf.placeholder(tf.float32,[self.args.BATCH_SIZE, self.args.IMG_H, self.args.IMG_W, 3],name='img1')
        #占位符img2
        img2=tf.placeholder(tf.float32,[self.args.BATCH_SIZE, self.args.IMG_H, self.args.IMG_W, 3],name='img2')
        #占位符gt
        gt= tf.placeholder(tf.float32,[self.args.BATCH_SIZE, self.args.IMG_H, self.args.IMG_W,2],name='gt')

        myseed =1998#np.random.randint(1, 100)
        print("seed ",myseed)
        
        ########################data argument#################################
        img1=tf.image.random_brightness(img1,max_delta=0.3,seed=myseed)
        img2=tf.image.random_brightness(img2,max_delta=0.3,seed=myseed)
        
        img1 = tf.image.random_saturation(img1,lower=0.2,upper=1.8)
        img2 = tf.image.random_saturation(img2,lower=0.2,upper=1.8)
        
        img1=tf.image.random_hue(tf.squeeze(img1),max_delta=0.1,seed=myseed)
        img1=tf.image.random_hue(tf.squeeze(img1),max_delta=0.1,seed=myseed)
        
        img1=tf.image.random_flip_up_down(img1,seed=myseed)
        img2=tf.image.random_flip_up_down(img2,seed=myseed)
        gt=tf.image.random_flip_up_down(gt,seed=myseed)
        
        
        img1 = tf.image.random_flip_left_right(img1,seed=myseed)
        img2 = tf.image.random_flip_left_right(img2,seed=myseed)
        gt = tf.image.random_flip_left_right(gt,seed=myseed)
        
        shape = [self.args.BATCH_SIZE, self.args.IMG_H, self.args.IMG_W, 3]
        
        dg=[90,180,270,360]
        index=random.randint(0,3)
        img1=tf.contrib.image.rotate(img1, dg[index] * math.pi / 180, interpolation='BILINEAR')
        img2=tf.contrib.image.rotate(img2, dg[index] * math.pi / 180, interpolation='BILINEAR')
        gt=tf.contrib.image.rotate(gt, dg[index] * math.pi / 180, interpolation='BILINEAR')
        ########################data argument################################
        
        gt_3d=gt
        gt_2d=tf.argmax(gt,axis=3)
        
        change_num=tf.reduce_sum(tf.cast(tf.equal(gt_2d,1),dtype=tf.float64))
        unchange_num=tf.reduce_sum(tf.cast(tf.equal(gt_2d,0),dtype=tf.float64))
        all_num=change_num+unchange_num
        
        rate1=unchange_num/all_num
        rate2=change_num/all_num
        
        img1 = tf.cast(img1,dtype=tf.float32)
        img2 = tf.cast(img2,dtype=tf.float32)
        
        img1 = tf.image.per_image_standardization(img1)
        img2 = tf.image.per_image_standardization(img2)
        
        #选择合适的网络结构
        net=Model(self.args)
        if self.networkstye=="FC_EF_SE":
            pred_3d=net.FC_EF_SE(img1,img2,net.trainingmode,rate1,rate2)
        else:
            print("network erros")

        pred_2d=pred_3d[:,:,:,1]
        
        pf=tf.layers.flatten(pred_3d,'predict_flatten')
        
        gf=tf.layers.flatten(gt_3d,'gt_flatten')
        
        gt_1d=tf.reshape(gt_2d,[-1],name=None)
        pred_1d=tf.reshape(pred_2d,[-1],name=None)

        weight=rate1/rate2
        weight=weight

        weight=tf.cast(weight,tf.float32)
        l=net.new_loss(pf,gf,weight)#############损失函数还有待确定

        learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=net.global_step,decay_steps=2500, decay_rate=0.5)
        
        train_op,regularization_loss=net.optimize(self.learning_rate)#优化方法
        
        data_manager = Data_CDD()
        output_path='CDD256.tfrecord'
        dataset = tf.data.TFRecordDataset(output_path)
        
        dataset = dataset.map(data_manager._parse_record)
        dataset = dataset.repeat(20000)
        dataset = dataset.batch(self.args.BATCH_SIZE)
        dataset = dataset.shuffle(buffer_size=302)#这个地方需要比batchsize大一点
        iterator = dataset.make_one_shot_iterator()
        img1tf,img2tf,gttf,img_id,patch_id = iterator.get_next()
        saver = tf.train.Saver()
        max_f1=0
        max_f1_index=-1
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(self.args.epoch):
                img1_,img2_,gt_=sess.run([img1tf,img2tf,gttf])
                rl,step,l_,train_op_= sess.run([regularization_loss,net.global_step,l,train_op],feed_dict={img1: img1_, img2:img2_, gt: gt_})
                pre2,gt2= sess.run([pred_2d,gt_2d],feed_dict={img1: img1_, img2:img2_, gt: gt_})
                pred1,gt1= sess.run([pred_1d,gt_1d],feed_dict={img1: img1_, img2:img2_, gt: gt_})
                pred3,gt3= sess.run([pred_3d,gt_3d],feed_dict={img1: img1_, img2:img2_, gt: gt_})  
                change_num_,unchange_num_= sess.run([change_num,unchange_num],feed_dict={img1: img1_, img2:img2_, gt: gt_})
                weight_=sess.run([weight],feed_dict={img1: img1_, img2:img2_, gt: gt_})
                if i%self.n1==0 and i!=0:
                    print("Training step: %d, loss: %g regulation loss:%g  weight:%g" % (i, l_,rl,weight_[0]))
                    self.losssave.append(l_)
                    saver.save(sess, './model/my_test_model')
                if i%self.n2==0 and i!=0:
                    cr1=self.one_test(sess,net,self.test_batch_size,i)#############LIU

    def one_test(self,sess,net,batch_size,step):
        path="./testresult"+str(step)
        os.makedirs(path)

        img1_p = tf.placeholder(tf.float32,[batch_size, 256, 256, 3],name='img1_test')
        img2_p=tf.placeholder(tf.float32,[batch_size, 256, 256, 3],name='img2_test')
        gt_p= tf.placeholder(tf.float32,[batch_size, 256, 256,2],name='gt_test')

        gt_2d=tf.argmax(gt_p,axis=3)
        img1_p = tf.image.per_image_standardization(img1_p)
        img2_p = tf.image.per_image_standardization(img2_p)
        img1_p = tf.cast(img1_p,dtype=tf.float32)
        img2_p = tf.cast(img2_p,dtype=tf.float32)
        
        if self.networkstye=="FC_EF_SE":
            predict=net.FC_EF_SE(img1_p,img2_p,net.testingmode,3,3)
        else:
            print("network erros")
        
        pred_2d=tf.argmax(predict, axis=3)
        pred_2d=tf.cast(pred_2d,dtype=tf.float32)
        gt_1d=tf.reshape(gt_2d,[-1],name=None)
        pred_1d=tf.reshape(pred_2d,[-1],name=None)
        data_manager2 = Data_CDD()
        output_path2='CDD_test.tfrecord'
        datasett = tf.data.TFRecordDataset(output_path2)
        datasett = datasett.map(data_manager2._parse_record2)
        datasett = datasett.repeat(10)
        datasett = datasett.batch(batch_size)
        iteratortt = datasett.make_one_shot_iterator()
        img1tt,img2tt,gttt,img_idt,patch_idt = iteratortt.get_next()

        num=self.testnum/batch_size
        num=int(num)
        sess2=tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess2, "./model/my_test_model")
        
        my_cm=[[0,0],[0,0]]
        
        
        for i in range(num):
            img1_,img2_,gt_=sess2.run([img1tt,img2tt,gttt])
            pre2,gt2= sess2.run([pred_2d,gt_2d],feed_dict={img1_p: img1_, img2_p:img2_, gt_p: gt_})
            pred1,gt1= sess2.run([pred_1d,gt_1d],feed_dict={img1_p: img1_, img2_p:img2_, gt_p: gt_})
            predict_= sess2.run(predict,feed_dict={img1_p: img1_, img2_p:img2_, gt_p: gt_})
            
            img1_ = img1_/255.
            img2_ = img2_/255.

            batch,rows,cols=pre2.shape  
    
            pre2=pre2

            try:
                cr2=classification_report(gt1, pred1,target_names=['unchanged','changed'],output_dict=True,digits=4)
            except:
                continue
            p=cr2['changed']['precision']
            r=cr2['changed']['recall']
            f=cr2['changed']['f1-score']

            dd=confusion_matrix(gt1, pred1)
            my_cm=my_cm+dd

            scipy.misc.imsave("./testresult"+str(step)+"/"+str(i)+"output.png", pre2[0,:,:])
                    
            plt.figure(i)
                   
            plt.subplot(221) 
            plt.axis('off') 
            plt.title("f:"+str(f),fontsize='medium',fontweight='bold')
            plt.imshow(img1_[0,:,:,:])
        
            plt.subplot(222) 
            plt.axis('off') 
            plt.title("p:"+str(p),fontsize='medium',fontweight='bold')
            plt.imshow(img2_[0,:,:,:])

            plt.subplot(223) 
            plt.axis('off') 
            plt.title("r:"+str(r),fontsize='medium',fontweight='bold')
            plt.imshow(pre2[0,:,:])

            plt.subplot(224) 
            plt.axis('off') 
            plt.title('ground truth',fontsize='medium',fontweight='bold')
            plt.imshow(gt2[0,:,:])
                    
            plt.savefig("./testresult"+str(step)+"/"+str(i)+"result.png",) 
                    
            plt.close()
            
        print("check for a while")

        TP=my_cm[1][1]
        FN=my_cm[1][0]
        FP=my_cm[0][1]
        TN=my_cm[0][0]
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        f1_score=2*precision*recall/(precision+recall)
        a1=(TN+TP)/(TP+FN+FP+TN)
        
        print("the precision,recall,f1-score are :")
        print("precision=",precision)
        print("recall=",recall)
        print("f1_score=",f1_score)
        print("accuracy1=",a1)
        print("混淆矩阵为：")
        print(my_cm)
        
        plt.figure(9999)
        plt.subplot(311) 
        plt.axis('off') 
        plt.title("f1_score:"+str(f1_score),fontsize='medium',fontweight='bold')
        plt.subplot(312) 
        plt.axis('off') 
        plt.title("precision:"+str(precision),fontsize='medium',fontweight='bold')
        plt.subplot(313) 
        plt.axis('off') 
        plt.title("recall:"+str(recall),fontsize='medium',fontweight='bold')
        plt.savefig("./testresult"+str(step)+"/"+"final_result.png")              
        plt.close()
        return cr2
  
if __name__ == '__main__':
    
	tf.reset_default_graph()
	m=Main()
	m.train()  