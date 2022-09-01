# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:09:13 2020

@author: Think
"""

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.decomposition as dp
from sklearn.decomposition import PCA
import cv2

class Data_CDD():
    def __init__(self):
        self.path='./CDD_TRAIN/'
        self.test_path='./CDD_TEST/'
        self.img1_path=[]
        self.img2_path=[]
        self.gt_path=[]
        self.get_image_list()
        self.img1=[]
        self.img2=[]
        self.gt=[]
        self.train_num=len(self.gt_path)
        self.patch_size=256

    def get_image_list(self):
            for file in os.listdir(self.path):
                for file2 in os.listdir(self.path+"/"+file):
                    for file3 in os.listdir(self.path+"/"+file+"/"+file2):
                        if file3=="im1.bmp":
                            self.img1_path.append(self.path+"/"+file+"/"+file2+"/"+file3)
                        if file3=="im2.bmp":
                            self.img2_path.append(self.path+"/"+file+"/"+file2+"/"+file3)
                        if file3=="gt.bmp":
                            self.gt_path.append(self.path+"/"+file+"/"+file2+"/"+file3)
    
    def read_imgs(self):
        lenth=len(self.gt_path)
        for i in range(lenth):
            img1_t=cv2.imread(self.img1_path[i])
            img2_t=cv2.imread(self.img2_path[i])          
            gt_t= cv2.imread(self.gt_path[i])                
            gt_t=gt_t[:,:,0]
            gt_t=gt_t/255.0
            self.img1.append(img1_t)
            self.img2.append(img2_t)
            self.gt.append(gt_t)

    def make_one_hot(self,gt):
        shape=gt.shape
        gt_H=shape[0]
        gt_W=shape[1]
        new_shape=[gt_H,gt_W,2]
        temp=np.zeros(new_shape, dtype=float)
        for i in range(gt_H):
            for j in range(gt_W):
                if gt[i,j]==1:
                    temp[i,j,1]=1
                    temp[i,j,0]=0
                if gt[i,j]==0:
                    temp[i,j,1]=0
                    temp[i,j,0]=1
                if gt[i,j]!=2 and  gt[i,j]!=1:
                    pass   
        return temp
            
    def show_one_sample(self):
        lenth=len(self.gt_path)
        for i in range(lenth):
            plt.figure()
            plt.subplot(131)
            plt.axis('off') 
            plt.title('im1',fontsize='medium',fontweight='bold')
            plt.imshow(self.img1[i])
            plt.subplot(132)
            plt.axis('off') 
            plt.title('im2',fontsize='medium',fontweight='bold')
            plt.imshow(self.img2[i])
            plt.subplot(133)
            plt.axis('off') 
            plt.title('gt',fontsize='medium',fontweight='bold')
            plt.imshow(self.gt[i])
             
    def padding(self,img,padding_h,padding_w):
        input_shape=img.shape
        n=len(input_shape)
        if n==3:
            img_new=np.pad(img,((0,padding_h),(0,padding_w),(0,0)),constant_values=((0,1),(0,1),(0,0))) 
        if n==2:
            img_new=np.pad(img,((0,padding_h),(0,padding_w)),constant_values=((0,1),(0,1))) #1 means unchanged
        return img_new
    
    def img_cropping(self,img,crop_size):
        img_H=img.shape[0]
        img_W=img.shape[1]
        Complete_patch_number_H=img_H//crop_size
        rest_patch_size_h=img_H-Complete_patch_number_H*crop_size
        Complete_patch_number_W=img_W//crop_size
        rest_patch_size_w=img_W-Complete_patch_number_W*crop_size
        padding_h=abs(crop_size-rest_patch_size_h)
        padding_w=abs(crop_size-rest_patch_size_w)
        img2=self.padding(img,padding_h,padding_w)
        assert img2.shape[0]%crop_size==0
        assert img2.shape[1]%crop_size==0
        img2_shape=img2.shape
        n=len(img2_shape)
        train_pathes=[]
        for i in range(Complete_patch_number_H+1):
            for j in range(Complete_patch_number_W+1):
                if n==3:
                    train_pathes.append(img2[i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size,:])
                if n==2:
                    train_pathes.append(img2[i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size])
        return train_pathes,Complete_patch_number_H+1,Complete_patch_number_W+1

    def write_tfrecord(self,img1,img2,gt,output_path,writer,img_id,patch_id):
        gtt=self.make_one_hot(gt)
        img2=img2.astype(np.float32).tostring()
        img1=img1.astype(np.float32).tostring()
        gt2=gtt.astype(np.float32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_index':tf.train.Feature(int64_list=tf.train.Int64List(value=[img_id])),
            'patch_index':tf.train.Feature(int64_list=tf.train.Int64List(value=[patch_id])),
            'img1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img1])),
            'img2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img2])),
            'gt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt2]))
        }
        ))
        writer.write(example.SerializeToString())
        print("generate tfrecord")
        
    def _parse_record(self,tfrecord):
        features = {
        'img_index': tf.FixedLenFeature([], tf.int64),
        'patch_index': tf.FixedLenFeature([], tf.int64),
        'img1': tf.FixedLenFeature((), tf.string),
        'img2': tf.FixedLenFeature((), tf.string),
        'gt': tf.FixedLenFeature((), tf.string)
        }
        parsed_features = tf.parse_single_example(tfrecord, features=features)
        img1=parsed_features['img1']
        img2=parsed_features['img2']
        gt=parsed_features['gt']
        img_id=parsed_features['img_index']
        patch_id=parsed_features['patch_index']
        img1 = tf.decode_raw(img1, tf.float32)
        img2 = tf.decode_raw(img2, tf.float32)
        gt = tf.decode_raw(gt, tf.float32) 
        shape1 = [self.patch_size,self.patch_size, 3]
        shape2 = [self.patch_size,self.patch_size,2]
        img1 = tf.reshape(img1, shape1)
        img2 = tf.reshape(img2, shape1)
        gt = tf.reshape(gt, shape2)
        return img1,img2,gt,img_id,patch_id
    
    def _parse_record2(self,tfrecord):
        features = {
        'img_index': tf.FixedLenFeature([], tf.int64),
        'patch_index': tf.FixedLenFeature([], tf.int64),
        'img1': tf.FixedLenFeature((), tf.string),
        'img2': tf.FixedLenFeature((), tf.string),
        'gt': tf.FixedLenFeature((), tf.string)
        }
        parsed_features = tf.parse_single_example(tfrecord, features=features)
        img1=parsed_features['img1']
        img2=parsed_features['img2']
        gt=parsed_features['gt']
        img_id=parsed_features['img_index']
        patch_id=parsed_features['patch_index']
        img1 = tf.decode_raw(img1, tf.float32)
        img2 = tf.decode_raw(img2, tf.float32)
        gt = tf.decode_raw(gt, tf.float32)
        shape1 = [256,256, 3]
        shape2 = [256,256,2]
        img1 = tf.reshape(img1, shape1)
        img2 = tf.reshape(img2, shape1)
        gt = tf.reshape(gt, shape2)
        return img1,img2,gt,img_id,patch_id

if __name__ == '__main__': 
    dataloader=Data_CDD()
    output_path='CDD256.tfrecord'
    all_the_patches_num=0
    if  not os.path.exists(output_path):
        writer = tf.python_io.TFRecordWriter(output_path)
        for i in range(dataloader.train_num):
            dataloader.write_tfrecord(dataloader.img1[i],dataloader.img2[i],dataloader.gt[i],output_path,writer,i,i)
            all_the_patches_num=all_the_patches_num+1
            print("finish",i)
        writer.close()
    print("total patches is %d"%(all_the_patches_num))
    dataset = tf.data.TFRecordDataset(output_path)
    dataset = dataset.map(dataloader._parse_record)
    dataset = dataset.batch(5)
    dataset = dataset.shuffle(buffer_size=20)
    iterator = dataset.make_one_shot_iterator()
    img1tf,img2tf,gttff,img_id,patch_id= iterator.get_next()
    
    with tf.Session() as sess:
        for i in range(10):
                img1_,img2_,gt_ = sess.run([img1tf,img2tf,gttff])
                img_id_ ,patch_id_= sess.run([img_id,patch_id])

                plt.figure()
                plt.subplot(131)
                plt.axis('off') 
                plt.title('img1',fontsize='medium',fontweight='bold')
                plt.imshow(img1_[0,:,:,0])
                
                plt.subplot(132)
                plt.axis('off') 
                plt.title('img2',fontsize='medium',fontweight='bold')
                plt.imshow(img2_[0,:,:,0])
                
                plt.subplot(133)
                plt.axis('off') 
                plt.title('gt',fontsize='medium',fontweight='bold')
                plt.imshow(gt_[0,:,:,0])
                plt.show()
                plt.close()

                print("shape of image")
                print(img1_[0,:,:,0].shape)