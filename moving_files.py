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

class Data_CCD():
    def __init__(self):
        self.path='./ChangeDetectionDataset'
        self.save_path='./CDD_TEST/'

        self.img1_path=[]
        self.img2_path=[]
        self.gt_path=[]

        self.test_img1_path=[]
        self.test_img2_path=[]
        self.test_gt_path=[]
        
        self.get_image_list()
        self.train_num=len(self.gt_path)
        self.test_num=len(self.test_gt_path)
        print("finish list")
        print(self.train_num)
        print( self.test_num)
        
        self.change_image_from_jpg2bmp()

    def get_image_list(self):
            for file in os.listdir(self.path):
                if file=="Real":
                    for file2 in os.listdir(self.path+"/"+file):
                        if file2=="subset":
                            for file3 in os.listdir(self.path+"/"+file+"/"+file2):
                                if file3=="train":
                                    for file4 in os.listdir(self.path+"/"+file+"/"+file2+"/"+file3):
                                        if file4=="A":
                                            for file5 in os.listdir(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4):
                                                #print(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4+"/"+file5)
                                                self.img1_path.append(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4+"/"+file5)
                                        if file4=="B":
                                            for file5 in os.listdir(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4):
                                                self.img2_path.append(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4+"/"+file5)
                                        if file4=="OUT":
                                            for file5 in os.listdir(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4):
                                                self.gt_path.append(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4+"/"+file5)
                                                
                                if file3=="test":
                                    for file4 in os.listdir(self.path+"/"+file+"/"+file2+"/"+file3):
                                        if file4=="A":
                                            for file5 in os.listdir(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4):
                                                self.test_img1_path.append(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4+"/"+file5)
                                        if file4=="B":
                                            for file5 in os.listdir(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4):
                                                self.test_img2_path.append(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4+"/"+file5)
                                        if file4=="OUT":
                                            for file5 in os.listdir(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4):
                                                self.test_gt_path.append(self.path+"/"+file+"/"+file2+"/"+file3 + "/"+file4+"/"+file5)
    
    def change_image_from_jpg2bmp(self):
        for i in range(self.test_num):
            os.makedirs(self.save_path+str(i))
            img1_t=cv2.imread(self.test_img1_path[i])
            img2_t=cv2.imread(self.test_img2_path[i])          
            gt_t= cv2.imread(self.test_gt_path[i])      
            path=self.save_path+str(i)+"/"
            cv2.imwrite(path+"im1.bmp",img1_t)
            cv2.imwrite(path+"im2.bmp",img2_t)
            cv2.imwrite(path+"gt.bmp",gt_t) 
            print("finish",i)
      
if __name__ == '__main__': 
    dataloader=Data_CCD()
    print("finish moving images&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")