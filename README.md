# Combined-Loss-Based-MFCN-for-Change-Detection<br>
<br>
Title: A Combined Loss-Based Multiscale Fully Convolutional Network for High-Resolution Remote Sensing Image Change Detection<br>
[paper](https://ieeexplore.ieee.org/abstract/document/9502172)<br>
[github](https://pages.github.com/)
<br>
<br>
**Introduction**<br>
In the task of change detection (CD), high-resolution remote sensing images (HRSIs) can provide rich ground object information. However, the interference from noise and complex background information can also bring some challenges to CD. In recent years, deep learning methods represented by convolutional neural networks (CNNs) have achieved good CD results. However, the existing methods have difficulty in detecting the detailed change information of the ground objects effectively. The imbalance of positive and negative samples can also seriously affect the CD results. In this letter, to solve the above problems, we propose a method based on a multiscale fully convolutional neural network (MFCN), which uses multiscale convolution kernels to extract the detailed features of the ground object features. A loss function combining weighted binary cross-entropy (WBCE) loss and dice coefficient loss is also proposed, so that the model can be trained from unbalanced samples. The proposed method was compared with six state-ofthe-art CD methods on the DigitalGlobe dataset. The experiments showed that the proposed method can achieve a higher F1-score,and the detection effect of the detailed changes was better thanthat of the other methods.
<br>
<br>
**Data Preparation**<br>
CDD dataser: you can dowmload from [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)<br>
<br>
<br>
**Usage**
Tensorflow<br>
run mian_CDD.py
