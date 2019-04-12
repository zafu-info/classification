#导入交叉验证库
from sklearn.model_selection import train_test_split#替换sklearn.cross_validation
#导入SVM分类算法库
from sklearn import svm
#生成预测结果准确率的混淆矩阵
from sklearn import metrics
import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
from keras.datasets import cifar10
import keras as ks
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Convolution2D
import ReadData as rd
#数据文件夹
data_dir = "./data_mc_allinone/"
fpaths, datas, labels = rd.read_data(data_dir)

Y=labels
n_samples = len(datas)
X = datas.reshape((n_samples, 160*160*3))
#随机抽取生成训练集和测试集，其中训练集的比例为60%，测试集40%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#cnn
train_img = X_train
val_img = X_test
#important
train_LBL = np_utils.to_categorical(y_train)
val_LBL = np_utils.to_categorical(y_test)

model = ks.models.Sequential()
model.add(Dense(128,input_dim=160*160*3))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit(x=train_img,y=train_LBL,batch_size=10,nb_epoch=100,verbose=1,validation_data=(val_img,val_LBL))

score = model.evaluate(val_img,val_LBL, verbose=0)
#print('Test score:', score[0])
my_confusion=metrics.confusion_matrix(val_LBL, y)
print('Test accuracy:.2f%%%', 100*score[1])
print(my_confusion)