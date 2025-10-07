#完成手写数字识别的任务
#导入必要的库
import numpy as np
import pandas as pd
#数据集来源
from keras.datasets import mnist
#加载数据集
(x_train_image,y_train_label),(x_test_image,y_test_label) = mnist.load_data()
#进行数据预处理
from tensorflow.keras.utils import to_categorical
x_train = x_train_image.reshape((60000, 28, 28, 1))
x_test = x_test_image.reshape((10000, 28, 28, 1))
y_train = to_categorical(y_train_label, 10)
y_test = to_categorical(y_test_label, 10)
#创建神经网络并开始训练
#构建一个比较简单的卷积神经网络
from keras import models
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
model = models.Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
#编译上述构建好的神经网络模型
#制定优化器为rmsprop
#制定损失函数为交叉熵损失
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#开始训练
model.fit(x_train,y_train,validation_split=0.3,epochs=5,batch_size=128)
#在测试集进行模型评估
score=model.evaluate(x_test,y_test)
#打印测试集的预测准确率
print('测试集预测准确率：',score[1])

#验证第一张图像
pred=model.predict(x_test[0].reshape(1,28,28,1))
#把SoftMax分类器输出转化为数字
print(pred[0],'转化一下格式得到: ',pred.argmax())
#导入绘图工具包
import matplotlib.pyplot as plt
#输出这张图像
plt.imshow(x_test[0].reshape(28,28), cmap='Greys')
plt.savefig('./test1.jpg')
plt.show()