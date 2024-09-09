from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from keras.models import Model
from keras.utils import np_utils
import os
from PIL import Image
from sklearn.model_selection import train_test_split

"""
数据集获取
"""


# 读取文件夹中的所有图片
label_mapping = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14,
    'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19
}
def load_images_from_folder(folder_path,target_size=(28, 28)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 加载图像
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).resize(target_size)
            images.append(np.array(img))

            # 提取标签
            label = filename.split('_')[-1].split('.')[0]
            label=label_mapping[label]
            labels.append(label)

    return img_path,images, labels



# 将图像和标签划分为训练集和测试集
def split_data(images, labels, train_size=0.7, val_size=0.2, test_size=0.1):
    x_temp, x_test, y_temp, y_test = train_test_split(images, labels, test_size=test_size, random_state=10)
    val_ratio = val_size / (train_size + val_size)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=val_ratio, random_state=10)
    return np.array(x_train), np.array(x_val), np.array(x_test), np.array(y_train), np.array(y_val), np.array(y_test)

# 示例：读取文件夹中的图片，并将其标签设为文件名中的最后一个数字
folder_path = 'temp/rec_datasets/train' # 替换为你的文件夹路径

# 提取图片数据和标签
img_path,images, labels = load_images_from_folder(folder_path)

# 划分数据集（7:2:1比例）
x_train, x_val, x_test, y_train, y_val, y_test = split_data(images, labels, train_size=0.7, val_size=0.2, test_size=0.1)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# 原始图像的像素灰度值为0-255，为了提高模型的训练精度，通常将数值归一化映射到0-1。
# x_train = x_train / 255
# x_val = x_val / 255
# x_test = x_test / 255

# print("y_train :{}".format(y_train))
# print("y_test :{}".format(y_test))
# print("y_val : {}".format(y_val))
# 图像标签一共有10个类别即0-9，这里将其转化为独热编码（One-hot）向量
y_train = np_utils.to_categorical(y_train,21)
y_val = np_utils.to_categorical(y_val,21)
y_test = np_utils.to_categorical(y_test,21)

# print("y_train :{}".format(y_train))
# print(type(x_train))
# print(x_test.shape)
# print("x_test :{}".format(y_test))
# print("y_val : {}".format(y_val))

"""
定义LeNet-5网络模型
"""

def LeNet5():

    input_shape = Input(shape=(28, 28, 1))

    x = Conv2D(6, (5, 5), activation="relu", padding="same")(input_shape)
    x = MaxPooling2D((2, 2), 2)(x)
    x = Conv2D(16, (5, 5), activation="relu", padding='same')(x)
    x = MaxPooling2D((2, 2), 2)(x)

    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    x = Dense(21,activation='softmax')(x)
    model = Model(input_shape, x)

    return model

"""
编译网络并训练
"""
model = LeNet5()

# 编译网络（定义损失函数、优化器、评估指标）
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# 开始网络训练（定义训练数据与验证数据、定义训练代数，定义训练批大小）
# train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=1, verbose=2)
train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=200, batch_size=5, verbose=2)
# 模型保存
# model.save('lenet_mnist.h5')

# 定义训练过程可视化函数（训练集损失、验证集损失、训练集精度、验证集精度）
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

# show_train_history(train_history, 'accuracy', 'val_accuracy')
# show_train_history(train_history, 'loss', 'val_loss')

# 输出网络在测试集上的损失与精度
# score = model.evaluate(x_test, y_test)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

#单个字符的识别
def process_images(directory):
    # 获取目录中的所有文件
    files = os.listdir(directory)
    # 过滤出图片文件（假设格式为jpg或png）
    image_files = [f for f in files if f.endswith(('.jpg', '.png'))]
    # 按照文件名排序（如果需要按其他顺序，请修改此部分）
    image_files.sort()
    j = 0
    # 逐个处理图片
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path).resize((28, 28))
        image = np.array(image).reshape(1, 28, 28, 1).astype('float32')

        # 进行预测
        prediction = model.predict(image)
        prediction = np.argmax(prediction, axis=1)[0]

        # 提取标签
        label = image_file.split('_')[-1].split('.')[0]
        label = label_mapping[label]

        # 检查预测结果
        if prediction != label:
            j += 1
            print("照片序号：{}, 预测结果 : {}, 标签结果：{}".format(image_file, prediction, label))

    print("预测错误数量 ：{}".format(j))
    print("单个字符预测准确：".format((5000-j)/5000))


# 调用函数并指定图片所在的目录
process_images('temp/rec_datasets/train')
# 整体字符识别
def process_images_in_batches(directory):
    # 获取目录中的所有文件
    files = os.listdir(directory)
    # 过滤出图片文件（假设格式为jpg或png）
    image_files = [f for f in files if f.endswith(('.jpg', '.png'))]

    # 按照文件名排序（如果需要按其他顺序，请修改此部分）
    image_files.sort()
    j=0
    # 处理每五个一组的图片
    for i in range(0, len(image_files), 5):
        batch = image_files[i:i + 5]
        images = []
        labels = []
        # 对每组图片进行处理
        for image_file in batch:
            image_path = os.path.join(directory, image_file)
            image = Image.open(image_path).resize((28, 28))
            images.append(np.array(image))

            # 如果需要保存处理后的图片，可以使用以下行
            # gray_image.save(f"processed_{image_file}")
            label = image_file.split('_')[-1].split('.')[0]
            label = label_mapping[label]
            labels.append(label)
        labels = np.array(labels)
        path=image_file.split('_')
        path="train_"+path[1]
        images=np.array(images)
        im = images.reshape(5, 28, 28, 1).astype('float32')
        predictions = model.predict(im)
        predictions = np.argmax(predictions, axis=1)
        # print("照片序号：{}".format(path))
        if (not np.array_equal(predictions, labels)):
            j=j+1
            print(j)
            print("照片序号：{},预测结果 : {},标签结果：{}".format(path,predictions, labels))
    print("预测错误数量 ：{}".format(j))
    print("整体预测准确：".format((1000 - j) / 1000))
# 调用函数并指定图片所在的目录
process_images_in_batches('temp/rec_datasets/train')





