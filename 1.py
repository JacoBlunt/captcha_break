# from captcha.image import ImageCaptcha
#import matplotlib.pyplot as plt
import numpy as np
# import random
import string
characters = string.digits + string.ascii_uppercase+string.ascii_lowercase
# print(characters)

import os 
def getSubDirs(rootDir): 
    return [ rootDir+d for d in os.listdir(rootDir)]

width, height,batchSize, n_len, n_class = 200, 50,500, 5,len(characters)
pic_pre_path='./pic_src/'
dirList = getSubDirs(pic_pre_path)
folderIndex = 0
# for x in dirList:
#     print(x)
# import matplotlib.image as mpimg
from PIL import Image

# def readPicFromHDD(os_path):
#     for i,lists in enumerate(os.listdir(os_path)): 
#         path = os.path.join(os_path, lists) 
#         fileNameList[i]=lists[:(len(lists)-4)]
#         imageList[i] = Image.open(path)
#     return imageList,fileNameList



def gen():
    global folderIndex
    batch_size=batchSize
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    fileNameList=["" for i in range(batch_size)]
    # generator = ImageCaptcha(width=width, height=height)
    while folderIndex<len(dirList):
        os_path=dirList[folderIndex]
        for i,lists in enumerate(os.listdir(os_path)): 
            if(i>=batch_size):
                break
            path = os_path+"/"+lists 
            fileNameList[i]=lists[:(len(lists)-4)]
            # print(fileNameList[i]+":"+path)
            X[i] = Image.open(path)
            random_str = fileNameList[i]
            # X[i] = imageList[i]
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        folderIndex=folderIndex+1
        yield X, y

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

X, y = next(gen())
# plt.imshow(X[0])
# plt.title(decode(y))

# from keras.utils.np_utils import to_categorical
from keras.models import *
from keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# from keras.utils.visualize_util import plot
# #from keras.utils.vis_utils import plot_model as plot
# from IPython.display import Image

# plot(model, to_file="model.png", show_shapes=True)
# Image('model.png')
try:
    model.fit_generator(gen(), samples_per_epoch=50000, nb_epoch=5,
                    validation_data=gen(), nb_val_samples=10)
except BaseException as e:
    print(e)

# X, y = next(gen())
# y_pred = model.predict(X)
# # plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
# # plt.imshow(X[0], cmap='gray')
# # plt.axis('off')

# from tqdm import tqdm
# def evaluate(model, batch_num=20):
#     batch_acc = 0
#     generator = gen()
#     for i in tqdm(range(batch_num)):
#         X, y = generator.next()
#         y_pred = model.predict(X)
#         batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
#     return batch_acc / batch_num

# evaluate(model)


# model.save('cnn.h5')
