import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)
    datas = np.array(datas)
    labels = np.array(labels)
    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels