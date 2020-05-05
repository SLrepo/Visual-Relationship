import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os.path as osp
import sys
import cv2
import math

obj = ['person',
       'sky',
       'building',
       'truck',
       'bus',
       'table',
       'shirt',
       'chair',
       'car',
       'train',
       'glasses',
       'tree',
       'boat',
       'hat',
       'trees',
       'grass',
       'pants',
       'road',
       'motorcycle',
       'jacket',
       'monitor',
       'wheel',
       'umbrella',
       'plate',
       'bike',
       'clock',
       'bag',
       'shoe',
       'laptop',
       'desk',
       'cabinet',
       'counter',
       'bench',
       'shoes',
       'tower',
       'bottle',
       'helmet',
       'stove',
       'lamp',
       'coat',
       'bed',
       'dog',
       'mountain',
       'horse',
       'plane',
       'roof',
       'skateboard',
       'traffic light',
       'bush',
       'phone',
       'airplane',
       'sofa',
       'cup',
       'sink',
       'shelf',
       'box',
       'van',
       'hand',
       'shorts',
       'post',
       'jeans',
       'cat',
       'sunglasses',
       'bowl',
       'computer',
       'pillow',
       'pizza',
       'basket',
       'elephant',
       'kite',
       'sand',
       'keyboard',
       'plant',
       'can',
       'vase',
       'refrigerator',
       'cart',
       'skis',
       'pot',
       'surfboard',
       'paper',
       'mouse',
       'trash can',
       'cone',
       'camera',
       'ball',
       'bear',
       'giraffe',
       'tie',
       'luggage',
       'faucet',
       'hydrant',
       'snowboard',
       'oven',
       'engine',
       'watch',
       'face',
       'street',
       'ramp',
       'suitcase']

rel = ['on',
       'wear',
       'has',
       'next to',
       'sleep next to',
       'sit next to',
       'stand next to',
       'park next',
       'walk next to',
       'above',
       'behind',
       'stand behind',
       'sit behind',
       'park behind',
       'in the front of',
       'under',
       'stand under',
       'sit under',
       'near',
       'walk to',
       'walk',
       'walk past',
       'in',
       'below',
       'beside',
       'walk beside',
       'over',
       'hold',
       'by',
       'beneath',
       'with',
       'on the top of',
       'on the left of',
       'on the right of',
       'sit on',
       'ride',
       'carry',
       'look',
       'stand on',
       'use',
       'at',
       'attach to',
       'cover',
       'touch',
       'watch',
       'against',
       'inside',
       'adjacent to',
       'across',
       'contain',
       'drive',
       'drive on',
       'taller than',
       'eat',
       'park on',
       'lying on',
       'pull',
       'talk',
       'lean on',
       'fly',
       'face',
       'play with',
       'sleep on',
       'outside of',
       'rest on',
       'follow',
       'hit',
       'feed',
       'kick',
       'skate on']


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def getAppr(im, bb):
    subim = im[bb[1]: bb[3], bb[0]: bb[2], :]
    subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0],
                       interpolation=cv2.INTER_LINEAR)
    pixel_means = np.array([[[103.939, 116.779, 123.68]]])
    subim -= pixel_means
    subim = subim.transpose((2, 0, 1))
    return subim


def getDualMask(ih, iw, bb):
    rh = 32.0 / ih
    rw = 32.0 / iw
    x1 = max(0, int(math.floor(bb[0] * rw)))
    x2 = min(32, int(math.ceil(bb[2] * rw)))
    y1 = max(0, int(math.floor(bb[1] * rh)))
    y2 = min(32, int(math.ceil(bb[3] * rh)))
    mask = np.zeros((32, 32))
    mask[y1: y2, x1: x2] = 1
    assert (mask.sum() == (y2 - y1) * (x2 - x1))
    return mask


this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', '..', '..', 'caffes', 'caffe-newest', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = "/home/sihuili01/CVproject/drnet_cvpr2017/prototxts/test_drnet_8units_relu_shareweight.prototxt"
PRETRAINED = "/home/sihuili01/CVproject/drnet_cvpr2017/snapshots/drnet_8units_relu_shareweight.caffemodel"

# load the model
# caffe.set_mode_gpu()
# caffe.set_device(0)
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# test on a image
# IMAGE_FILE = "/home/sihuili01/dataset/vrd/test/3845770407_1a8cd41230_b.jpg"
IMAGE_FILE = "test1.png"
# IMAGE_FILE = "/home/sihuili01/dataset/vrd/test/19558736_62d4cc76b8_o.jpg"
input_image = caffe.io.load_image(IMAGE_FILE)
im = cv2.imread(IMAGE_FILE).astype(np.float32, copy=False)
height = im.shape[0]
width = im.shape[1]
ims = []
bb = [0, 0, height, width]
ims.append(getAppr(im, bb))
ims = np.array(ims)
# predict takes any number of images,
# and formats them for the Caffe net automatically
forward_args = {}
qas = []
qbs = []
qa = [0.0] * 100
qa[13] = 1.0
qb = [0.0] * 100
qb[5] = 1.0
qas.append(qa)
qbs.append(qb)
qas = np.array(qas)
qbs = np.array(qbs)
poses = []
b_bb = [49, 261, 484, 630]
a_bb = [246, 295, 290, 342]
rMask = np.array([getDualMask(height, width, a_bb), getDualMask(height, width, b_bb)])
poses.append(rMask)
poses = np.array(poses)
net.blobs["im"].reshape(*(ims.shape))
forward_args["im"] = ims.astype(np.float32, copy=False)
net.blobs["qa"].reshape(*(qas.shape))
forward_args["qa"] = qas.astype(np.float32, copy=False)
net.blobs["qb"].reshape(*(qbs.shape))
forward_args["qb"] = qbs.astype(np.float32, copy=False)
net.blobs["posdata"].reshape(*(poses.shape))
forward_args["posdata"] = poses.astype(np.float32, copy=False)
# forward_args["im"] = ims.astype(np.float32, copy=False)
net_out = net.forward(**forward_args)
itr_pred = net_out["pred"].copy()
idx = np.argmax(itr_pred[0], axis=0)
print("The predicate is, ", rel[idx])
