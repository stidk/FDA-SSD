import os, math
# from PIL import Image as IMG
# import numpy.random as RAN
import random
import numpy as np
from imageio import imread

def avr(im_id):
    pth = 'dataset/dataset/NWPUVHR-10/JPG/train/%s'%im_id
    im = imread(pth)
    n = im.shape[0] * im.shape[1]

    avr = np.sum(np.sum(im,0),0)/n
    return avr


def std2(im_id, avr):
    pth = 'dataset/dataset/NWPUVHR-10/JPG/train/%s'%im_id
    im = imread(pth)
    n = im.shape[0] * im.shape[1]

    minus = np.subtract(im, avr)
    minus2 = np.power(minus,2)
    std2 = np.sum( np.sum(minus2,0),0 )/n
    return std2

N = 494
AVR = [0,0,0]
STD2 = [0,0,0]
list_im = []
list_i = os.listdir('dataset/dataset/NWPUVHR-10/JPG/train/')
for i in list_i:
    if(i[-1] = 't'):
        list_im.append(i)
print(list_im)
list_im_ = random.sample(list_im,N)
# ran = RAN.randint(0,len(list_im),N)
# list_im_ = [list_im[i] for i in ran]
count = 0
for i in list_im_:
    count=count+1
    avr__ = avr(i)
    AVR = [avr__[i]+AVR[i] for i in range(3)]
    print('avr: count {} / 494'.format(count))
AVR = [AVR[i]/N for i in range(3)]

count = 0
for i in list_im_:
    count=count+1
    std2__ = std2(i,AVR)
    STD2 = [std2__[i]+STD2[i] for i in range(3)]
    print('std2: count {} / 494'.format(count))
STD2 = [STD2[i]/N for i in range(3)]
STD = [math.sqrt(STD2[i]) for i in range(3)]
print( 'AVR:\n', AVR, '\nSTD:\n', STD )
