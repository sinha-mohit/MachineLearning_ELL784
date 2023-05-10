import PIL
from PIL import Image
import numpy as np
import os
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt

path = '/your path/'
files = os.listdir(path)

[X,y] = read_images(path)

def read_images(path, sz=None):
    c = 0
    X,y = [], []
    
    for dirname , dirnames , filenames in os.walk(path):

        for subdirname in dirnames:
            print(subdirname)
            
            subject_path = os.path.join(dirname , subdirname)
            
            for filename in os.listdir(subject_path):
                
                im = Image.open(os.path.join(subject_path , filename))
                im = im.convert("L")
                image_numpy = np.asarray(np.asarray(im, dtype=np.uint8))
                X.append(image_numpy)
                y.append(c)
                
            c = c+1
            
    return [X,y]