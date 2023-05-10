import match
import numpy as np

def back_gnd(img2,val,vec):
    img3=img2.copy()
    for i in range(val):
        x = img2[i]
        g,vec = match.synced(x,i,vec)
        if g == -5:
            x = 255  # WHITE
            img2[i] = vec[i][0][np.argmax(np.divide(vec[i][2],vec[i][1]))]
        else:
            x = 0  # BLACK
        img3[i]= x

    return img3


