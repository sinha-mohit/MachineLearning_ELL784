import cv2
import numpy as np
import math
import maths




K =5                                        #NO. OF GAUSSIANS
B =2                                        #NO. OF BACKGROUND GAUSSIAN

new_w =0.001
alpha=0.015                            #LEARNING RATE
new_sig=13                              #parametres of a new gaussian


VCAPTURE = cv2.VideoCapture('umcp.mpg')                 #cature new video
col = int(VCAPTURE.get(3))                              #row and  col
row = int(VCAPTURE.get(4))
if not VCAPTURE.isOpened():
    print ('File did not open')

bg_out = cv2.VideoWriter('DEMO1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (col,row))                          #output window with 30 fps
fg_out = cv2.VideoWriter('FGVideo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (col,row))
All_vect = [[[[50,100,150,125,200],[8,8,8,8,8],[.2,.2,.2,.2,.2]] for x in range(col)] for y in range(row)]            #mean, sigma and weights
print(col,row)



def matched(x,i,j):

    u = All_vect[i][j][0]
    sig = All_vect[i][j][1]                                    # para. of a gaussian(u,sig,w)
    w = All_vect[i][j][2]

    mtc_gauss = -1                                          #return -1 if no gaussian is matched
    mod_val = np.absolute(np.subtract(u,x))                 # TO GET ABSOLUTE DIFFERENCE FOR MATCHING
    
    for z in range(K):
        if mod_val[z] < sig[z] * 2.5 :
            mtc_gauss = z                                   # matched gaussian is updated to z

    
    update_var(x,w,sig,mtc_gauss,u)                          #UPDATING THE PARAMETRE
    
    if mtc_gauss != -1:

        List=np.divide(All_vect[i][j][2],All_vect[i][j][1])                   #(WEIGHT/SIGMA) CALUCULATION
        MAX_UP=77000                                                           #value change kar di hai maine
        for x1 in range(B):
            max = -1
            for x2 in range(len(np.divide(All_vect[i][j][2],All_vect[i][j][1]))):
                if List[x2]>max and List[x2]<MAX_UP:
                    max = List[x2]
                    index=x2
            if mtc_gauss is index:
                mtc_gauss=z
            MAX_UP = max

    return mtc_gauss

#................ UPDATING THE PARAMETER......................................

def update_var(x,w,sig,mtc_gauss,u):


    if (mtc_gauss is -1) :
        # code for replacing one of the existing guassian model
        index_min = np.argmin(np.divide(w,sig))             #position of minimum value in the list of (weight/sigma)
        w[index_min] = new_w
        u[index_min] = x
        sig[index_min] = new_sig
        w = list(map(lambda x: x/np.sum(w), w))             #normalizing the list of weights by sum of the weights
    else:
        var = sig[mtc_gauss]**2                             #sigma^2
        deno = (2*np.pi*var)**.5
        num = math.exp(-(float(x)-float(u[mtc_gauss]))**2/(2*var))
        gaussian = num/deno
        rho = alpha*gaussian                                           #SECOND LEARNING FACTOR
        u[mtc_gauss] = (1-rho)*u[mtc_gauss] + rho*x               #updation of the parametres
        sig[mtc_gauss] = np.sqrt((1-rho)*var + rho*(x-u[mtc_gauss])**2)
        for z in range(K):
            if z==mtc_gauss:                                      #if gaussian is matched the update the weight by one formula
                w[z] = (1-alpha)*w[z] + alpha                     #                 &&
            else:                                                 # if not matched the by other formula
                w[z] = (1-alpha)*w[z]
 
#................ FOREGROUND .......................................

# def fore_gnd(i,j,img_4,gaus_no):
#
#     if gaus_no == -1:
#         img_4[i][j]=255                      #WHITE
#     else:
#         img_4[i][j]=0                        #BLACK
#     return img_4
#.................BACKGROUND ........................................

def back_gnd (img_2):

    img3 = img_2.copy()
    for i in range(row):
        for j in range(col):
            x = img_2[i,j]
            g = matched(x,i,j)
            img3=maths.fore_gnd(i,j,img3,g)
            if (g is -1) :
                img_2[i,j] =All_vect[i][j][0][np.argmax(np.divide(All_vect[i][j][2],All_vect[i][j][1]))]        #the maximum value

    return img_2, img3


#.....................................................

while(VCAPTURE.isOpened()):
    ret, img_1 = VCAPTURE.read()
    if not ret:
        break
    img_2 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)

    img_3,img_4 = back_gnd(img_2)


    cv2.imshow('This is Grayscale', img_3)
    bg_out.write(cv2.cvtColor(img_2,cv2.COLOR_GRAY2BGR))
    cv2.imshow('This is BG substraction',img_4)
    bg_out.write(img_3)
    fg_out.write(img_4)

    if(cv2.waitKey(1) == 27) & 0xff:
        break

VCAPTURE.release()
cv2.destroyAllWindows()
bg_out.release()
fg_out.release()

