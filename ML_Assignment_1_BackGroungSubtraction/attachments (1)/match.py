
import numpy as np
import math


K = 5  # NO. OF GAUSSIANS
B = 2
neww = 0.01
alpha = 0.09  # LEARNING RATE
newsig = 6  # parametres of a new gaussian

def synced(x,i,vec):
    u = vec[i][0]
    sig = vec[i][1]  # para. of a gaussian(u,sig,w)
    w = vec[i][2]

    mtc_gauss = -5  # some -ve value if no gaussian is matched

    for z in range(K):
        if (np.absolute(np.subtract(u[z], x))) < sig[z] * 2.5:
            mtc_gauss = z  # matched gaussian is updated to z
            break

    #w,sig,u=update.updatevar(x, w, sig, mtc_gauss,u,K)  # UPDATING THE PARAMETRE
    if (mtc_gauss!=-5):
        var = sig[mtc_gauss] ** 2  # sigma^2
        beta =alpha * ((math.exp(-(float(x) - float(u[mtc_gauss])) ** 2 / (2 * var))) / ((2 * np.pi*var) ** .5))

        u[mtc_gauss] = (1 - beta) * u[mtc_gauss] + beta * x  # updation of the parametres
        sig[mtc_gauss] = np.sqrt((1 - beta) * var + beta * (x - u[mtc_gauss]) ** 2)
        for j in range(K):
            if j == mtc_gauss:  # if gaussian is matched the update the weight by one formula
                w[j] = (1 - alpha) * w[j] + alpha  # &&
            else:  # if not matched the by other formula
                w[j] = (1 - alpha) * w[j]


    else:
        # code for replacing one of the existing guassian model
        index_min = np.argmin(np.divide(w, sig))  # position of minimum value in the list of (weight/sigma)
        w[index_min] = neww
        u[index_min] = x
        sig[index_min] = newsig
        w = list(map(lambda x: x /np.sum(w), w))  # normalizing the list of weights by sum of the weights


    if mtc_gauss != -5:
        L = np.divide(vec[i][2], vec[i][1])  # (WEIGHT/SIGMA) CALUCULATION
        a = list(np.array(L).argpartition(-2)[-2:])
        #for i in range(B):
        if mtc_gauss is a[0] or mtc_gauss is a[1]:
            mtc_gauss = z
            

    return mtc_gauss,vec

