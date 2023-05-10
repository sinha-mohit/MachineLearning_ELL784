
def fore_gnd(i,j,img_4,gaus_no):

    if gaus_no == -1:
        img_4[i][j]=255                      #WHITE
    else:
        img_4[i][j]=0                        #BLACK
    return img_4
