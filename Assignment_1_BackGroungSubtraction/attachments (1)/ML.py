import cv2
import background
import kmeans

inivar=[13,13,13,13,13]
CAP = cv2.VideoCapture('umcp.mpg')
ret,img=CAP.read()
a=img.shape
img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inimean,iniwts=kmeans.km(img_2,5)
row=int(a[0])
col=int(a[1])
val=row*col
print(inimean)
print(iniwts)
vec = [[inimean,inivar, iniwts] for x in range(val)]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
org_out = cv2.VideoWriter('orginal.avi',fourcc, 17,(col, row),0)  # output window with 25 fps
fg_out = cv2.VideoWriter('foregndvid.avi', fourcc, 17, (col, row),0)


i=0
while (CAP.isOpened()):
    ret,img1 = CAP.read()
    if not ret:
        break
    img_2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    print(i)
    i=i+1
    img2=img_2.reshape(-1)
    img_4=background.back_gnd(img2,val,vec)
    img4=img_4.reshape(row,col)
    cv2.imshow('BG substraction', img4)
    org_out.write(img_2)
    fg_out.write(img4)

    if (cv2.waitKey(1) == 27) & 0xff:
        break

CAP.release()
cv2.destroyAllWindows()
org_out.release()
fg_out.release()

