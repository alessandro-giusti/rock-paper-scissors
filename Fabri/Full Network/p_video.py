import numpy as np
import cv2
import skimage
import skimage.viewer
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import keras
from keras import backend as K

# USER OPTIONS
model_name="rps.model"
class1="IMAGES/rock.jpg"
class2="IMAGES/paper.jpg"
class3="IMAGES/scissors.jpg"
legend=skimage.io.imread("IMAGES/rps_legend.bmp")
feed="video"
# VARIABLES
model = keras.models.load_model(model_name)
nc=model.output_shape[1]

title=skimage.io.imread("IMAGES/title.bmp")
x_axis=skimage.io.imread("IMAGES/x_axis.bmp")
neuron_nr=skimage.io.imread("IMAGES/neuron_nr.bmp")
output=[]*3
output.append(skimage.io.imread(class1))
output.append(skimage.io.imread(class2))
output.append(skimage.io.imread(class3))
get = K.function([model.layers[0].input], [model.layers[13].output])
get2=[]*12
Z=350
b2, F, G, H=0, 0, 0, 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

def adapt_input(im, size):
    h, w = im.shape[0:2]
    sz = min(h, w)
    im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:] 
    im = skimage.transform.resize(im, (size, size, 3), mode='reflect')
    return im
a_input="IMAGES/japan.jpg"
input2=skimage.img_as_ubyte(adapt_input(skimage.io.imread(a_input), 64))
input2b=skimage.img_as_ubyte(adapt_input(cv2.cvtColor(skimage.io.imread(a_input), cv2.COLOR_BGR2RGB), 80))

## STATIC IMAGE
cap = cv2.VideoCapture(0)
ret2, image2 = cap.read()

for i in range(12):
    get2.append(K.function([model.layers[0].input], [model.layers[i].output]))
frame2 = []
frame2.append(adapt_input(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB), 64))
images3=[]*11
for i in range(11):
    images3.append(skimage.img_as_ubyte(np.clip(np.asarray(get2[i]([frame2])[0][0]), -1, 1)))
#img, imgsmall = [0]*nc, [0]*nc

# SFONDO BIANCO
image2[:, :, 0:3].fill(255)
# TITLE
image2[8:8+title.shape[0],60:60+title.shape[1],0:3]=title
# LABEL1
image2[image2.shape[0]-40:image2.shape[0]-24,10:270,0:3]=x_axis
# LABEL2
image2[image2.shape[0]-20:image2.shape[0]-6,111:173,0:3]=neuron_nr
# LEGEND
W=240
image2[image2.shape[0]-30-legend.shape[0]:image2.shape[0]-30,image2.shape[1]-W-legend.shape[1]:image2.shape[1]-W,0:3]=cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
# INPUT
for j in range(3):
    A2 = 200+(64*3+2)//2-64-(64+1)*j
    image2[A2:A2+64, 10:74, 0:3].fill(0)
        
B=94
for j in range(11):
    C=images3[j].shape[2]
    D=images3[j].shape[0]
    for i in range(images3[j].shape[2]):
        A = 200+(D*C+C-1)//2-D-(D+1)*i
        image2[A:A+D, B:B+D, 0:3].fill(0)
    B=B+10+D

def colorMatrix(layer):
    A = np.zeros((5, 5))    
    for l in range(5):
        for k in range (5):
            for i in range (3):
                for j in range (3):
                    A[k][l] += model.layers[layer].get_weights()[0][i][j][k][l]
#                    if l==2:
#                        if k==2:
#                            print (model.layers[layer].get_weights()[0][i][j][k][l])
#                            if i==2:
#                                if j==2:
#                                    print (A[k][l])
    return A

W1=[]*16
for i in range(16):
    W1.append(model.layers[i].get_weights())

W8=np.asarray(W1[8])[0]
print("hello")
for i in range(5):
    print (W8[:,:,i,0])

W11=np.asarray(W1[12])[0]
#print(W11.shape)
W13=np.asarray(W1[15])[0]
W113=np.dot(W11,W13)
#print(W113)
W118=np.reshape(W113,(6, 6, 5, 3))
#for i in range(5):
#    print(W118[:, :, i])
#print(W113.shape)
W114=[0]*180
for i in range(180):
    W114[i]=np.argmax(W113[i,:])
W117=np.reshape(W114, (6, 6, 5))
#print(W114)
#for i in range(5):
#    print(W117[:, :, i])

W115=np.reshape(np.asarray(W114),(6, 6, 5))
W115=np.repeat(W115, 6, axis=0)
W115=np.repeat(W115, 6, axis=1)
#print(W115.shape)
W114=np.reshape(W114, (6,6,5))

A1 = np.round(colorMatrix(8), decimals=1)
print ("\nColor convolution 1 \n", A1)


T1=[]
T=np.zeros((6, 6, 5))
T[:, :, 3].fill(255)
T1.append(T)
get5 = K.function([model.layers[11].input], [model.layers[13].output])
layer_output = get5([T1])
layer_output = layer_output[0].tolist()
print(np.round(layer_output, decimals=1))

## MAIN LOOP
while(True):
    ret, image = cap.read()
    size=200
    A, B, C, D = int(image.shape[0]//2-size/2)-40, int(image.shape[0]//2+size/2)-40, int(image.shape[1]//2-size/2), int(image.shape[1]//2+size/2)
    frame = []
    if feed=="video":
        frame.append(adapt_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 64))
    else:
        frame.append(input2)
    a = model.predict(np.asarray(frame)).tolist()
    b = a[0].index(max(a[0]))
    f_v = get([frame])[0][0]
    images=[]*11
    for i in range(11):
        images.append(skimage.img_as_ubyte(np.clip(np.asarray(get2[i]([frame])[0][0]), -1, 1)))
    w = [[0]*128 for i in range(nc)]
    input_image=skimage.img_as_ubyte(adapt_input(image, 80))
    input_image2=skimage.img_as_ubyte(adapt_input(image, 64))
    image[:, :, :]=image2[:,:,:]
               
# INPUT
    if feed=="video":
        image[50:130, image.shape[1]-130:image.shape[1]-50]=input_image                
    else:
        image[50:130, image.shape[1]-130:image.shape[1]-50]=input2b                

# LAYERS
    for j in range(3):
        A2 = 200-(64*3+2)//2+(64+1)*j    
        if feed=="video":
            image[A2:A2+64, 10:74, (2-j)%3]=input_image2[:,:,j]
        else:
            for k in range(3):
                image[A2:A2+64, 10:74, k]=input2[:,:,j]

    B=94
    for j in range(11):
        C=images[j].shape[2]
        D=images[j].shape[0]
        for i in range(C):
            A = 200-(D*C+C-1)//2+(D+1)*i
            if b2!=b:
                image[A:A+D, B:B+D, 0:3].fill(0)
            if j<2:
                image[A:A+D, B:B+D, 1]=images[j][:, :, i]
            elif j==2 or j==3:
                image[A:A+D, B:B+D, 2]=images[j][:, :, i]
            elif j==5 or j==6:
                image[A:A+D, B:B+D, 0]=images[j][:, :, i]
#            elif j==4:
#                image[A:A+D, B:B+D, 0:3].fill(0)
#                for k in range(30):
#                    for l in range(30):
#                        image[A+k, B+l, (2+W115[k,l,i])%3]=255
#            elif j==8 or j==9:
#                image[A:A+D, B:B+D, 0]=images[j][:, :, i]
#                image[A:A+D, B:B+D, 1]=images[j][:, :, i]
            elif j==8 or j==9 or j==10:
#                for k in range(6):
#                    for l in range(6):
#                        image[A+k, B+l, (2+W114[k,l,i])%3]=images[j][k, l, i]
                if i==3:
                    image[A:A+D, B:B+D, 0]=images[j][:, :, i]
                elif i==0 or i==2:
                    image[A:A+D, B:B+D, 2]=images[j][:, :, i]
                else:
                    image[A:A+D, B:B+D, 1]=images[j][:, :, i]     
            else:
                for k in range(2):
                    image[A:A+D, B:B+D, k+1]=images[j][:, :, i]
        B=B+10+D
        
# PENULTIMATE LAYER
    for i in range(128):
        for j in range (nc):
            w[j][i]=model.layers[15].get_weights()[0][i][j]
        color=np.argmax(np.asarray(w)[:,i])
#        if color==1:
#            print ("numero:")
#            print (i)
        F, G, H=image.shape[0]-41-int(f_v[i]*50), image.shape[0]-40, 14+i*2
        image[F:G,H:H+2,0:3].fill(0)
        image[F:G,H:H+2,(color-1)%3].fill(255)

# OUTPUT
    image[image.shape[0]-130:image.shape[0]-50, image.shape[1]-130:image.shape[1]-50]=output[b]
    
#    T2=skimage.img_as_ubyte(np.reshape(np.asarray(get2[11]([frame]))[0][0], (6, 5, 6)))
#    print(T2)
    b2=b
    out.write(image)    
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# GRAPH2
#if nc==3:
#        cv2.line(image2, (Z, image2.shape[0]-60), (Z, image2.shape[0]-110), (0,0,255), 2)
#        cv2.line(image2, (Z, image2.shape[0]-60), (Z+50, image2.shape[0]-35), (255,0,0), 2)
#        cv2.line(image2, (Z, image2.shape[0]-60), (Z-50, image2.shape[0]-35), (0,127,0), 2)
#    cv2.rectangle(image,(image.shape[1]-150,10),image.shape[1]-50,100,(0,255,0),3)
### BORDO
###    image[50:150,image.shape[1]-150:image.shape[1]-149,0:3]=np.asarray([[[10]*3 for i in range(1)]for j in range(100)])
###    image[A-2:A,0:image.shape[1],0:3]=np.asarray([[[10]*3 for i in range(image.shape[1])]for j in range(2)])
###    image[A:B,D-1:D,0:3]=np.asarray([[[10]*3 for i in range(1)]for j in range(B-A)])
###    image[B-2:B,0:image.shape[1],0:3]=np.asarray([[[10]*3 for i in range(image.shape[1])]for j in range(2)])
# GRAPH        
##        if f_v[i]>0.2 and nc>2:
##            y = int(image.shape[0]-60-w[0][i]*50+25*w[1][i]+25*w[2][i])
##            x = int(Z+5*(8.66*w[1][i]-8.66*w[2][i]))
##            if color==0:
##                cv2.circle(image,(x, y), 3, (0,0,255), -1)
##            elif color==1:
##                cv2.circle(image,(x, y), 3, (255,0,0), -1)
##            else:
##                cv2.circle(image,(x, y), 3, (0,127,0), -1)
# OUTPUT

