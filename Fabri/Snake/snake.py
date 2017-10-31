import numpy as np
import cv2
import skimage
import skimage.viewer
from skimage import img_as_ubyte
import keras
from keras import backend as K

# USER OPTIONS
model_name="rps.model"
class1="IMAGES/rock.jpg"
class2="IMAGES/paper.jpg"
class3="IMAGES/scissors.jpg"
legend=skimage.io.imread("IMAGES/rps_legend.bmp")

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
get2=[]*11
b2, F, G, H=0, 0, 0, 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

def adapt_input(im, size):
    h, w = im.shape[0:2]
    sz = min(h, w)
    im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:] 
    im = skimage.transform.resize(im, (size, size, 3), mode='reflect')
    return im
a_input="IMAGES/cred.jpg"
input2=skimage.img_as_ubyte(adapt_input(skimage.io.imread(a_input), 64))
input2b=skimage.img_as_ubyte(adapt_input(cv2.cvtColor(skimage.io.imread(a_input), cv2.COLOR_BGR2RGB), 80))

## STATIC IMAGE
cap = cv2.VideoCapture(0)
ret2, image2 = cap.read()

Z=image2.shape[1]//2
center=image2.shape[0]*2//3-40
length=150
for i in range(11):
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
image2[image2.shape[0]//3:image2.shape[0]//3+16,10:270,0:3]=x_axis
# LABEL2
image2[image2.shape[0]//3+16:image2.shape[0]//3+30,111:173,0:3]=neuron_nr
# LEGEND
W=240
image2[40:40+legend.shape[0],40:40+legend.shape[1],0:3]=cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
# GRAPH2
cv2.circle(image2,(Z, center-1), length, (240,240,240), -1)
cv2.line(image2, (Z, center), (Z, center-length), (0,0,255), 2)
cv2.line(image2, (Z, center), (Z, center+length), (220,220,220), 1)
cv2.line(image2, (Z, center), (Z+int(0.866*length), center+int(0.866*length//2)), (255,0,0), 2)
cv2.line(image2, (Z, center), (Z-int(0.866*length), center-int(0.866*length//2)), (220,220,220), 1)
cv2.line(image2, (Z, center), (Z-int(0.866*length), center+int(0.866*length//2)), (0,127,0), 2)
cv2.line(image2, (Z, center), (Z+int(0.866*length), center-int(0.866*length//2)), (220,220,220), 1)

#    cv2.rectangle(image,(image.shape[1]-150,10),image.shape[1]-50,100,(0,255,0),3)


## MAIN LOOP
while(True):
    ret, image = cap.read()
    size=200
    A, B, C, D = int(image.shape[0]//2-size/2)-40, int(image.shape[0]//2+size/2)-40, int(image.shape[1]//2-size/2), int(image.shape[1]//2+size/2)
    Z=image2.shape[1]//2
    center=image2.shape[0]*2//3-40
    frame = []
    frame.append(adapt_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 64))
#    frame.append(input2)
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
    image[image.shape[0]-120:image.shape[0]-40, 40:120]=input_image                
#    image[50:130, image.shape[1]-130:image.shape[1]-50]=input2b                
    Z1=Z
    center1=center
# PENULTIMATE LAYER
    for i in range(128):
        for j in range (nc):
            w[j][i]=model.layers[15].get_weights()[0][i][j]
        color=np.argmax(np.asarray(w)[:,i])
        F, G, H=image.shape[0]//3-int(f_v[i]*50), image.shape[0]//3-1, 14+i*2
        image[F:G,H:H+2,0:3].fill(0)
        image[F:G,H:H+2,(color-1)%3].fill(255)

# GRAPH        
##        if f_v[i]>0.2 and nc>2:
##            y = int(center-w[0][i]*length+length//2*w[1][i]+length//2*w[2][i])
##            x = int(Z+10*(8.66*w[1][i]-8.66*w[2][i]))
##            if color==0:
##                cv2.circle(image,(x, y), 3, (0,0,255), -1)
##            elif color==1:
##                cv2.circle(image,(x, y), 3, (255,0,0), -1)
##            else:
##                cv2.circle(image,(x, y), 3, (0,127,0), -1)
        
        y = int(-w[0][i]*length+length//2*w[1][i]+length//2*w[2][i])
        x = int(10*(8.66*w[1][i]-8.66*w[2][i]))
        cv2.line(image, (Z, center), ((Z+int(x*f_v[i]/12)), center+int(y*f_v[i]/12)), (200,200,200), 2)
        Z=Z+int(x*f_v[i]/12)
        center=center+int(y*f_v[i]/12)
##        if color==0:
##                cv2.circle(image,(x, y), 3, (0,0,255), -1)
##            elif color==1:
##                cv2.circle(image,(x, y), 3, (255,0,0), -1)
##            else:
##                cv2.circle(image,(x, y), 3, (0,127,0), -1)
    
# OUTPUT
    image[image.shape[0]-130:image.shape[0]-50, image.shape[1]-130:image.shape[1]-50]=output[b]
    
    b2=b
    out.write(image)    
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

### BORDO
###    image[50:150,image.shape[1]-150:image.shape[1]-149,0:3]=np.asarray([[[10]*3 for i in range(1)]for j in range(100)])
###    image[A-2:A,0:image.shape[1],0:3]=np.asarray([[[10]*3 for i in range(image.shape[1])]for j in range(2)])
###    image[A:B,D-1:D,0:3]=np.asarray([[[10]*3 for i in range(1)]for j in range(B-A)])
###    image[B-2:B,0:image.shape[1],0:3]=np.asarray([[[10]*3 for i in range(image.shape[1])]for j in range(2)])


