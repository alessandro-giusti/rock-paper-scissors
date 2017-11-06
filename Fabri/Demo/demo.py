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
feed="video"

# VARIABLES
model = keras.models.load_model(model_name)
nc=model.output_shape[1]

title=skimage.io.imread("IMAGES/title.png")
x_axis=skimage.io.imread("IMAGES/x_axis.bmp")
neuron_nr=skimage.io.imread("IMAGES/neuron_nr.bmp")
output=[]*3
output.append(skimage.io.imread(class1))
output.append(skimage.io.imread(class2))
output.append(skimage.io.imread(class3))
get = K.function([model.layers[0].input], [model.layers[13].output])
get2=[]*11
b2, F, G, H=0, 0, 0, 0

def adapt_input(im, size):
    h, w = im.shape[0:2]
    sz = min(h, w)
    im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:] 
    im = skimage.transform.resize(im, (size, size, 3), mode='reflect')
    return im
a_input="IMAGES/striato.jpg"
input2=skimage.img_as_ubyte(adapt_input(skimage.io.imread(a_input), 64))
input2b=skimage.img_as_ubyte(adapt_input(cv2.cvtColor(skimage.io.imread(a_input), cv2.COLOR_BGR2RGB), 80))
b_input="IMAGES/japan.jpg"
c_input="IMAGES/test.jpg"
input3=skimage.img_as_ubyte(adapt_input(skimage.io.imread(b_input), 64))
input3b=skimage.img_as_ubyte(adapt_input(cv2.cvtColor(skimage.io.imread(b_input), cv2.COLOR_BGR2RGB), 80))
input4=skimage.img_as_ubyte(adapt_input(skimage.io.imread(c_input), 64))
input4b=skimage.img_as_ubyte(adapt_input(cv2.cvtColor(skimage.io.imread(c_input), cv2.COLOR_BGR2RGB), 80))

inputtino=input2
inputtinob=input2b
conc=0

## STATIC IMAGE
cap = cv2.VideoCapture(1)
ret2, image2 = cap.read()

Z=image2.shape[1]-215
center=255
if conc==0:
    length=138
for i in range(11):
    get2.append(K.function([model.layers[0].input], [model.layers[i].output]))
frame2 = []
frame2.append(adapt_input(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB), 64))
images3=[]*11
for i in range(11):
    images3.append(skimage.img_as_ubyte(np.clip(np.asarray(get2[i]([frame2])[0][0]), -1, 1)))


# SFONDO BIANCO
image2[:, :, 0:3].fill(255)
image5=np.copy(image2)
for j in range(3):
    A2 = 240+(64*3+2)//2-64-(64+1)*j
    image5[A2:A2+64, 98:162, 0:3].fill(0)
        
B=182
for j in range(11):
    C=images3[j].shape[2]
    D=images3[j].shape[0]
    for i in range(images3[j].shape[2]):
        A = 240+(D*C+C-1)//2-D-(D+1)*i
        image5[A:A+D, B:B+D, 0:3].fill(0)
    B=B+10+D
# TITLE
image2[8:8+title.shape[0],60:60+title.shape[1],0:3]=title
# LABEL1
image2[image2.shape[0]-50:image2.shape[0]-34,36:296,0:3]=x_axis
# LABEL2
image2[image2.shape[0]-34:image2.shape[0]-20,168-31:168+31,0:3]=neuron_nr
# GRAPH2
cv2.circle(image2,(Z, center-1), length, (220,220,220), -1)
cv2.line(image2, (Z, center), (Z, center-length), (0,0,255), 2)
cv2.line(image2, (Z, center), (Z, center+length), (180,180,180), 1)
cv2.line(image2, (Z, center), (Z+int(0.866*length), center+int(0.866*length//2)), (255,0,0), 2)
cv2.line(image2, (Z, center), (Z-int(0.866*length), center-int(0.866*length//2)), (180,180,180), 1)
cv2.line(image2, (Z, center), (Z-int(0.866*length), center+int(0.866*length//2)), (0,127,0), 2)
cv2.line(image2, (Z, center), (Z+int(0.866*length), center-int(0.866*length//2)), (180,180,180), 1)
# LEGEND
W=240
image2[image2.shape[0]-50-legend.shape[0]:image2.shape[0]-50,image2.shape[1]-40-legend.shape[1]:image2.shape[1]-40,0:3]=cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)

def spegni_mappa_0(mappa):
    C = model.layers[0].get_weights()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[0][i][j][k][mappa-1]= 0
    model.layers[0].set_weights(C)
    
def spegni_mappa_1(mappa):
    C = model.layers[2].get_weights()
    for i in range(3):
        for j in range(3):
            for k in range(5):
                C[0][i][j][k][mappa-1]= 0
    model.layers[2].set_weights(C)


def spegni_mappa_2(mappa):
    C = model.layers[5].get_weights()
    for i in range(3):
        for j in range(3):
            for k in range(5):
                C[0][i][j][k][mappa-1]= 0
    model.layers[5].set_weights(C)


def spegni_mappa_3(mappa):
    C = model.layers[8].get_weights()
    for i in range(3):
        for j in range(3):
            for k in range(5):
                C[0][i][j][k][mappa-1]= 0
    model.layers[8].set_weights(C)


def spegni_mappe1():
    spegni_mappa_0(1)
    spegni_mappa_0(2)
    #spegni_mappa_0(3)
    spegni_mappa_0(4)
    spegni_mappa_0(5)

def spegni_mappe2():
    spegni_mappa_1(1)
    spegni_mappa_1(2)
    spegni_mappa_1(3)
#spegni_mappa_1(4)
    spegni_mappa_1(5)

def spegni_mappe3():
    spegni_mappa_2(1)
#spegni_mappa_2(2)
#spegni_mappa_2(3)
    spegni_mappa_2(4)
#    spegni_mappa_2(5)

def spegni_mappe4():
#    spegni_mappa_3(1)
#    spegni_mappa_3(2)
#    spegni_mappa_3(3)
#spegni_mappa_3(4)
    spegni_mappa_3(5)    


remap_std=0.3
def remap(m):
    return np.clip(((m-np.mean(m))/np.std(m)*remap_std+0.5)*255,0,255).astype("uint8")


## MAIN LOOP
while(True):
    ret, image = cap.read()
    size=200
    A, B, C, D = int(image.shape[0]//2-size/2)-40, int(image.shape[0]//2+size/2)-40, int(image.shape[1]//2-size/2), int(image.shape[1]//2+size/2)
    Z=image2.shape[1]-215
    center=255
    frame3 = []
    if feed=="video":
        frame3.append(adapt_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 64))
    else:
        frame3.append(inputtino)
#    frame3.append(adapt_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 64))
#    frame3.append(input2)
    a = model.predict(np.asarray(frame3)).tolist()
    b = a[0].index(max(a[0]))
    f_v = get([frame3])[0][0]
    images=[]*11
    for i in range(11):
#        images.append(skimage.img_as_ubyte(np.clip(get2[i]([frame3])[0][0], -1, 1)))
        images.append(remap(get2[i]([frame3])[0][0]))
    w = [[0]*128 for i in range(nc)]
    input_image=skimage.img_as_ubyte(adapt_input(image, 80))
    input_image3=skimage.img_as_ubyte(adapt_input(image, 120))
    input_image2=skimage.img_as_ubyte(adapt_input(image, 64))
    image[:, :, :]=image2[:,:,:]
               
# INPUT
    if conc==0:
        if feed=="video":
            image[50:170, 50:170]=cv2.flip(input_image3, 0)                
        else:
            image[50:130, 50:130]=inputtinob
    if conc==1:
        if feed=="video":
            image5[192:272, 10:90]=cv2.flip(input_image, 0)                
        else:
            image5[192:272, 10:90]=inputtinob

#    image[60:180, 60:180]= cv2.flip(input_image, 1)                
#    image[50:130, image.shape[1]-130:image.shape[1]-50]=input2b                

# LAYERS
    if conc==1:
        for j in range(3):
            A2 = 240-(64*3+2)//2+(64+1)*j    
            if feed=="video":
                image5[A2:A2+64, 98:162, (2-j)%3]=input_image2[:,:,j]
            elif feed=="vid":
                for k in range(3):
                    image5[A2:A2+64, 98:162, k]=inputtino[:,:,j]

        B=182
        for j in range(10):
            C=images[j].shape[2]
            D=images[j].shape[0]
            for i in range(C):
                A = 240-(D*C+C-1)//2+(D+1)*i
                if b2!=b:
                    image5[A:A+D, B:B+D, 0:3].fill(0)
                if j<2:
                    image5[A:A+D, B:B+D, 1]=images[j][:, :, i]
                elif j==2 or j==3:
                    image5[A:A+D, B:B+D, 2]=images[j][:, :, i]
                elif j==5 or j==6:
                    image5[A:A+D, B:B+D, 0]=images[j][:, :, i]
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
                        image5[A:A+D, B:B+D, 0]=images[j][:, :, i]
                    elif i==0 or i==2:
                        image5[A:A+D, B:B+D, 2]=images[j][:, :, i]
                    else:
                        image5[A:A+D, B:B+D, 1]=images[j][:, :, i]     
                else:
                    for k in range(2):
                        image5[A:A+D, B:B+D, k+1]=images[j][:, :, i]
            B=B+10+D

# PENULTIMATE LAYER
    for i in range(128):
        for j in range (nc):
            w[j][i]=model.layers[15].get_weights()[0][i][j]
        color=np.argmax(np.asarray(w)[:,i])
        F, G, H=image.shape[0]-50-min(int(f_v[i]*50), 298), image.shape[0]-51, 40+i*2
        image[F:G,H:H+2,0:3].fill(0)
        image[F:G,H:H+2,(color-1)%3].fill(255)

        y = -w[0][i]*length+length//2*w[1][i]+length//2*w[2][i]
        x = 10*(8.66*w[1][i]-8.66*w[2][i])
        cv2.line(image, (int(Z), int(center)), ((int(Z+ x*f_v[i]/12)), int(center+ y*f_v[i]/12)), (20,20,20), 2)
        Z=Z+int(x*f_v[i]/12)
        center=center+y*f_v[i]/12
    
# OUTPUT
    image[50:130, image.shape[1]-130:image.shape[1]-50]=output[b]
    
    b2=b
#    out.write(image)

    key = cv2.waitKey(20)
    if key & 0xFF == ord('0'):
        feed="video"
    if key & 0xFF == ord('1'):
        conc=1
    if key & 0xFF == ord('2'):
        feed, inputtino, inputtinob="vid", input2, input2b
    if key & 0xFF == ord('3'):
        feed, inputtino, inputtinob="vid", input3, input3b
    if key & 0xFF == ord('4'):
        feed, inputtino, inputtinob="vid", input4, input4b
    if key & 0xFF == ord('5'):
        spegni_mappe1()        
    if key & 0xFF == ord('6'):
        spegni_mappe2()
    if key & 0xFF == ord('7'):
        spegni_mappe3()
    if key & 0xFF == ord('8'):
        spegni_mappe4()
    if conc==1:
        image=np.concatenate((image5,image), axis=1)
    if key & 0xFF == ord('q'):
        break

    cv2.imshow('The Penultimate Layer',image)
    
cap.release()
cv2.destroyAllWindows()


