import numpy as np
import cv2
import skimage
import skimage.viewer
from skimage import img_as_ubyte
import keras
from keras import backend as K
import matplotlib.pyplot as plt

# USER OPTIONS
model_name="rps.model"
class1="IMAGES/rock.jpg"
class2="IMAGES/paper.jpg"
class3="IMAGES/scissors.jpg"
legend=skimage.io.imread("IMAGES/rps_legend.bmp")

# VARIABLES
model = keras.models.load_model(model_name)
nc=model.output_shape[1]
neuron_i=0
video=True
title=skimage.io.imread("IMAGES/title.png")
x_axis=skimage.io.imread("IMAGES/x_axis.bmp")
neuron_nr=skimage.io.imread("IMAGES/neuron_nr.bmp")
special_input, special_input2, special_input3, special_input4=[],[],[],[]
special_input.append("IMAGES/japan2.jpg")
special_input.append("IMAGES/striato2.jpg")
special_input.append("IMAGES/test2.jpg")
output=[]*3
output.append(skimage.io.imread(class1))
output.append(skimage.io.imread(class2))
output.append(skimage.io.imread(class3))
conc=0
paperino=0
l_verme=100
b2, F, G, H=0, 0, 0, 0
def adapt_input(im, size):
    h, w = im.shape[0:2]
    sz = min(h, w)
    im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:] 
    im = skimage.transform.resize(im, (size, size, 3), mode='reflect')
    return im

for i in range(len(special_input)):
    special_input2.append(adapt_input(skimage.io.imread(special_input[i]), 64))
    special_input3.append(skimage.img_as_ubyte(adapt_input(skimage.io.imread(special_input[i])[:,:,::-1], 80)))
    special_input4.append(skimage.img_as_ubyte(adapt_input(skimage.io.imread(special_input[i])[:,:,::-1], 120)))
inputtino, inputtinob, inputtinod=special_input2[0], special_input3[0], special_input4[0]
neuron, neuronb=[],[]
for i in range(128):
    neuron.append(adapt_input(skimage.io.imread("IMAGES/NEURONS/neuron%d.jpg" %(i+1)), 64))
    neuronb.append(skimage.img_as_ubyte(adapt_input(skimage.io.imread("IMAGES/NEURONS/neuron%d.jpg"%(i+1))[:,:,::-1], 80)))
for i in range(3):
    neuron.append(adapt_input(skimage.io.imread("IMAGES/NEURONS/output%d.jpg" %(i+1)), 64))
    neuronb.append(skimage.img_as_ubyte(adapt_input(skimage.io.imread("IMAGES/NEURONS/output%d.jpg"%(i+1))[:,:,::-1], 80)))
    
get = K.function([model.layers[0].input], [model.layers[13].output])
get2=[]*11
for i in range(11):
    get2.append(K.function([model.layers[0].input], [model.layers[i].output]))

index=[0, 2, 5, 8]

def spegni_mappa(l, m):
    C = model.layers[index[l-1]].get_weights()
    for n in range(len(m)):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[0][i][j][k][m[n]-1]= 0
    model.layers[index[l-1]].set_weights(C)

def spegni_mappe1():
    l1=1
    m1=[1, 2, 4, 5]
    spegni_mappa(l1, m1)

def spegni_mappe2():
    l2=2
    m2=[1, 2, 3, 5]
    spegni_mappa(l2, m2)

def spegni_mappe3():
    l3=3
    m3=[1, 4]
    spegni_mappa(l3, m3)

def spegni_mappe4():
    l4=4
    m4=[5]
    spegni_mappa(l4, m4)

def generate_heatmap(input1):
    input1=input1[np.newaxis,:,:,:]
    a = model.predict([input1])
    b = np.argmax(a[0])
    print(b)
    label= model.output[:,b]
    lastconv=model.get_layer("conv2d_12")

    grads=K.gradients(label, lastconv.output)[0]
    pool_g=K.mean(grads, axis=(0, 1, 2))
    get=K.function([model.input, K.learning_phase()], [pool_g, lastconv.output[0]])
    pool_g_value, conv_value=get([input1, 1])
    for i in range(5):
        conv_value[:,:,i] *=pool_g_value[i]
    heatmap= np.mean(conv_value, axis=-1)

    heatmap=np.maximum(heatmap, 0)
    heatmap/=np.max(heatmap)
    heatmap=cv2.resize(heatmap, (input1.shape[1], input1.shape[0]))
    heatmap=np.uint8(255*heatmap)
    heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_image=np.uint8(input1*255)*0.5+heatmap
    superimposed_image=adapt_input(superimposed_image[0], 120)
    return superimposed_image 

#remap_std=0.3
def remap(m, max_pix, min_pix):
    return np.clip((m/max_pix*192)+64, 0, 255).astype("uint8")    
#    return np.clip(((m-np.mean(m))/np.std(m)*remap_std+0.5)*255,0,255).astype("uint8")

## STATIC IMAGE
cap = cv2.VideoCapture(0)
ret2, image2 = cap.read()

Z=image2.shape[1]-215
center=255
length=138

a=[0]*11
for i in range(11):
    a[i]=model.layers[i].output_shape
    
# SFONDO BIANCO
image2[:, :, 0:3].fill(255)
image5=np.copy(image2)
# SFONDO LAYER
for j in range(3):
    A2 = 240+(64*3+2)//2-64-(64+1)*j
    image5[A2:A2+64, 98:162, 0:3].fill(0)
        
B=182
for j in range(11):
    C=a[j][3]
    D=a[j][1]
    for i in range(C):
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



## MAIN LOOP
while(True):
    ret, image = cap.read()
    Z=image2.shape[1]-215
    center=255
    max_pix, min_pix=0, 0

    if video==True:
        input_nn=adapt_input(image[:,:,::-1], 64)
    else:
        input_nn=inputtino
    a = model.predict([input_nn[np.newaxis,:,:,:]]).tolist()
    b = a[0].index(max(a[0]))
    f_v = get([input_nn[np.newaxis,:,:,:]])[0][0]
    images=[]*11
    for i in range(11):
        images.append(get2[i]([input_nn[np.newaxis,:,:,:]])[0][0])
        max_pix=max(max_pix,np.max(images[i]))
        min_pix=min(min_pix,np.min(images[i]))
    for i in range(11):
        images[i]=remap(images[i],max_pix, min_pix)
    w = [[0]*128 for i in range(nc)]
    input_image=skimage.img_as_ubyte(adapt_input(image, 80))
    if (paperino==0):
        input_image3=skimage.img_as_ubyte(adapt_input(image, 120))
    input_image2=skimage.img_as_ubyte(adapt_input(image, 64))
    image[:, :, :]=image2[:,:,:]
               
# INPUT
    if conc==0:
        if video==True:
            image[50:170, 50:170]=input_image3[:,::-1,:]                
        else:
            image[50:170, 50:170]=inputtinod
    if conc==1:
        if video==True:
            image5[192:272, 10:90]=input_image[:,::-1,:]                
        else:
            image5[192:272, 10:90]=inputtinob
               

# LAYERS
    if conc==1:
        for j in range(3):
            A2 = 240-(64*3+2)//2+(64+1)*j    
            if video==True:
                image5[A2:A2+64, 98:162, (2-j)%3]=input_image2[:,:,j]
            else:
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
                elif j==8 or j==9 or j==10:
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
    Z1=Z
    for i in range(128):
        for j in range (nc):
            w[j][i]=model.layers[15].get_weights()[0][i][j]
        color=np.argmax(np.asarray(w)[:,i])
        F, G, H=image.shape[0]-51-min(int(f_v[i]*10), 298), image.shape[0]-51, 40+i*2
        image[F:G,H:H+2,0:3].fill(0)
        image[F:G,H:H+2,(color-1)%3].fill(255)

        y = -w[0][i]*length+length//2*w[1][i]+length//2*w[2][i]
        x = 10*(8.66*w[1][i]-8.66*w[2][i])
        cv2.line(image, (int(Z1), int(center)), ((int(Z1+ x*f_v[i]/l_verme)), int(center+ y*f_v[i]/l_verme)), (20,20,20), 2)
        Z1=Z1+int(x*f_v[i]/l_verme)
        center=center+y*f_v[i]/l_verme
    
# OUTPUT
    image[50:130, image.shape[1]-130:image.shape[1]-50]=output[b]
    
    b2=b
#    out.write(image)

    key = cv2.waitKey(20)
    if key & 0xFF == ord('1'):
        video=True
        paperino=0
    if key & 0xFF == ord('c'):
        conc=1
    if key & 0xFF == ord('2'):
        video, inputtino, inputtinob, inputtinod=False, special_input2[0], special_input3[0], special_input4[0]
    if key & 0xFF == ord('3'):
        video, inputtino, inputtinob, inputtinod=False, special_input2[1], special_input3[1], special_input4[1]
    if key & 0xFF == ord('4'):
        video, inputtino, inputtinob, inputtinod=False, special_input2[2], special_input3[2], special_input4[2]
    if key & 0xFF == ord('5'):
        video=False
        inputtino, inputtinod=neuron[neuron_i], skimage.img_as_ubyte(adapt_input(neuronb[neuron_i], 120))
        neuron_i=(neuron_i+1)%128
    if key & 0xFF == ord('6'):
        inputtino, inputtinod=neuron[128], skimage.img_as_ubyte(adapt_input(neuronb[128], 120))
    if key & 0xFF == ord('7'):
        inputtino, inputtinod=neuron[129], skimage.img_as_ubyte(adapt_input(neuronb[129], 120))
    if key & 0xFF == ord('8'):
        inputtino, inputtinod=neuron[130], skimage.img_as_ubyte(adapt_input(neuronb[130], 120))
    if key & 0xFF == ord('9'):
        if video==False:
            papero=generate_heatmap(inputtino)
            inputtinod=papero
        elif video==True:
            papero=generate_heatmap(input_image2)      
            input_image3=papero
        paperino=1
    if conc==1:
        image=np.concatenate((image5,image), axis=1)
    if key & 0xFF == ord('q'):
        break

    cv2.imshow('The Penultimate Layer',image)
    
cap.release()
cv2.destroyAllWindows()


