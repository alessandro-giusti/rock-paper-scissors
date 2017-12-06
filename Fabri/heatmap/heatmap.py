## FIND IDEAL INPUT
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras import layers

# USER OPTIONS
model_name="rps.model"
step=10
iterations=100
trick1=1
output_nr=2

# VARIABLES
model = keras.models.load_model(model_name)
layer_name=model.layers[6].name
nc=model.output_shape[1]
final_output=np.zeros((64*2, 64*iterations//9, 3))
W1=[]
A=64
conv_indices=[0,2,5,8,12,15]
for i in (conv_indices):
    W = model.layers[i].get_weights()
    W1.append(np.copy(W[0]))
print(model.summary())

def trick(i):
    if i <=3:
        W = model.layers[conv_indices[i]].get_weights()
        W[0]=(abs(W[0])+W[0])/2
        model.layers[conv_indices[i]].set_weights(W)
    else:
        pass

def untrick():
    for i, index in enumerate(conv_indices):
        W = model.layers[index].get_weights()
        W[0]=W1[i]
        model.layers[index].set_weights(W)

def visualization2():
#    X=np.random.random((1, 64, 64, 3))*0.2+0.5
    A=64
    X=np.zeros((1, 64, 64, 3))
    X.fill(0.5)
#    objective = model.get_layer(layer_name).output[0,:,:,0]
    objective = model.output[0,output_nr]
    c=K.gradients(objective, model.input)[0]
#    c /= (K.sqrt(K.mean(K.square(c))) + 1e-5)
    get=K.function([model.input, K.learning_phase()],[objective, c])
    for j in range(iterations):
        if (trick1==1 and j<6):
            trick(j)
#            spegni_mappe1()
        loss_value, grads_value=get([X, 1])
        print(grads_value[0,30,:,1])
        print(np.mean(grads_value))
        if np.max(grads_value)>0:
            step=0.6/np.max(grads_value)
        if (j%10)==0:
            final_output[0:64,A:(64+A),:]=(np.clip(grads_value[0]*step+0.5, 0, 1))
            final_output[64:128,A:(64+A),:]=X[0]
            A+=64
#        grads_value[0,:,:,1].fill(0)
#        grads_value[0,:,:,2].fill(0)
        X += grads_value*step
        X=np.clip(X, 0, 1)
        untrick()
        print(j)
    print("grads_value", np.mean(grads_value))
    print("np.min", np.min(X[0]))
    print("np.max", np.max(X[0]))
#    X[:,:,0].fill(0)
#    final_output[:,:,1].fill(0) final_output[:,:,2].fill(0)
    plt.imshow(final_output, vmin=0, vmax=1)
    plt.show()
    plt.imsave('image.jpg', X[0], vmin=0, vmax=1)

### PLOT
##def visualization1():
##    for i in range(128):
##        print(i)
##        map_image=loop(i)
###        scipy.misc.imsave('IMAGES/fishneuron%d.jpg' %(i+1), map_image)
##
##def visualization3():
###    input1=np.zeros((1, 64, 64, 3))+127
##    input1=skimage.io.imread("IMAGES/paper3.jpg")
##    input1=input1[np.newaxis,:,:,:]
##    input1=input1.astype("float32")
##
##    heatmap=np.zeros((64, 64))
##    a = model.predict([input1])
##    b = np.argmax(a[0])
##    print(b)
##    african_elephant_output= model.output[:,b]
##    lcl=model.get_layer("conv2d_12")
##    grads=K.gradients(african_elephant_output, lcl.output)[0]
##    pool_g=K.mean(grads, axis=(0, 1, 2))
##    get=K.function([model.input, K.learning_phase()], [pool_g, lcl.output[0]])
##    pool_g_value, clo_value=get([input1, 1])
##    for i in range(3):
##        clo_value[:,:,i] *=pool_g_value[i]
##    heatmap= np.mean(clo_value, axis=-1)
##    heatmap=np.maximum(heatmap, 0)
##    
##    img=cv2.imread("IMAGES/paper3.jpg")
##    print(img.shape)
##    heatmap=np.uint8(255*heatmap)
##    print(heatmap.shape)
##    heatmap=np.resize(heatmap, (img.shape[1], img.shape[0]))
##    heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
##    superimposed_img=heatmap*0.4+img
##    cv2.imwrite("ciao.jpg", superimposed_img)


visualization2()

#layer_name=model.layers[12].name
#layer_name=model.layers[0].name
#w = [[0]*128 for i in range(nc)]
#for i in range(128):
#    for j in range (nc):
#        w[j][i]=model.layers[15].get_weights()[0][i][j]
#print("hello")

##def remap(m):
##    return ((0.5+m/2)*255).astype("uint8") 
###    return np.clip(((m-np.mean(m))/np.std(m)+0.5)*255,0,255).astype("uint8")

##def adapt_input(im, size):
##    h, w = im.shape[0:2]
##    sz = min(h, w)
##    im=im[(h//2-sz//2):(h//2+sz//2),(
##        w//2-sz//2):(w//2+sz//2),:] 
##    im = skimage.transform.resize(im, (size, size, 3), mode='reflect')
##    return im

##def loop(neuron_nr):
##    X=np.random.random((1, 64, 64, 3))
##    X=np.zeros((1, 64, 64, 3))
##    X.fill(0.5)
##    objective=model.get_layer(layer_name).output[0,neuron_nr]
##    c=K.gradients(objective, model.input)[0]
##    c /= (K.sqrt(K.mean(K.square(c))) + 1e-5)
##    get=K.function([model.input],[objective, c])
##    for j in range(iterations):
###        img2=remap(input1[0])
###        scipy.misc.imsave('IMAGES/scissors_it%d.jpg' %(j+1), img2)
##        loss_value, grads_value=get([X])
##        print(grads_value)
##        print(np.mean(grads_value))
###        print(grads_value[0].shape)
###        grads_value[0,:,:,1].fill(0)
###        grads_value[0,:,:,2].fill(0)
###        plt.imshow(np.clip(grads_value[0]*step, -1, 1), vmin=-1, vmax=1)
###        plt.show()
##        X += grads_value*step
##        X=np.clip(X, 0, 1)
##        print(j)
##    plt.imshow(X[0])
##    plt.show()
##    return X

#    plt.matshow(heatmap)
#    plt.show()
#    image=np.zeros((nr_images*64+(nr_images-1)*margin,640, 3)).astype("uint8")
#    image[:,:,:].fill(255)
#    get2 = K.function([model.layers[0].input], [model.layers[13].output])
#        for j in range(128):
#            color=np.argmax(np.asarray(w)[:,j])
#            F, G, H=64*i+margin*i+64-min(int(f_v[j]*10), 64), 64*i+margin*i+64, 100+j*2
#            image[F:G,H:H+2,0:3].fill(0)
#            print(color)
#            image[F:G,H:H+2, color].fill(255)

#    plt.imshow(image)
#    plt.show()
#    results[0:64,0:64,:]=map_image
#    w = [[0]*128 for i in range(3)]
#    results=np.zeros((64,m*64+(m-1)*margin, 3))
#    for n in range(1):
#        for j in range (2):
#            w[j][n]=model.layers[15].get_weights()[0][neuron_nr-1][j]
#            color=np.argmax(np.asarray(w)[:,neuron_nr-1])
#    if color==0:
#        scipy.misc.imsave('IMAGES/NEURONS/rock/neuron_nr%d_.jpg' %(neuron_nr), img)
#    elif color==1:
#        scipy.misc.imsave('IMAGES/NEURONS/paper/neuron_nr%d_.jpg' %(neuron_nr), img)
#    elif color==2:
#        scipy.misc.imsave('IMAGES/NEURONS/scissors/neuron_nr%d_.jpg' %(neuron_nr), img)
#    return img

#ROCK:0, 1, 2
#PAPER: 3, 6, 7, 39, 40
#SCISSORS: 9, 11
#input1=skimage.io.imread("dory.jpg")
#input1=input1[np.newaxis,:,:,:]
#input1=input1.astype("float32")

# FLOAT - DA ZERO A UNO: DA NERO A BIANCO
# UINT8 - DA ZERO A 255
# ciccio=np.zeros((100,100,3))
#for i in range(10):
#    for j in range(10):
#        ciccio[i*10:i*10+10,0+j*10:10+j*10,:].fill(j*30+i)
#plt.imshow(ciccio.astype('uint8'))
#plt.show()

##ciccio=np.random.random((64,64,3))*20+128
##m=np.mean(ciccio)
##ciccio=ciccio/255
##print(m)
##plt.imshow(ciccio)
##plt.show()



#    return deprocess_image(img)
##def deprocess_image(x):
##    x -= x.mean()
##    x /= (x.std() + 1e-5)
##    x *=0.1
##    x+=0.5
##    x=np.clip(x, 0, 1)
##    x*= 255
##    x = np.clip(x, 0, 255).astype('uint8')
##    return x

#Visual Studio Code
##title=skimage.io.imread("IMAGES/title.png")
##x_axis=skimage.io.imread("IMAGES/x_axis.bmp")
##neuron_nr=skimage.io.imread("IMAGES/neuron_nr.bmp")
##output=[]*3
##output.append(skimage.io.imread(class1))
##output.append(skimage.io.imread(class2))
##output.append(skimage.io.imread(class3))
##get = K.function([model.layers[0].input], [model.layers[13].output])
##get2=[]*11
##b2, F, G, H=0, 0, 0, 0
##
##def adapt_input(im, size):
##    h, w = im.shape[0:2]
##    sz = min(h, w)
##    im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:] 
##    im = skimage.transform.resize(im, (size, size, 3), mode='reflect')
##    return im
##a_input="IMAGES/striato.jpg"
##input2=skimage.img_as_ubyte(adapt_input(skimage.io.imread(a_input), 64))
##input2b=skimage.img_as_ubyte(adapt_input(cv2.cvtColor(skimage.io.imread(a_input), cv2.COLOR_BGR2RGB), 80))
##b_input="IMAGES/japan.jpg"
##c_input="IMAGES/test.jpg"
##input3=skimage.img_as_ubyte(adapt_input(skimage.io.imread(b_input), 64))
##input3b=skimage.img_as_ubyte(adapt_input(cv2.cvtColor(skimage.io.imread(b_input), cv2.COLOR_BGR2RGB), 80))
##input4=skimage.img_as_ubyte(adapt_input(skimage.io.imread(c_input), 64))
##input4b=skimage.img_as_ubyte(adapt_input(cv2.cvtColor(skimage.io.imread(c_input), cv2.COLOR_BGR2RGB), 80))
##
##inputtino=input2
##inputtinob=input2b
##conc=0
##
#### STATIC IMAGE
##cap = cv2.VideoCapture(0)
##ret2, image2 = cap.read()
##
##Z=image2.shape[1]-215
##center=255
##if conc==0:
##    length=138
##for i in range(11):
##    get2.append(K.function([model.layers[0].input], [model.layers[i].output]))
##frame2 = []
##frame2.append(adapt_input(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB), 64))
##images3=[]*11
##for i in range(11):
##    images3.append(skimage.img_as_ubyte(np.clip(np.asarray(get2[i]([frame2])[0][0]), -1, 1)))
##
##
### SFONDO BIANCO
##image2[:, :, 0:3].fill(255)
##image5=np.copy(image2)
##for j in range(3):
##    A2 = 240+(64*3+2)//2-64-(64+1)*j
##    image5[A2:A2+64, 98:162, 0:3].fill(0)
##        
##B=182
##for j in range(11):
##    C=images3[j].shape[2]
##    D=images3[j].shape[0]
##    for i in range(images3[j].shape[2]):
##        A = 240+(D*C+C-1)//2-D-(D+1)*i
##        image5[A:A+D, B:B+D, 0:3].fill(0)
##    B=B+10+D
### TITLE
##image2[8:8+title.shape[0],60:60+title.shape[1],0:3]=title
### LABEL1
##image2[image2.shape[0]-50:image2.shape[0]-34,36:296,0:3]=x_axis
### LABEL2
##image2[image2.shape[0]-34:image2.shape[0]-20,168-31:168+31,0:3]=neuron_nr
### GRAPH2
##cv2.circle(image2,(Z, center-1), length, (220,220,220), -1)
##cv2.line(image2, (Z, center), (Z, center-length), (0,0,255), 2)
##cv2.line(image2, (Z, center), (Z, center+length), (180,180,180), 1)
##cv2.line(image2, (Z, center), (Z+int(0.866*length), center+int(0.866*length//2)), (255,0,0), 2)
##cv2.line(image2, (Z, center), (Z-int(0.866*length), center-int(0.866*length//2)), (180,180,180), 1)
##cv2.line(image2, (Z, center), (Z-int(0.866*length), center+int(0.866*length//2)), (0,127,0), 2)
##cv2.line(image2, (Z, center), (Z+int(0.866*length), center-int(0.866*length//2)), (180,180,180), 1)
### LEGEND
##W=240
##image2[image2.shape[0]-50-legend.shape[0]:image2.shape[0]-50,image2.shape[1]-40-legend.shape[1]:image2.shape[1]-40,0:3]=cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
##
##def spegni_mappa_0(mappa):
##    C = model.layers[0].get_weights()
##    for i in range(3):
##        for j in range(3):
##            for k in range(3):
##                C[0][i][j][k][mappa-1]= 0
##    model.layers[0].set_weights(C)
##    
##def spegni_mappa_1(mappa):
##    C = model.layers[2].get_weights()
##    for i in range(3):
##        for j in range(3):
##            for k in range(5):
##                C[0][i][j][k][mappa-1]= 0
##    model.layers[2].set_weights(C)
##
##
##def spegni_mappa_2(mappa):
##    C = model.layers[5].get_weights()
##    for i in range(3):
##        for j in range(3):
##            for k in range(5):
##                C[0][i][j][k][mappa-1]= 0
##    model.layers[5].set_weights(C)
##
##
##def spegni_mappa_3(mappa):
##    C = model.layers[8].get_weights()
##    for i in range(3):
##        for j in range(3):
##            for k in range(5):
##                C[0][i][j][k][mappa-1]= 0
##    model.layers[8].set_weights(C)
##
##
##def spegni_mappe1():
##    spegni_mappa_0(1)
##    spegni_mappa_0(2)
##    #spegni_mappa_0(3)
##    spegni_mappa_0(4)
##    spegni_mappa_0(5)
##
##def spegni_mappe2():
##    spegni_mappa_1(1)
##    spegni_mappa_1(2)
##    spegni_mappa_1(3)
###spegni_mappa_1(4)
##    spegni_mappa_1(5)
##
##def spegni_mappe3():
##    spegni_mappa_2(1)
###spegni_mappa_2(2)
###spegni_mappa_2(3)
##    spegni_mappa_2(4)
###    spegni_mappa_2(5)
##
##def spegni_mappe4():
###    spegni_mappa_3(1)
###    spegni_mappa_3(2)
###    spegni_mappa_3(3)
###spegni_mappa_3(4)
##    spegni_mappa_3(5)    
##

remap_std=0.3
def remap(m):
    return np.clip(((m-np.mean(m))/np.std(m)*remap_std+0.5)*255,0,255).astype("uint8")

##
#### MAIN LOOP
##while(True):
##    ret, image = cap.read()
##    size=200
##    A, B, C, D = int(image.shape[0]//2-size/2)-40, int(image.shape[0]//2+size/2)-40, int(image.shape[1]//2-size/2), int(image.shape[1]//2+size/2)
##    Z=image2.shape[1]-215
##    center=255
##    frame3 = []
##    if feed=="video":
##        frame3.append(adapt_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 64))
##    else:
##        frame3.append(inputtino)
###    frame3.append(adapt_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 64))
###    frame3.append(input2)
##    a = model.predict(np.asarray(frame3)).tolist()
##    b = a[0].index(max(a[0]))
##    f_v = get([frame3])[0][0]
##    images=[]*11
##    for i in range(11):
###        images.append(skimage.img_as_ubyte(np.clip(get2[i]([frame3])[0][0], -1, 1)))
##        images.append(remap(get2[i]([frame3])[0][0]))
##    w = [[0]*128 for i in range(nc)]
##    input_image=skimage.img_as_ubyte(adapt_input(image, 80))
##    input_image3=skimage.img_as_ubyte(adapt_input(image, 120))
##    input_image2=skimage.img_as_ubyte(adapt_input(image, 64))
##    image[:, :, :]=image2[:,:,:]
##               
### INPUT
##    if conc==0:
##        if feed=="video":
##            image[50:170, 50:170]=cv2.flip(input_image3, 0)                
##        else:
##            image[50:130, 50:130]=inputtinob
##    if conc==1:
##        if feed=="video":
##            image5[192:272, 10:90]=cv2.flip(input_image, 0)                
##        else:
##            image5[192:272, 10:90]=inputtinob
##
###    image[60:180, 60:180]= cv2.flip(input_image, 1)                
###    image[50:130, image.shape[1]-130:image.shape[1]-50]=input2b                
##
### LAYERS
##    if conc==1:
##        for j in range(3):
##            A2 = 240-(64*3+2)//2+(64+1)*j    
##            if feed=="video":
##                image5[A2:A2+64, 98:162, (2-j)%3]=input_image2[:,:,j]
##            elif feed=="vid":
##                for k in range(3):
##                    image5[A2:A2+64, 98:162, k]=inputtino[:,:,j]
##
##        B=182
##        for j in range(10):
##            C=images[j].shape[2]
##            D=images[j].shape[0]
##            for i in range(C):
##                A = 240-(D*C+C-1)//2+(D+1)*i
##                if b2!=b:
##                    image5[A:A+D, B:B+D, 0:3].fill(0)
##                if j<2:
##                    image5[A:A+D, B:B+D, 1]=images[j][:, :, i]
##                elif j==2 or j==3:
##                    image5[A:A+D, B:B+D, 2]=images[j][:, :, i]
##                elif j==5 or j==6:
##                    image5[A:A+D, B:B+D, 0]=images[j][:, :, i]
##    #            elif j==4:
##    #                image[A:A+D, B:B+D, 0:3].fill(0)
##    #                for k in range(30):
##    #                    for l in range(30):
##    #                        image[A+k, B+l, (2+W115[k,l,i])%3]=255
##    #            elif j==8 or j==9:
##    #                image[A:A+D, B:B+D, 0]=images[j][:, :, i]
##    #                image[A:A+D, B:B+D, 1]=images[j][:, :, i]
##                elif j==8 or j==9 or j==10:
##    #                for k in range(6):
##    #                    for l in range(6):
##    #                        image[A+k, B+l, (2+W114[k,l,i])%3]=images[j][k, l, i]
##                    if i==3:
##                        image5[A:A+D, B:B+D, 0]=images[j][:, :, i]
##                    elif i==0 or i==2:
##                        image5[A:A+D, B:B+D, 2]=images[j][:, :, i]
##                    else:
##                        image5[A:A+D, B:B+D, 1]=images[j][:, :, i]     
##                else:
##                    for k in range(2):
##                        image5[A:A+D, B:B+D, k+1]=images[j][:, :, i]
##            B=B+10+D
##
### PENULTIMATE LAYER
##    for i in range(128):
##        for j in range (nc):
##            w[j][i]=model.layers[15].get_weights()[0][i][j]
##        color=np.argmax(np.asarray(w)[:,i])
##        F, G, H=image.shape[0]-50-min(int(f_v[i]*50), 298), image.shape[0]-51, 40+i*2
##        image[F:G,H:H+2,0:3].fill(0)
##        image[F:G,H:H+2,(color-1)%3].fill(255)
##
##        y = -w[0][i]*length+length//2*w[1][i]+length//2*w[2][i]
##        x = 10*(8.66*w[1][i]-8.66*w[2][i])
##        cv2.line(image, (int(Z), int(center)), ((int(Z+ x*f_v[i]/12)), int(center+ y*f_v[i]/12)), (20,20,20), 2)
##        Z=Z+int(x*f_v[i]/12)
##        center=center+y*f_v[i]/12
##    
### OUTPUT
##    image[50:130, image.shape[1]-130:image.shape[1]-50]=output[b]
##    
##    b2=b
###    out.write(image)
##
##    key = cv2.waitKey(20)
##    if key & 0xFF == ord('0'):
##        feed="video"
##    if key & 0xFF == ord('1'):
##        conc=1
##    if key & 0xFF == ord('2'):
##        feed, inputtino, inputtinob="vid", input2, input2b
##    if key & 0xFF == ord('3'):
##        feed, inputtino, inputtinob="vid", input3, input3b
##    if key & 0xFF == ord('4'):
##        feed, inputtino, inputtinob="vid", input4, input4b
##    if key & 0xFF == ord('5'):
##        spegni_mappe1()        
##    if key & 0xFF == ord('6'):
##        spegni_mappe2()
##    if key & 0xFF == ord('7'):
##        spegni_mappe3()
##    if key & 0xFF == ord('8'):
##        spegni_mappe4()
##    if conc==1:
##        image=np.concatenate((image5,image), axis=1)
##    if key & 0xFF == ord('q'):
##        break
##
##    cv2.imshow('The Penultimate Layer',image)
##    
##cap.release()
##cv2.destroyAllWindows()
##

