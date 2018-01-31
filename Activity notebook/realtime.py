#%%
import numpy as np
import cv2
import skimage
import skimage.io
import skimage.viewer
import keras
import warnings

# User options
model_name="models/model_venus.model"
ssz=150
barw=50

classimages0=[]
classimages1=[]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for c in range(3):
        cim=cv2.cvtColor(skimage.img_as_ubyte(skimage.transform.resize(
                skimage.io.imread("figures/c{}.png".format(c)),(barw,barw)))[:,:,0:3], 
                cv2.COLOR_RGB2BGR)
        classimages1.append(cim)
        classimages0.append(255-(255-cim)//3)

# Setup acquisition
cap = cv2.VideoCapture(0)
cap.release()
cap = cv2.VideoCapture(0)

from keras import backend as K
K.set_learning_phase(0)
model = keras.models.load_model(model_name)
process = K.function([model.layers[0].input], [model.layers[-1].output])
def analyze(frame):
    data = process([frame[np.newaxis,:,:,:]])
    return data[0][0,:],[d[0,:] for d in data[1:]]

windowname="Image"
cv2.namedWindow(windowname)
cv2.moveWindow(windowname, 0, 0)

def preprocess(im):
    return skimage.transform.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB),(64,64))

def drawbars(im, values):
    barh=200
    h,w=im.shape[0:2]
    for i,v in enumerate(values):
        x0, x1 = i*barw, i*barw+barw
        y0, y1 = h-barw, h
        cv2.rectangle(im, (x0,h-barw), (x1, h-int(v*barh)-barw), (255,255,255), -1)
        if(v==max(values)):
            im[y0:y1,x0:x1,:] = classimages1[i][:,:,0:3]
        else:
            im[y0:y1,x0:x1,:] = classimages0[i][:,:,0:3]
            
            
while(True):
    ret, im = cap.read()
    im = cv2.flip(im, 1)
    h, w = im.shape[0:2]
    r0, r1 = h//2-ssz, h//2+ssz
    c0, c1 = w//2-ssz, w//2+ssz
    
    frame = preprocess(im[r0:r1,c0:c1,:])
    out,hidden = analyze(frame)
    
    cv2.circle(im, (w//2, h//2), ssz, (255,255,255), 5)
    drawbars(im,out)
    cv2.imshow(windowname,im)
    
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.startWindowThread()
cv2.destroyAllWindows()
