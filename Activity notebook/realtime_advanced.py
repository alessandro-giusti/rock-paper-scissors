#%%
import numpy as np
import cv2
import skimage
import skimage.io
import skimage.viewer
import keras
import warnings
import matplotlib.pyplot as plt
import argparse
from collections import deque
from cmap_utils import generate_custom_cmap
parser = argparse.ArgumentParser(description='Rock-paper-scissors realtime demo.')
parser.add_argument('--fullscreen', action='store_true', help='run in fullscreen')
args = parser.parse_args()

# User options
model_name="models/model_venus.model"
ssz=150
barw=50
mkviz=True
apply_cmap = False
cmaps = deque(['viridis'] + generate_custom_cmap())

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

def preprocess(im):
    return skimage.transform.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB),(64,64))

from keras import backend as K
K.set_learning_phase(0)
model = keras.models.load_model(model_name)
process = K.function([model.layers[0].input], 
                     [model.layers[-1].output,
                      model.layers[0].output,
                      #model.layers[2].output,
                      model.layers[5].output,
                      model.layers[8].output,
                      model.layers[12].output])
def analyze(frame):
    data = process([frame[np.newaxis,:,:,:]])
    return data[0][0,:],[d[0,:] for d in data[1:-1]],data[-1][0,:]

out,hidden,dense = analyze(preprocess(np.zeros((100,100,3),"uint8")))

remap_std=0.3
def remap(m):
    return np.clip(((m-np.mean(m))/np.std(m)*remap_std+0.5)*255,0,255).astype("uint8")

nrowss=[6,5,4]


def mkmappings_color():
    mappings=[np.random.randint(0,hidden[i].shape[2],(nrowss[i],3)) for i in range(len(hidden))] 
    densemapping=np.random.randint(0,dense.shape[0],(7,80,3))
    return mappings,densemapping

def mkmappings_gray():
    mappings=[np.random.randint(0,hidden[i].shape[2],(nrowss[i],1)) for i in range(len(hidden))] 
    mappings=[np.repeat(mapping,3,axis=1) for mapping in mappings]
    densemapping=np.random.randint(0,dense.shape[0],(7,80,1))
    densemapping=np.repeat(densemapping,3,axis=2)
    return mappings,densemapping

mappings,densemapping=mkmappings_gray()

def mkcolumn(h,mapping,height):
    hm=np.concatenate([remap(h[:,:,mapping[i,:]]) for i in range(mapping.shape[0])], axis=0)
    return skimage.transform.resize(hm,[height,int(height/hm.shape[0]*hm.shape[1])],order=0,preserve_range=True).astype("uint8")


windowname="Image"
if(args.fullscreen):
	cv2.namedWindow(windowname, cv2.WND_PROP_FULLSCREEN)
	cv2.moveWindow(windowname, 0, 0)
	cv2.setWindowProperty(windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
	cv2.namedWindow(windowname)

#inner="Inner"
#cv2.namedWindow(inner)
#cv2.moveWindow(inner, 700, 0)
#cv2.setWindowProperty(windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#cv2.startWindowThread()



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
            
def drawtinybars(im, values):
    barh=100
    barw=3
    for i,v in enumerate(values):
        x0, x1 = i*barw, i*barw+barw
        cv2.rectangle(im, (x0,0), (x1, max(0,int(v*barh))), (255,255,255), -1)

            
while(True):
    ret, im = cap.read()
    im = cv2.flip(im, 1)
    h, w = im.shape[0:2]
    r0, r1 = h//2-ssz, h//2+ssz
    c0, c1 = w//2-ssz, w//2+ssz
    
    frame = preprocess(im[r0:r1,c0:c1,:])
    out,hidden,dense = analyze(frame)
    
    cv2.circle(im, (w//2, h//2), ssz, (255,255,255), 5)
    drawbars(im,out)
    #drawtinybars(im,dense)

    if(mkviz):
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                viz=cv2.cvtColor(skimage.img_as_ubyte(frame), cv2.COLOR_BGR2RGB)
                
        #viz=skimage.transform.rescale(viz, 10, order=0)
        viz=np.concatenate([mkcolumn(h,mapping,im.shape[0]) for h,mapping in zip(hidden,mappings)],1)
        if apply_cmap:
            viz=cv2.cvtColor(plt.get_cmap(cmaps[0])(viz[:, :, 0], bytes=True)[:, :, :3], cv2.COLOR_BGR2RGB)

        im=np.concatenate([np.pad(im,((0,0),(0,5),(0,0)),mode="constant"),viz],1)
        viz=remap(dense[densemapping])
        viz=skimage.transform.resize(viz,(int(viz.shape[0]/viz.shape[1]*im.shape[1]),im.shape[1]),order=0,preserve_range=True).astype("uint8")
        if apply_cmap:
            viz=cv2.cvtColor(plt.get_cmap(cmaps[0])(viz[:, :, 0], bytes=True)[:, :, :3], cv2.COLOR_BGR2RGB)

        im=np.concatenate([np.pad(im,((0,5),(0,0),(0,0)),mode="constant"),viz],0)

        
    cv2.imshow(windowname,im)
    #cv2.imshow(inner,viz)
    
    key = cv2.waitKey(20)
    if key & 0xFF == ord('1'):
        remap_std=0.1
    if key & 0xFF == ord('2'):
        remap_std=0.2
    if key & 0xFF == ord('3'):
        remap_std=0.3
    if key & 0xFF == ord('4'):
        remap_std=0.4
    if key & 0xFF == ord('5'):
        remap_std=0.5
    if key & 0xFF == ord('c'):
        mappings,densemapping=mkmappings_color()
        apply_cmap = False
        mkviz=True
    if key & 0xFF == ord('g'):
        mappings,densemapping=mkmappings_gray()
        apply_cmap = False
        mkviz=True
    if key & 0xFF == ord('m'):
        mappings,densemapping=mkmappings_color()
        apply_cmap = True
        mkviz=True
    if key & 0xFF == ord('n'):
        cmaps.rotate(-1)
    if key & 0xFF == ord('p'):
        cmaps.rotate(1)
    if key & 0xFF == ord('v'):
        mkviz=not mkviz
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.startWindowThread()
cv2.destroyAllWindows()
