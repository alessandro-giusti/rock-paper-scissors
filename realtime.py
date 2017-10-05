#%%
import numpy as np
import cv2
import skimage
import skimage.viewer
import keras

print('hello, world!')

# USER OPTIONS
model_name="models/model_venus.model"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

cap = cv2.VideoCapture(0)
cap.release()
cap = cv2.VideoCapture(0)
model = keras.models.load_model(model_name)

def resize(im, size):
    im = skimage.transform.resize(im, (size, size, 3), mode='reflect')
    return im

def square(im):
    h, w = im.shape[0:2]
    sz = min(h, w)
    im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:] 
    return im

while(True):
    ret, image = cap.read()
    cv2.flip(image, 1)
    
    size=200
    A, B, C, D = int(image.shape[0]//2-size/2), int(image.shape[0]//2+size/2), int(image.shape[1]//2-size/2), int(image.shape[1]//2+size/2)
    frame = np.array([resize(square(cv2.cvtColor(image[A:B,C:D,0:3], cv2.COLOR_BGR2RGB)), 64)])
    a = model.predict(frame)[0,:]
    print(a)
    
    cv2.circle(image,(image.shape[1]//2,image.shape[0]//2), 100, (255,255,255), 5)
    w=30
    h=200
    cv2.rectangle(image, (0*w,image.shape[0]), (1*w, image.shape[0]-int(a[0]*200)), (255,255,255), -1)
    cv2.rectangle(image, (1*w,image.shape[0]), (2*w, image.shape[0]-int(a[1]*200)), (255,255,255), -1)
    cv2.rectangle(image, (2*w,image.shape[0]), (3*w, image.shape[0]-int(a[2]*200)), (255,255,255), -1)
    
    fr=(frame[0,:,:,:]*255).astype("uint8")
    
    out.write(image)    
    cv2.imshow('frame',image)
    cv2.imshow('fr',fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
