#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:46:44 2017

@author: fede
"""

"""First hug API (local, command-line, and HTTP access)"""
import hug
import os
import logging
import datetime
import skimage
from skimage import io
from skimage import transform
from os import makedirs
import keras
import numpy as np



###Global variables:
logger_container = []
model_container = []
model_name="rps.model"

def translate_id(dev_id,label):
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "-" + label + "-" + dev_id + ".jpg"

def init_logger():
   
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler('hello.log')
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

def adapt_input(im, size):
    h, w = im.shape[0:2]
    sz = min(h, w)
    im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:] 
    im = skimage.transform.resize(im, (size, size, 3), mode='reflect')
    return im


@hug.startup()
def init(api):
     logger_container.append(init_logger())
     model_container.append(keras.models.load_model(model_name))
    
    
    
    
@hug.local()
@hug.get(examples='name=Timothy&age=26')
def happy_birthday(name, age, hug_timer=3):
    """Says happy birthday to a user"""
    return {'message': 'Happy {0} Birthday {1}!'.format(age, name),
            'took': float(hug_timer)}
@hug.get()
def test():
     return {'message': 'server is online'}

@hug.get("/tost")
def testString():
    return "Ciao"

@hug.post()
def photo(body,request):
    """
    Labels: "rock, paper, scissor, testing"
    """
    
    logger = logger_container[0]
    model = model_container[0]
    
    label = body["label"].decode("utf-8") 
    device_id = body["deviceId"].decode("utf-8") 
    
    
    file_name = translate_id(device_id, label) #TODO check if JPG is alright
    
    full_path = label + "/" + file_name   
    makedirs(label, exist_ok=True)
    

    
    
    with open(full_path, 'wb+') as file:
        file.write(body["image"])
    
   
    
    logger.info("\n-----"+
                "\nLabel:" + label +
                "\nDeviceId:" + device_id +
                "\nFileName:" + file_name +
                "\n-----")
    
    if label != "testing":
        return file_name
    else:
        a_input=full_path
        to_test=[]
        to_test.append((adapt_input(skimage.io.imread(a_input), 64)))
        return model.predict(np.asarray(to_test)).tolist()
    
    #return filename
    #if testing, return test results
    
@hug.delete('/photo/{deviceId}')
def photo(deviceId):
    label=deviceId.split("-")[1]
    os.remove(label+"/"+deviceId) 
       
