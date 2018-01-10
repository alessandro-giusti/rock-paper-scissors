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
from falcon import HTTP_400


###Global variables:
logger_container = []
model_container = []
allowed=[]
pending=[]
black=[]

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

def check_if_allowed(device_id):
    if device_id in allowed:
        return True
    else:
        return False

def check_if_pending(device_id):
    if device_id in pending:
        return True
    else:
        return False

    
def check_if_black(device_id):
    if device_id in black:
        return True
    else:
        return False
    
def check_evil_id(device_id):
    if '/' in device_id or '.' in device_id:
        black.append[device_id]
        with open('black.txt', "a") as f:
            f.write(device_id + '\n')
            
def check_id(device_id, response):
    check_evil_id(device_id)
    if check_if_allowed(device_id):
        return 0
    else:
        if check_if_black(device_id):
            response.status = HTTP_400
            return 1
        else:
            with open('pending.txt', "a") as f:
                f.write(device_id + '\n')
            response.status = HTTP_400
            return 2


@hug.post()
def refresh():
    
    with open('allowed.txt', "r") as f:
         allowed.clear()
         allowed.extend(f.read().splitlines())
    with open('pending.txt', "r") as f:
         pending.clear()
         pending.extend(f.read().splitlines())
    with open('black.txt', "r") as f:
         black.clear()
         black.extend(f.read().splitlines())




@hug.startup()
def init(api):
     logger_container.append(init_logger())
     model_container.append(keras.models.load_model(model_name))
     with open('allowed.txt', "r") as f:
         allowed.extend(f.read().splitlines())
     with open('pending.txt', "r") as f:
         pending.extend(f.read().splitlines())
     with open('black.txt', "r") as f:
         black.extend(f.read().splitlines())
         
     
    
    
    
@hug.get()
def testDeviceId(device_id,response):
    control= check_id(device_id, response)
    if control == 2:
        return {'error': "not approved yet"}
    elif control == 1:
        return {'error': "device blacklisted"} 

@hug.get('/')
def test():
     return {'message': 'server is online'}

@hug.post()
def photo(body,request, response):
    """
    Labels: "rock, paper, scissor, testing"
    """
    
    logger = logger_container[0]
    model = model_container[0]
    
    label = body["label"].decode("utf-8") 
    device_id = body["deviceId"].decode("utf-8") 
    
    control= check_id(device_id, response)
    if control == 2:
        return {'error': "not approved yet"}
    elif control == 1:
        return {'error': "device blacklisted"} 
    
    
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
    
@hug.delete('/photo/{photoname}')
def photo(photoname,response):
    label=photoname.split("-")[1]
    device_id=photoname.split("-")[2]
    control= check_id(device_id, response)
    if control == 2:
        return {'error': "not approved yet"}
    elif control == 1:
        return {'error': "device blacklisted"} 
    
    os.remove(label+"/"+photoname) 


       
