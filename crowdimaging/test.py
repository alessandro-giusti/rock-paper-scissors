#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:51:43 2018

@author: federico
"""


"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Imports
"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import hug

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker, relationship, backref
from marshmallow_sqlalchemy import ModelSchema
from sqlalchemy.orm.session import Session
import logging
import keras
import hashlib
import os
import skimage
import numpy as np
import datetime




"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Global variables
"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dbname = "database.db"
engine = sa.create_engine('sqlite:///' + dbname)
session = scoped_session(sessionmaker(bind=engine))
Base = declarative_base()
allowed=[]
pending=[]
black=[]



"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Helper methods
"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


     
def create_and_store_photo(problem,label,authed_user,current_time,device_id,image ):
    full_path = problem + "/" + label + "/" + authed_user + "/" + current_time + "_" +  device_id + "_" +  authed_user + "_" +  label + ".jpg"   
    for element in [problem,label,authed_user,device_id]:
        if '/' in element or '.' in element:
            return "total fail"
    os.makedirs(problem + "/" + label + "/" + authed_user, exist_ok=True)
    with open(full_path, 'wb+') as file:
        file.write(image)   
    return full_path
     

def adapt_input(im, size):
    h, w = im.shape[0:2]
    sz = min(h, w)
    im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:] 
    im = skimage.transform.resize(im, (size, size, 3), mode='reflect')
    return im        
    
@hug.directive()
class Resource(object):

    def __init__(self, *args, **kwargs):
        self._db = session()
        self.autocommit = True

    @property
    def db(self) -> Session:
        return self._db

    def cleanup(self, exception=None):
        if exception:
            self.db.rollback()
            return
        if self.autocommit:
            self.db.commit()


@hug.directive()#why?
def return_session() -> Session:
    return session()

def init_logger():
   
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.CRITICAL)

    # create a file handler
    handler = logging.FileHandler('logger.log')
    handler.setLevel(logging.CRITICAL)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

def hash_password(password, salt):
    """
    Securely hash a password using a provided salt
    :param password:
    :param salt:
    :return: Hex encoded SHA512 hash of provided password
    """
    password = str(password).encode('utf-8')
    salt = str(salt).encode('utf-8')
    return hashlib.sha512(password + salt).hexdigest()


def gen_api_key(username):
    """
    Create a random API key for a user
    :param username:
    :return: Hex encoded SHA512 random string
    """
    salt = str(os.urandom(64)).encode('utf-8')
    return hash_password(username, salt)

@hug.cli()
def authenticate_user(resource: Resource,username, password):
    """
    Authenticate a username and password against our database
    :param username:
    :param password:
    :return: authenticated username
    """
    user = None
    try:
        user=resource.db.query(User).filter(User.name==username).one()
        if user['password'] == hash_password(password, user.get('salt')):
            return user['username']
    except Exception as err:
        #TODO logger.warning("User %s not found", username)
        #TODO log exception
        resource.cleanup(exception = err)
        return False #TODO Add response status, or not? Check the false thing
    return False #Same error for wrong user and wrong password?
    
 

@hug.cli()
def authenticate_key(resource: Resource,api_key):
    """
    Authenticate an API key against our database
    :param api_key:
    :return: authenticated username
    """
    user = None
    try:
        user=resource.db.query(User).filter(User.api_key==api_key).one()
        return user['username']
    except Exception as err:
        #TODO logger.warning("User %s not found", username)
        #TODO log exception
        resource.cleanup(exception = err)
        return False #TODO Add response status, or not? Check the false thing
    return False #Same error for wrong user and wrong password?
    




"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Database classes
"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class User(Base):
    __tablename__ = 'users'
    id = sa.Column(sa.Integer, primary_key=True)
    username = sa.Column(sa.String,nullable=False,unique=True)
    
    password = sa.Column(sa.String,nullable=False)
    salt = sa.Column(sa.String,nullable=False)
    api_key = sa.Column(sa.String,nullable=False)#Unique, but should not be necessary to check
    
    affidability = sa.Column(sa.Integer,nullable=False)
    privilege_level=sa.Column(sa.Integer, primary_key=True) #100 Admin - 5 standard user
    #def __repr__(self):
    #    return '<User(name={self.name!r})>'.format(self=self)


class UserSchema(ModelSchema):
    class Meta:
        model = User
        #sqla_session = session




class Project(Base):
    __tablename__ = 'projects'
    id = sa.Column(sa.Integer, primary_key=True)
    model_path = sa.Column(sa.String, nullable=False)
    title = sa.Column(sa.String, nullable=False)
    owner_id = sa.Column(sa.Integer, sa.ForeignKey('users.id'),nullable=False)
    owner = relationship("User", backref=backref('projects'))
    private = sa.Column(sa.Boolean, nullable=False)
    extra_features = sa.Column(sa.String,nullable=False) #"" for no extra features, 


class ProjectSchema(ModelSchema):
    class Meta:
        model = Project
        #sqla_session = session



class Photo(Base):
    __tablename__ = 'photos'
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String,nullable=False,unique=True)
    label = sa.Column(sa.Integer,nullable=False)#for each int, there is a String associated
    owner_id = sa.Column(sa.Integer, sa.ForeignKey('users.id'),nullable=False)
    owner = relationship("User", backref=backref('photos'))
    project_id = sa.Column(sa.Integer, sa.ForeignKey('projects.id'),nullable=False)
    project = relationship("Project", backref=backref('photos'))
    affidability = sa.Column(sa.Integer,nullable=False)
    extra_features = sa.Column(sa.String,nullable=False)

class PhotoSchema(ModelSchema):
    class Meta:
        model = Photo
        #sqla_session = session










Base.metadata.create_all(engine)
api_key_authentication = hug.authentication.api_key(authenticate_key)
basic_authentication = hug.authentication.basic(authenticate_user)


"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Actual APIs
"""#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@hug.get('/user')
def get_user(resource: Resource,username):#,response):
    user_schema = UserSchema()
    test_model = resource.db.query(User).filter(User.username==username).one()
    dump_data = user_schema.dump(test_model).data
    return dump_data


@hug.post('/user')
def create_user(resource: Resource, username, password): #TODO add admin auth
   #TODO Query DB and check if user already exist
    new_user = User()
    new_user.username = username 
    
    salt = hashlib.sha512(str(os.urandom(64)).encode('utf-8')).hexdigest()
    password = hash_password(password, salt)
    api_key = gen_api_key(username)
    
    new_user.salt=salt
    new_user.password=password
    new_user.api_key=api_key
    
    
    #new_user.XXX = XXX
    new_user.affidability = 50
    try:
        resource.db.add(new_user)
        resource.db.flush()
        resource.cleanup()
        return {"Result":"succes"}
    except Exception as err:
        resource.cleanup(exception = err)
        return {"Result":err} #TODO Add response status

@hug.get('/api/get_api_key', requires=basic_authentication)
def get_token(resource: Resource, authed_user: hug.directives.user):
    """
    Get Job details
    :param authed_user:
    :return:
    """
    user=resource.db.query(User).filter(User.username==authed_user).one()

    if user:
        out = {
            'user': user['username'],
            'api_key': user['api_key']
        }
    else:
        # this should never happen
        out = {
            'error': 'User {0} does not exist'.format(authed_user)
        }

    return out

@hug.post('/photo')
def photo_post(resource: Resource, authed_user: hug.directives.user,body,response):
    """
    Labels: "rock, paper, scissor, testing"
    """
    
    global logger
    
    label = body["label"].decode("utf-8") 
    device_id = body["deviceId"].decode("utf-8")
    problem = body["problem"].decode("utf-8") 
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    image = body["image"]
    full_path=create_and_store_photo(problem,label,authed_user,current_time,device_id,image )

    logger.info("\n-----"+
                "\nLabel:" + problem +
                "\nLabel:" + label +
                "\nLabel:" + authed_user +
                "\nDeviceId:" + device_id +
                "\nFileName:" + current_time +
                "\n-----")
    
    if label != "testing":
        return {"Result":"Success"}
    else:
        a_input=full_path
        to_test=[]
        project=resource.db.query(Project).filter(Project.title==problem).one()
        model=keras.models.load_model(project.model)
        to_test.append((adapt_input(skimage.io.imread(a_input), 64)))
        return model.predict(np.asarray(to_test)).tolist()



@hug.not_found()
def not_found():
    return {'Nothing': 'to see'}


#user_schema = UserSchema()
# {'books': [123], 'id': 321, 'name': 'Chuck Paluhniuk'}
#dump_data={"name":"mauro","test":"ciao"}
#test=user_schema.load(dump_data, session=session).data
# <Author(name='Chuck Paluhniuk')>