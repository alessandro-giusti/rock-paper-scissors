#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:51:43 2018

@author: federico
"""

import hug

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker, relationship, backref
from marshmallow_sqlalchemy import ModelSchema
from sqlalchemy.orm.session import Session


engine = sa.create_engine('sqlite:///test.db')
session = scoped_session(sessionmaker(bind=engine))
Base = declarative_base()




class User(Base):
    __tablename__ = 'users'
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String,nullable=False)
    
   
    #def __repr__(self):
    #    return '<User(name={self.name!r})>'.format(self=self)

class UserSchema(ModelSchema):
    class Meta:
        model = User

class Project(Base):
    __tablename__ = 'projects'
    id = sa.Column(sa.Integer, primary_key=True)
    title = sa.Column(sa.String)
    author_id = sa.Column(sa.Integer, sa.ForeignKey('users.id'))
    author = relationship("User", backref=backref('projects'))

class ProjectSchema(ModelSchema):
    class Meta:
        model = Project
        # optionally attach a Session
        # to use for deserialization
        sqla_session = session

Base.metadata.create_all(engine)




#
#author_schema = AuthorSchema()
#author = Author(name='Chuck Paluhniuk')
#book = Book(title='Fight Club', author=author)
#session.add(author)
#session.add(book)
#session.commit()





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


@hug.get('/test')
def make_simple_query(resource: Resource):
    for word in ["gianni", "pinotto", "amerigo"]:
        test_model = User()
        test_model.name = word
        resource.db.add(test_model)
        resource.db.flush()
    return " ".join([obj.name for obj in resource.db.query(User).all()])

#user_schema = UserSchema()
# {'books': [123], 'id': 321, 'name': 'Chuck Paluhniuk'}
#dump_data={"name":"mauro","test":"ciao"}
#test=user_schema.load(dump_data, session=session).data
# <Author(name='Chuck Paluhniuk')>