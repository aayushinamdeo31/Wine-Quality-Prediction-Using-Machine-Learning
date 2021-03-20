# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:39:41 2020

@author: DELL
"""

from flask import Flask, render_template, request# Flask-It is our framework which we are going to use to run/serve our application.
#request-for accessing file which was uploaded by the user on our application.

import numpy as np #used for numerical analysis

import pickle

model = pickle.load(open("wine_quality_model.pkl", "rb"))
app = Flask(__name__) #our flask app


@app.route('/') #default route
def hello_world():
  return render_template("index.html")
  
@app.route('/login', methods = ['POST']) #Main page route
def admin():

  p = request.form['type']

  if p =="White":
    p = 0
  elif p == "Red":
    p = 1

  q = float(request.form['fa'])
  r = float(request.form['va'])
  s = float(request.form['ca'])
  t = float(request.form['rs'])
  u = float(request.form['chl'])
  v = float(request.form['fsd'])
  w = float(request.form['tsd'])
  x = float(request.form['d'])
  y = float(request.form['ph'])
  z = float(request.form['sp'])
  a = float(request.form['ah'])

  sample = [[p, q, r, s, t, u, v, w, x, y, z, a]]

  test = model.predict(sample)

  if test == 0:
    test = "Low"
  elif test == 1:
    test = "Medium"
  elif test == 2:
    test = "High"

  return render_template("index.html", test = "The Quality of Wine is " + test)


# type,fixed acidity,volatile acidity,
# citric acid,residual sugar,chlorides,free sulfur dioxide,
# total sulfur dioxide,density,pH,sulphates,alcohol

@app.route('/user')
def user():
  return "Hye User"



if __name__ == '__main__':
  #app.run(host='0.0.0.0', port=8000,debug=False)
  app.run(debug = True) #running our flask app