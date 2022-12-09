#Importing libraries necessary for CNN Image Prediction
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

#Importing libraries necessary for Flask Application development.
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#Loading saved prediction model.
app = Flask(__name__)
model = load_model("ECG.h5")

#Page loaded when the app begins...
@app.route("/") #default route to the page 
def about():
    return render_template("about.html")

#Loading HOME page...
@app.route("/about") #default route to the page 
def home():
    return render_template("about.html") #rendering template of home page(about.html)

#Loading INFO page...
@app.route("/info") #default route to the page 
def information():
    return render_template("info.html") #rendering template of home page(info.html)

#Loading UPLOAD page...
@app.route("/upload") #default route to the page 
def test():
    return render_template("index6.html") #rendering template of upload page(index6.html)

#Loading PREDICT page...
@app.route("/predict",methods=["GET","POST"]) # CNN PREDICTION
def upload():
    if request.method=='POST':
        #ACCESSING FILE UPLOADED...
        f=request.files['file'] #Requesting file
        basepath=os.path.dirname('__file__') #Storing File Directory
        filepath=os.path.join(basepath,"uploads",f.filename) #Storing image in 'uploads' folder
        f.save(filepath) #File Saving
        
        #LOADING IMAGE FOR PREDICTION...
        img=image.load_img(filepath,target_size=(64,64)) #Image Loading and reshaping
        x=image.img_to_array(img) #Image to array convertion
        x=np.expand_dims(x,axis=0)#changing the dimensions of the image
        
        #PREDICTING THE IMAGE UPLOADED...
        pred=model.predict(x)#predicting classes
        y_pred = np.argmax(pred) 
        print("prediction",y_pred) #printing the prediction
    
        #DISPLAYING 'NAME' OF THE PREDICTED OUTPUT...
        index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction',
       'Premature Ventricular Contractions', 'Right Bundle Branch Block','Ventricular Fibrillation']
        result=str(index[y_pred])   # retriving the name of PREDICTION (to be displayed).
        
        #RESTURNING THE RESULT...
        return result 
    return None

#This runs only when this code is the first one to load.
if __name__=="__main__":
    app.run(debug=False) #RUNNING THE APP. 
    