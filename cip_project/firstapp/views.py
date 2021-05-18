from django.shortcuts import render

# Create your views here.
from django.core.files.storage import FileSystemStorage

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


import tensorflow as tf
import json
import numpy as np

modell=tf.keras.models.load_model('./model/predictmodel.h5')
char_array = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}


def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictges(request):
    print(request)
    print(request.POST.dict())
    fileobj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileobj.name,fileobj)
    filePathName=fs.url(filePathName)
    testimg='.' + filePathName
    img = image.load_img(testimg, target_size=(64,64))
    img =  image.img_to_array(img)             
    img = np.expand_dims(img, axis=0)         
    img = tf.keras.applications.vgg16.preprocess_input(img)    
    
    pred=modell.predict(img)
    index = np.argmax(pred)
    pred_value = char_array[index]

    context={'filePathName':filePathName,'PredictedLabel':pred_value}
    return render(request,'index.html',context)  
