from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import joblib
from joblib import load
import numpy as np
from keras.preprocessing import image
import json
import math
import os
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import ResNet50,MobileNet, DenseNet201, InceptionV3, NASNetLarge, InceptionResNetV2, NASNetMobile
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
 


app = Flask(__name__)

def changeYesNo(s):
    if s == 'Yes':
        return 1
    else:
        return 0

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    SmallLump = request.form['SmallLump']
    ChangesinSizeShape = request.form['ChangesinSizeShape']
    SkinChanges = request.form['SkinChanges']
    NippleChanges = request.form['NippleChanges']
    LumpinUnderarm = request.form['LumpinUnderarm']
    Swelling = request.form['Swelling']
    LargeLumpMass = request.form['LargeLumpMass']
    BonePain = request.form['BonePain']
    Cough = request.form['Cough']
    AbdominalPain = request.form['AbdominalPain']
    Headaches = request.form['Headaches']
    WeightLossFatigue = request.form['WeightLossFatigue']
    
    SmallLump = changeYesNo(SmallLump)
    ChangesinSizeShape = changeYesNo(ChangesinSizeShape)
    SkinChanges = changeYesNo(SkinChanges)
    NippleChanges = changeYesNo(NippleChanges)
    LumpinUnderarm = changeYesNo(LumpinUnderarm)
    Swelling = changeYesNo(Swelling)
    LargeLumpMass = changeYesNo(LargeLumpMass)
    BonePain = changeYesNo(BonePain)
    Cough = changeYesNo(Cough)
    AbdominalPain = changeYesNo(AbdominalPain)
    Headaches = changeYesNo(Headaches)
    WeightLossFatigue = changeYesNo(WeightLossFatigue)
    
    arr = np.array([[SmallLump, ChangesinSizeShape, SkinChanges, NippleChanges, LumpinUnderarm, Swelling, LargeLumpMass, BonePain, Cough, AbdominalPain, Headaches, WeightLossFatigue]])
    print(arr)
    
    clf = load('decisiontree.joblib')
    stage = clf.predict(arr)
    file = request.files['file1']
    file.save('static/file1.png')  # Save as PNG
    model = load_model("trained.h5")
    img_path = 'static/file1.png'  # File path with correct extension
    img = cv2.imread(img_path)
    tempimg = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(tempimg)
    img_array = preprocess_input(img_array)
    img = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img)

    # Check if the maximum predicted probability is greater than or equal to 0.5
    if np.max(prediction) >= 0.5:
        prediction_label = "Malignant"
    else:
        prediction_label = "Benign"

    classification = 'Prediction : %s' % (prediction_label)
    

    if prediction_label == "Malignant":
        phase = ''
        for ele in stage:
            phase += ele
        classification = "Prediction: Malignant and Stage: %s" % (phase)

        # Add recommendations based on cancer stage
        recommendations = get_recommendations_based_on_stage(phase)
        classification += "\n\nRecommendations:\n%s" % (recommendations)

    return render_template("/predict.html", prediction_label=classification)

def get_recommendations_based_on_stage(stage):
   
    recommendations = {
        'Stage 1': 'Surgery, such as lumpectomy or mastectomy, often accompanied by radiation therapy; additional treatments like hormone therapy or chemotherapy based on tumor characteristics.',
        'Stage 2': 'Combined treatment approach involving surgery, chemotherapy, radiation, and hormone therapy; specifics depend on tumor size and lymph node involvement.',
        'Stage 3': 'Combined treatment approach involving surgery, chemotherapy, radiation, and hormone therapy; specifics depend on tumor size and lymph node involvement.',
        'Stage 4': 'Metastatic or advanced breast cancer management with systemic therapies like chemotherapy, hormone therapy, targeted therapy, or immunotherapy; emphasis on quality of life, symptom control, and regular monitoring.'
    }

    return recommendations.get(stage, 'No specific recommendations available for this stage.')

if __name__ == "__main__":
    app.run(debug=True)
