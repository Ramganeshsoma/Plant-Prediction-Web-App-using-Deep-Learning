import numpy as np
import os
from flask import Flask, request, render_template
from keras.models import model_from_json
from keras.optimizers import Adam
import cv2

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    global loaded_model
    global model
    # loading saved model
    print(" * Loading Keras model...")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("amodel.h5")
    loaded_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy',
                         metrics=['accuracy'])
    print("Model Loaded")


    target = os.path.join(APP_ROOT, 'images/')
    print('Saving Uploaded image into Images Directory')
    if not os.path.isdir(target):
        os.mkdir(target)
    global destination
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
    print('Image Saved')
    #replacing '\' with '\\' to avoid reading it as a special character
    print('Converting Path text to Raw Path Text')
    print(destination)
    dest = destination.replace("\\", "\\\\")
    dest1 = dest.replace("//", "\\\\")
    print("done converting")
    print(dest1)
    print('Reading Image')
    image = cv2.imread(dest1)
    print('Resizing image to 96x96')
    image = cv2.resize(image, (96, 96))
    print('Converting image into Numpy array')
    np_image = np.array(image)
    print('Setting Numpy array dimensions')
    y = np.expand_dims(np_image, axis=0)
    print('Predicting')
    pred = loaded_model.predict(y)
    print(pred)
    print('It is predicted as:')

    y_pred_binary = pred.argmax(axis=1)
    print(y_pred_binary)

    if y_pred_binary == [0]:
        print("Black-Grass")
        y_pred_word = "Black-Grass"
    elif y_pred_binary == [1]:
        print("Charlock")
        y_pred_word = "Charlock"
    elif y_pred_binary == [2]:
        print("Cleavers")
        y_pred_word = "Cleavers"
    elif y_pred_binary == [3]:
        print("Common Chickweed")
        y_pred_word = "Common Chickweed"
    elif y_pred_binary == [4]:
        print("Common Wheat")
        y_pred_word = "Common Wheat"
    elif y_pred_binary == [5]:
        print("Fat Hen")
        y_pred_word = "Fat Hen"
    elif y_pred_binary == [6]:
        print("Loose silky-bent")
        y_pred_word = "Loose silky-bent"
    elif y_pred_binary == [7]:
        print("Maize")
        y_pred_word = "Maize"
    elif y_pred_binary == [8]:
        print("Scentless Mayweed")
        y_pred_word = "Scentless Mayweed"
    elif y_pred_binary == [9]:
        print("Shepherd's Purse")
        y_pred_word = "Shephers's Purse"
    elif y_pred_binary == [10]:
        print("Small-flowered Cranesbill")
        y_pred_word = "Small-flowered Cranesbill"
    elif y_pred_binary == [11]:
        print("Sugar beet")
        y_pred_word = "Sugar beet"

    return render_template("result.html",prediction = y_pred_word)

if __name__ == '__main__':
    app.run(debug=True)
