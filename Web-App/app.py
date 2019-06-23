from flask import Flask, render_template
import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from keras.models import model_from_json
from keras.optimizers import Adam

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('home.html')


def get_model():
    global model
    global loaded_model
    # loading saved model

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("amodel.h5")
    loaded_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy',
                         metrics=['accuracy'])
    print("Model Loaded")


def preprocess_image(image, target_size):

    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


print(" * Loading Keras model...")
get_model()


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(96, 96))

    predic = loaded_model.predict(processed_image).tolist()
    prediction = predic.argmax(axis=1)

    response = {
        'prediction':
             prediction


    }
    return jsonify(response)




if __name__ == '__main__':
    app.run()
