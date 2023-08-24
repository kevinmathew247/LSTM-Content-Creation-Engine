import os
from flask import render_template, request
import keras
import numpy as np
import pandas as pd
from app.__init__ import app
import wikipedia
from yaml import load, SafeLoader
import tensorflow as tf
from keras.layers import TextVectorization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# no secret key set yet
SECRET_KEY = os.urandom(32)
app.config["SECRET_KEY"] = SECRET_KEY

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/submit", methods = ['POST','GET'])
def predict_answer():
    #THINK ABOUT A WAY TO STORE INPUT TEXT

    # Get the input text
    input_text = request.form['textbox']

    model = keras.models.load_model('/Users/kevinmathew/Documents/Content-creation-engine/hub/examples/speech_training/toxicity.h5')

    df = pd.read_csv('/Users/kevinmathew/Documents/Dissertation/jigsaw-toxic-comment-classification-challenge/train.csv')
    X = df['comment_text']
    y = df[df.columns[2:]].values
    vectorizer = TextVectorization(max_tokens=200000,
                                output_sequence_length=1800,
                                output_mode='int')

    vectorizer.adapt(X.values)

    input_text_vectorized = vectorizer(input_text)
    # Make the prediction
    predictions_nested = model.predict(np.expand_dims(input_text_vectorized,0))
    # predictions_nested = model.predict(input_sequences)
    predictions = predictions_nested[0]

    # Print the predicted class label and probability
    class_names = ['Toxic', 'Severely Toxic','Obscene', 'Threat', 'Insult', 'Identity Hate']
    predictions_100 = [value * 100 for value in predictions]
    predictions_100 = [ '%.2f' % elem for elem in predictions_100 ]

    predictions_dict = {}
    for id, class_name in enumerate(class_names):
        predictions_dict[class_name] = predictions_100[id]

    predicted_class_index = np.argmax(predictions)     
    predicted_class_label = class_names[predicted_class_index]
    predicted_probability = predictions_dict[predicted_class_label]

    if(predicted_class_label == 'Toxic'):
        result_text = 'Your content has a very high level of toxicity.'
    if(predicted_class_label == 'Severely Toxic'):
        result_text = 'Your content is severely toxic and is dangerous to use.'
    if(predicted_class_label == 'Obscene'):
        result_text = 'Your content is obscene.'
    if(predicted_class_label == 'Threat'):
        result_text = 'Your content contains elements of threat.'
    if(predicted_class_label == 'Insult'):
        result_text = 'Your content contains insulting statements.'
    if(predicted_class_label == 'Identity hate'):
        result_text = 'Your content promotes identity hate and is dangerous to use.'
    if all(i <= 0.3 for i in predictions):
        safe_message = "No hate speech detected. Your content is safe to use"
        unsafe_message = ""
        result_text = ""
        return render_template(
            "result.html",
            predicted_class_label = predicted_class_label, predictions_dict = predictions_dict, safe_message = safe_message,
            unsafe_message = unsafe_message, result_text=result_text)
        
    else:
        unsafe_message = "Hate speech detected"
        safe_message =""

        return render_template(
            "result.html",
            predicted_class_label = predicted_class_label, predictions_dict = predictions_dict, safe_message = safe_message,
            unsafe_message = unsafe_message, result_text=result_text)

