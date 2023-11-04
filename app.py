from flask import Flask, render_template, request
import numpy as np

import pickle

import pandas

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('page1.html')


@app.route('/predict1', methods=['POST'])
def predict1():
    cgpa = float(request.form.get('cgpa'))
    iq = int(request.form.get('iq'))
    profile_score = int(request.form.get('profile_score'))

    # return "The cgpa is {}, the iq is {} and the profile_score {}".format(cgpa, iq, profile_score)

    res = model.predict(np.array([cgpa, iq, profile_score]).reshape(1,3))

    if res[0] == 1:
        res = 'placed'
    else:
        res = 'not placed'

    return render_template('page1.html', result=res)

if __name__ == '__main__':
    app.run(debug=True)
