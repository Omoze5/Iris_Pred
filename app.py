from flask import Flask, url_for, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('Flower.html')

@app.route('/submit', methods = ['POST', 'GET'])
def submit():
    feature = [float(x) for x in request.form.values()]
    float_feature = [np.array(feature)]
    prediction = model.predict(float_feature)

    return render_template('Flower.html', Prediction= "The flower class is {} " .format(prediction))






if __name__ == '__main__':
    app.run(debug=True)
