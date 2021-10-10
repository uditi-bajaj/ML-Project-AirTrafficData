import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('webpage.html')

    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = np.round(prediction[0], 2)
    return render_template('webpage.html', prediction_text='Adjusted passenger count is :{}'.format(output))

if __name__=="__main__":
    app.run(debug=True)



    