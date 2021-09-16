from predictionFile import Prediction
from trainQnAModel import ModelTraining
from wsgiref import simple_server
from flask import Flask, request, jsonify,render_template
import os
from flask_cors import CORS, cross_origin


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    data = request.json['data']
    user_story = data[0].split(' ')
    user_query = data[1].split(' ')
    predctnObj = Prediction()
    prediction = predctnObj.executeProcessing(user_story, user_query)
    return jsonify({ "Answer" : prediction })


if __name__ == "__main__":
    port = int(os.getenv("PORT"))
    host = '0.0.0.0'
    httpd = simple_server.make_server(host=host,port=port, app=app)
    httpd.serve_forever()
