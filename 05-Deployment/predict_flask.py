
import pickle
from flask import Flask
from flask import request                   # since we will pass on the json file to the predict method and the output will also be json, and this contains requests to get the json and output json
from flask import jsonify                   # to convert a dictionary to json


model_input = 'model_C=1.0.bin'             # name of the model file

with open(model_input,'rb') as f_in:        #opening the file
    dv,model = pickle.load(f_in)            #Loading the dictionary vectorizer and model from the file

app = Flask('churn')                        #name of the app

@app.route('/predict',methods = ['POST'])    #the reason that we are using POST not GET is because we want to send some information about the customer and we can not do that using the GET method. And this customer information will be sent in a json format and the resopose out will also be jason                                              #json looks exactly like dictionary it is just that instead of single quotes we use double quotes.
def Predict():                              # customer will come as an input into this web serive as json
    
    customer = request.get_json()           #it will take in the data from the request and assume that it is json and parse it as a python dictionary
    
    X=dv.transform(customer)
    y_pred = model.predict_proba(X)[0,1]
    
    #prepare json to output the file
    result = {
        "churn_probability": float(y_pred),             # the reason float is used since to convert it to python float instead of np.float that is the format of y_pred as that will throw error in json
        "churn" : bool(y_pred>0.5)                      #Similar reason that we are converting it to python bool 
        }
    
    
    return jsonify(result)



if __name__ == "__main__":   # this main method is run only when it is run as a script or as python -m
    app.run(debug=True,host = '0.0.0.0',port = 9696)  #0.0.0.0 is also known as the localhost, also this thing should live inside the main method that is the top level script environment.