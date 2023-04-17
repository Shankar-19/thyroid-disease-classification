from flask import Flask, render_template, request, url_for
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open('thyroid_1_model.pkl', 'rb'))
le = pickle.load(open("label_encoder.pkl", "rb"))

app = Flask(__name__)

# home page
@app.route("/")
@app.route("/home")
def home():
	return render_template("home.html")


# predict page
@app.route("/predict")
def formPage():
	return render_template("predict.html")

# submit page
@app.route("/submit", methods=['POST'])
def predict():
	
	goitre = request.form.get("goitre")
	tumor = request.form.get("tumor")
	hypopituitary = request.form.get("hypopituitary")
	psych = request.form.get("psych")
	TSH = request.form.get("TSH")
	T3 = request.form.get("T3")
	TT4 = request.form.get("TT4")
	T4U = request.form.get("T4U")
	FTI = request.form.get("FTI")
	TBG = request.form.get("TBG")

	x = [[float(goitre), float(tumor), float(hypopituitary), float(psych), float(TSH), float(T3), float(TT4), float(T4U), float(FTI), float(TBG)]]

	col = ['goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
	x = pd.DataFrame(x, columns=col)

	pred = model.predict(x)
	pred = le.inverse_transform(pred)

	return render_template("submit.html", result=str(pred))


# running flask app
if __name__ == "__main__":
	app.run(debug=True)