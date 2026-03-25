from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# load model
model = pickle.load(open("model/model_pipeline.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        try:
            age = float(request.form["Age"])
            income = float(request.form["MonthlyIncome"])
            job = float(request.form["JobLevel"])
            years = float(request.form["YearsAtCompany"])

            # pakai DataFrame biar aman
            data = pd.DataFrame([[age, income, job, years]],
                                columns=["Age", "MonthlyIncome", "JobLevel", "YearsAtCompany"])

            result = model.predict(data)

            prediction = "Yes (Attrition)" if result[0] == 1 else "No (Stay)"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("formprediction.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)