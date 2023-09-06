from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData



app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict/", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        data = CustomData(
            carat = float(request.form["carat"]),
            depth = float(request.form["depth"]),
            table = float(request.form["table"]),
            x = float(request.form["x"]),
            y= float(request.form["y"]),
            z = float(request.form["z"]),
            cut = request.form["cut"],
            color = request.form["color"],
            clarity = request.form["clarity"]
        )

        final_df = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_df)
        results = round(pred[0], 2)
        
        return render_template("result.html", result=results)



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port="8000")