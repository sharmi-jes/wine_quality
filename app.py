from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
# fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol  
    else:
        data=CustomData(
            fixed_acidity=request.form.get("fixed_acidity"),
            volatile_acidity=request.form.get("volatile_acidity"),
            citric_acid=request.form.get("citric_acid"),
            residual_sugar=request.form.get("residual_sugar"),
            chlorides=request.form.get("chlorides"),
            free_sulfur_dioxide=request.form.get("free_sulfur_dioxide"),
            total_sulfur_dioxide=request.form.get("total_sulfur_dioxide"),
            density=request.form.get("density"),
            pH=request.form.get("pH"),
            sulphates=request.form.get("sulphates"),
            alcohol=request.form.get("alcohol")


        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        # Predict the result using the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Making prediction...")
        results = predict_pipeline.predict(pred_df)
        print("Prediction made")

        # Return results to the home page
        return render_template('home.html', results=results[0])  # Show the first result
 



if __name__=="__main__":
    app.run(debug=True)






