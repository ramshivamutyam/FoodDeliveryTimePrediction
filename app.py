from flask import Flask,render_template,request
from inference import preprocess,score


app=Flask(__name__)


@app.route("/")
def home():
    return render_template(template_name_or_list="index.html")

@app.route("/predict_api",methods=['POST'])
def predict_api():
    
    data=request.form.to_dict()
    list_values=preprocess(data)
    output=score(list_values)
    
    return render_template("index.html", prediction=round(output))

if __name__=='__main__':
    app.run(debug=True)