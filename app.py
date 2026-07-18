from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("naive_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------------
# CHANGE THESE FEATURE NAMES ACCORDING TO YOUR DATASET
# -------------------------------------------------------
FEATURES = [
    "Feature 1",
    "Feature 2",
    "Feature 3",
    "Feature 4"
]
# -------------------------------------------------------

HTML = """

<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="UTF-8">
<title>Naive Bayes Prediction System</title>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:'Segoe UI',sans-serif;
}

body{

background:linear-gradient(135deg,#0f172a,#1e293b,#334155);
height:100vh;
display:flex;
justify-content:center;
align-items:center;

}

.container{

width:900px;
background:white;
border-radius:20px;
overflow:hidden;
display:flex;
box-shadow:0px 20px 50px rgba(0,0,0,.35);

}

.left{

width:40%;
background:linear-gradient(180deg,#0F766E,#115E59);
color:white;
padding:50px;

display:flex;
flex-direction:column;
justify-content:center;

}

.left h1{

font-size:32px;
margin-bottom:20px;

}

.left p{

line-height:28px;
font-size:16px;
opacity:.9;

}

.right{

width:60%;
padding:45px;

}

.right h2{

color:#0F172A;
margin-bottom:30px;

}

.input-group{

margin-bottom:18px;

}

label{

display:block;
margin-bottom:7px;
font-weight:600;
color:#334155;

}

input{

width:100%;
padding:14px;
border-radius:10px;
border:1px solid #CBD5E1;
font-size:15px;

}

input:focus{

outline:none;
border:1px solid #0F766E;

}

button{

width:100%;
padding:15px;
background:#0F766E;
border:none;
color:white;
font-size:18px;
font-weight:bold;
border-radius:10px;
cursor:pointer;
transition:.3s;

}

button:hover{

background:#115E59;
transform:scale(1.02);

}

.result{

margin-top:25px;
padding:18px;
background:#ECFDF5;
border-left:6px solid #10B981;
border-radius:10px;
font-size:20px;
font-weight:bold;
color:#065F46;

}

.footer{

margin-top:25px;
text-align:center;
font-size:14px;
color:#94A3B8;

}

</style>

</head>

<body>

<div class="container">

<div class="left">

<h1>Business Intelligence</h1>

<p>

Professional Machine Learning Prediction Dashboard

built using Flask & Scikit-Learn.

Designed for business presentations and client demonstrations.

</p>

</div>

<div class="right">

<h2>Naive Bayes Prediction</h2>

<form method="POST">

{% for feature in features %}

<div class="input-group">

<label>{{feature}}</label>

<input
type="number"
step="any"
name="{{feature}}"
required>

</div>

{% endfor %}

<button type="submit">

Predict

</button>

</form>

{% if prediction %}

<div class="result">

Prediction : {{prediction}}

</div>

{% endif %}

<div class="footer">

Machine Learning Deployment using Flask + Render

</div>

</div>

</div>

</body>

</html>

"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        values = []

        for feature in FEATURES:
            values.append(float(request.form[feature]))

        data = np.array(values).reshape(1, -1)

        pred = model.predict(data)

        prediction = pred[0]

    return render_template_string(
        HTML,
        features=FEATURES,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
