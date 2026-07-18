import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Base64 or inline model loading safely
MODEL_PATH = "naive_model.pkl"

def get_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Professional Business-Meeting Ready HTML & CSS Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Analytics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --accent-color: #10b981;
            --accent-hover: #059669;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .dashboard-container {
            width: 100%;
            max-width: 550px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }

        .dashboard-header {
            padding: 32px 32px 24px 32px;
            border-bottom: 1px solid var(--border-color);
            text-align: center;
        }

        .dashboard-header h1 {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 6px;
        }

        .dashboard-header p {
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        .dashboard-body {
            padding: 32px;
        }

        .form-group {
            margin-bottom: 24px;
        }

        .form-group label {
            display: block;
            font-size: 0.8125rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .form-control {
            width: 100%;
            padding: 12px 16px;
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }

        .form-control:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15);
        }

        select.form-control {
            appearance: none;
            cursor: pointer;
        }

        .btn-submit {
            width: 100%;
            padding: 14px;
            background-color: var(--accent-color);
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        .result-box {
            margin-top: 28px;
            padding: 20px;
            border-radius: 8px;
            background-color: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.3);
            text-align: center;
        }
        
        .result-box.negative {
            background-color: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .result-title {
            font-size: 0.8125rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .result-value {
            font-size: 1.25rem;
            font-weight: 700;
        }
        
        .result-box.positive .result-value { color: #10b981; }
        .result-box.negative .result-value { color: #ef4444; }
    </style>
</head>
<body>

<div class="dashboard-container">
    <div class="dashboard-header">
        <h1>Predictive Business Intelligence</h1>
        <p>Naive Bayes Classification Model Deployment</p>
    </div>
    
    <div class="dashboard-body">
        <form method="POST" action="/">
            <div class="form-group">
                <label for="gender">Gender</label>
                <select name="gender" id="gender" class="form-control" required>
                    <option value="1" {% if inputs and inputs['gender'] == '1' %}selected{% endif %}>Male</option>
                    <option value="0" {% if inputs and inputs['gender'] == '0' %}selected{% endif %}>Female</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" name="age" id="age" class="form-control" placeholder="e.g., 35" min="0" value="{{ inputs['age'] if inputs else '' }}" required>
            </div>
            
            <div class="form-group">
                <label for="salary">Estimated Salary ($)</label>
                <input type="number" name="salary" id="salary" class="form-control" placeholder="e.g., 75000" min="0" value="{{ inputs['salary'] if inputs else '' }}" required>
            </div>
            
            <button type="submit" class="btn-submit">Execute Prediction</button>
        </form>

        {% if prediction is not none %}
            {% if prediction == 1 %}
            <div class="result-box positive">
                <div class="result-title">Model Target Output</div>
                <div class="result-value">Positive / Likely to Purchase (Class 1)</div>
            </div>
            {% else %}
            <div class="result-box negative">
                <div class="result-title">Model Target Output</div>
                <div class="result-value">Negative / Unlikely to Purchase (Class 0)</div>
            </div>
            {% endif %}
        {% endif %}
    </div>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    inputs = None
    
    if request.method == "POST":
        try:
            # Capturing inputs based on feature order: Gender, Age, EstimatedSalary
            gender = int(request.form.get("gender"))
            age = float(request.form.get("age"))
            salary = float(request.form.get("salary"))
            
            inputs = {
                "gender": request.form.get("gender"),
                "age": request.form.get("age"),
                "salary": request.form.get("salary")
            }
            
            # Formulate feature array matching the model structure
            features = np.array([[gender, age, salary]])
            
            # Inference
            model = get_model()
            prediction = int(model.predict(features)[0])
        except Exception as e:
            prediction = f"Error evaluating inputs: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction, inputs=inputs)

if __name__ == "__main__":
    # Bind to PORT provided by Render environment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
