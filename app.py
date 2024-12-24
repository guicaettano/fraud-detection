from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Criar a aplicação Flask
app = Flask(__name__)

# Carregar os modelos treinados e colunas do dataset
rf_model = joblib.load("results/random_forest_model.pkl")
lr_model = joblib.load("results/logistic_regression_model.pkl")
columns = joblib.load("results/columns.pkl")

@app.route('/')
def home():
    return "<h1>API de Detecção de Fraude</h1><p>Envie uma requisição POST para /predict com os dados da transação.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter os dados JSON da requisição
        data = request.get_json()
        
        # Converter os dados para um DataFrame
        df = pd.DataFrame([data], columns=columns)
        
        # Fazer previsões com ambos os modelos
        rf_prediction = rf_model.predict(df)[0]
        rf_probability = rf_model.predict_proba(df)[0][1]

        lr_prediction = lr_model.predict(df)[0]
        lr_probability = lr_model.predict_proba(df)[0][1]

        # Retornar os resultados como JSON
        response = {
            "Random Forest": {
                "Prediction": int(rf_prediction),
                "Probability": rf_probability
            },
            "Logistic Regression": {
                "Prediction": int(lr_prediction),
                "Probability": lr_probability
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
