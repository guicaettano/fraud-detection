#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib



# Criar pasta para salvar resultados
os.makedirs("results", exist_ok=True)




train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')




X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']

X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']




rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)



lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)



rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

rf_auc = roc_auc_score(y_test, rf_probabilities)
rf_mse = mean_squared_error(y_test, rf_probabilities)
rf_mae = mean_absolute_error(y_test, rf_probabilities)




with open("results/random_forest_report.txt", "w") as f:
    f.write("Random Forest - Relatório de Classificação:\n")
    f.write(classification_report(y_test, rf_predictions))
    f.write("\nRandom Forest - Matriz de Confusão:\n")
    f.write(str(confusion_matrix(y_test, rf_predictions)))
    f.write(f"\nRandom Forest - AUC: {rf_auc:.4f}\n")
    f.write(f"Random Forest - MSE: {rf_mse:.4f}\n")
    f.write(f"Random Forest - MAE: {rf_mae:.4f}\n")





lr_predictions = lr_model.predict(X_test)
lr_probabilities = lr_model.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(y_test, lr_probabilities)
lr_mse = mean_squared_error(y_test, lr_probabilities)
lr_mae = mean_absolute_error(y_test, lr_probabilities)




with open("results/logistic_regression_report.txt", "w") as f:
    f.write("Regressão Logística - Relatório de Classificação:\n")
    f.write(classification_report(y_test, lr_predictions))
    f.write("\nRegressão Logística - Matriz de Confusão:\n")
    f.write(str(confusion_matrix(y_test, lr_predictions)))
    f.write(f"\nRegressão Logística - AUC: {lr_auc:.4f}\n")
    f.write(f"Regressão Logística - MSE: {lr_mse:.4f}\n")
    f.write(f"Regressão Logística - MAE: {lr_mae:.4f}\n")




rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probabilities)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probabilities)

plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.4f})", color='blue')
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.4f})", color='green')
plt.plot([0, 1], [0, 1], 'k--', label="Baseline")
plt.title('Curva ROC')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.legend()
plt.savefig("results/roc_curve.png")
plt.show()



feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importances.to_csv("results/feature_importances.csv", index=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Importância das Features - Random Forest')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.savefig("results/feature_importances.png")
plt.show()

rf_model = RandomForestClassifier()  # Carregue o modelo Random Forest treinado
lr_model = LogisticRegression()      # Carregue o modelo de Regressão Logística treinado
columns = ['feature1', 'feature2', 'feature3', 'feature4']  # Substitua pelos nomes das features

# Salve os modelos e as colunas
joblib.dump(rf_model, "results/random_forest_model.pkl")
joblib.dump(lr_model, "results/logistic_regression_model.pkl")
joblib.dump(columns, "results/columns.pkl")




