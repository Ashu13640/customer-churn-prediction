# 📊 Customer Churn Prediction

A machine learning web application that predicts whether a telecom customer is likely to churn based on their demographic information, services, contract, payment method, and account details.

The project uses **XGBoost** for classification and **Streamlit** for the interactive web application.

---

## 🚀 Live Demo



👉 **[Launch Customer Churn Prediction App](https://customer-churn-prediction-xxxxx.streamlit.app/)**

---

## 📌 Project Overview

Customer churn is an important business problem for subscription-based companies.

The goal of this project is to build a machine learning system that can identify customers who are at higher risk of leaving the service.

The application takes customer information as input and provides:

- Churn prediction
- Churn probability
- Risk level
- Customer retention insight

---

## 🎯 Problem Statement

The objective is to predict the `Churn` status of a customer:

- `0` → Customer stays
- `1` → Customer churns

The model is trained using the **IBM Telco Customer Churn dataset**.

---

## 📂 Dataset

The dataset contains information about telecom customers, including:

- Gender
- Senior citizen status
- Partner and dependents
- Tenure
- Phone service
- Internet service
- Online security
- Online backup
- Device protection
- Technical support
- Streaming services
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges
- Churn

The original dataset contains more than 7,000 customer records.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- XGBoost
- Streamlit
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Train/Test Split
   ↓
XGBoost Model
   ↓
Class Imbalance Handling
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Application