import streamlit as st
import xgboost as xgb
import json
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = xgb.XGBClassifier()
model.load_model("model/churn_model.json")

with open("model/feature_columns.json", "r") as f:
    feature_columns = json.load(f)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Customer Churn Prediction System")

st.markdown(
    """
    **Machine Learning powered customer retention analysis**

    Enter customer information below to estimate the probability
    that the customer may leave the service.
    """
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "77.68%")

with col2:
    st.metric("Precision", "56.51%")

with col3:
    st.metric("Recall", "84.62%")

with col4:
    st.metric("F1 Score", "67.76%")


st.divider()


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

with col2:

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

with col3:

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=840.0,
        step=10.0
    )


# ============================================================
# SERVICES
# ============================================================

st.header("📡 Services")

col1, col2, col3 = st.columns(3)

with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

with col3:

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )


# ============================================================
# ADDITIONAL SERVICES
# ============================================================

st.header("🛠️ Additional Services")

col1, col2, col3 = st.columns(3)

with col1:

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col2:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

with col3:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)


# ============================================================
# CONTRACT & PAYMENT
# ============================================================

st.header("💳 Contract & Payment")

col1, col2 = st.columns(2)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)

if predict_button:

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [
            1 if senior_citizen == "Yes" else 0
        ],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # --------------------------------------------------------
    # Encode input
    # --------------------------------------------------------

    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True
    )

    input_encoded = input_encoded.astype(int)


    # --------------------------------------------------------
    # Match model features
    # --------------------------------------------------------

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_encoded)[0]

    probability = model.predict_proba(
        input_encoded
    )[0][1]


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()

    st.header("📈 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error(
                "🔴 High Risk of Customer Churn"
            )

        else:

            st.success(
                "🟢 Customer Likely to Stay"
            )

    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    st.progress(
        float(probability)
    )


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if probability < 0.30:

        st.info(
            "🟢 Low churn risk — customer is relatively likely to stay."
        )

    elif probability < 0.60:

        st.warning(
            "🟡 Moderate churn risk — customer may require attention."
        )

    else:

        st.error(
            "🔴 High churn risk — consider customer retention action."
        )


# ============================================================
# ABOUT MODEL
# ============================================================

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        "This Customer Churn Prediction system uses an "
        "XGBoost machine learning classifier."
    )

    st.write(
        "The model was trained using the IBM Telco Customer "
        "Churn dataset."
    )

    st.write(
        "The application preprocesses customer information, "
        "converts categorical variables into numerical features, "
        "and generates a churn probability."
    )

    st.write("### Model Metrics")

    st.write("Accuracy: **77.68%**")
    st.write("Precision: **56.51%**")
    st.write("Recall: **84.62%**")
    st.write("F1 Score: **67.76%**")