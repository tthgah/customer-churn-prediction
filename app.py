import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Giao diện Streamlit ---
st.title("Customer Churn Prediction - Decision Tree")
st.markdown("Ứng dụng này dự đoán khách hàng có rời bỏ ngân hàng hay không.")

# --- Load dữ liệu ---
@st.cache_data
def load_data():
    df = pd.read_csv("BankChurners.csv")
    df.drop(['CLIENTNUM'], axis=1, inplace=True)
    df['Attrition_Flag'] = df['Attrition_Flag'].map({
        'Existing Customer': 0,
        'Attrited Customer': 1
    })
    # Mã hóa các biến dạng object
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df

df = load_data()
st.subheader("Dữ liệu gốc")
st.write(df.head())

# --- Tách biến ---
X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Huấn luyện mô hình ---
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Hiển thị kết quả ---
st.subheader("Kết quả dự đoán trên tập kiểm tra")
st.write(f"🎯 Accuracy: `{accuracy:.4f}`")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# --- Dự đoán với dữ liệu người dùng ---
st.subheader("Dự đoán khách hàng mới")

user_input = X.columns
input_data = {}

for feature in user_input:
    input_data[feature] = st.number_input(f"{feature}", value=float(X[feature].mean()))

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

if st.button("Dự đoán"):
    if prediction == 1:
        st.error("⚠️ Dự đoán: Khách hàng có thể sẽ rời bỏ.")
    else:
        st.success("✅ Dự đoán: Khách hàng sẽ tiếp tục sử dụng dịch vụ.")
