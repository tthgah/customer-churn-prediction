import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# --- Cấu hình giao diện ---
st.set_page_config(layout="wide")
st.title("📊 Customer Churn Analysis Dashboard")

# --- Load dữ liệu ---
@st.cache_data
def load_data():
    df = pd.read_csv("BankChurners.csv")
    df = df.drop(['CLIENTNUM'], axis=1)
    df['Attrition_Flag'] = df['Attrition_Flag'].map({
        'Existing Customer': 0,
        'Attrited Customer': 1
    })
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df

df = load_data()

# --- Hiển thị dữ liệu ---
st.subheader("📄 Bảng dữ liệu đầu vào")
st.dataframe(df.head())

# --- Biểu đồ 1: Phân phối khách hàng ---
st.subheader("📊 Biểu đồ phân phối trạng thái khách hàng")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='Attrition_Flag', ax=ax1)
ax1.set_title("Customer Churn Distribution")
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Existing', 'Attrited'])
st.pyplot(fig1)

# --- Biểu đồ 2: Phân phối giới tính ---
st.subheader("📊 Biểu đồ giới tính khách hàng")
fig2, ax2 = plt.subplots()
sns.countplot(data=df, x='Gender', hue='Attrition_Flag', ax=ax2)
ax2.set_title("Gender vs Churn")
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['F', 'M'])
st.pyplot(fig2)

# --- Biểu đồ 3: Heatmap tương quan ---
st.subheader("🔍 Ma trận tương quan giữa các biến")
fig3, ax3 = plt.subplots(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False, ax=ax3)
ax3.set_title("Correlation Matrix")
st.pyplot(fig3)

# --- Biểu đồ 4: Phân phối thu nhập ---
st.subheader("💰 Thu nhập và tỷ lệ rời bỏ")
fig4, ax4 = plt.subplots()
sns.boxplot(data=df, x='Attrition_Flag', y='Income_Category', ax=ax4)
ax4.set_title("Income vs Churn")
ax4.set_xticklabels(['Existing', 'Attrited'])
st.pyplot(fig4)

# --- Biểu đồ 5: Số lượng sản phẩm sử dụng ---
st.subheader("📦 Số sản phẩm sử dụng và tỷ lệ rời bỏ")
fig5, ax5 = plt.subplots()
sns.countplot(data=df, x='Total_Relationship_Count', hue='Attrition_Flag', ax=ax5)
ax5.set_title("Product Usage vs Churn")
st.pyplot(fig5)

# --- Biểu đồ 6: Tổng số dư thẻ ---
st.subheader("💳 Tổng số dư thẻ vs tình trạng rời bỏ")
fig6, ax6 = plt.subplots()
sns.kdeplot(data=df[df['Attrition_Flag'] == 0]['Total_Revolving_Bal'], label='Existing', shade=True)
sns.kdeplot(data=df[df['Attrition_Flag'] == 1]['Total_Revolving_Bal'], label='Attrited', shade=True)
ax6.set_title("Total Revolving Balance Distribution")
ax6.legend()
st.pyplot(fig6)
