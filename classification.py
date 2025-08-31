import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(df.iloc[:, :-1], df['species'])

# Sidebar inputs
st.sidebar.title("🌸 Input Features")
sepal_length = st.sidebar.slider("Sepal length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width  = st.sidebar.slider("Sepal width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width  = st.sidebar.slider("Petal width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)
predicted_species = target_names[prediction[0]]

# Output
st.subheader("🌼 Prediction Result")
st.success(f"The predicted species is: **{predicted_species}**")

# Probability chart
st.subheader("📊 Prediction Probabilities")
prob_df = pd.DataFrame(prediction_proba, columns=target_names)
st.bar_chart(prob_df.T)

# Feature Importance
st.subheader("🔎 Feature Importance (Model Explainability)")
fig, ax = plt.subplots()
ax.barh(df.columns[:-1], model.feature_importances_)
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.pyplot(fig)

# Optional dataset preview
if st.checkbox("Show sample dataset"):
    st.dataframe(df.head())
