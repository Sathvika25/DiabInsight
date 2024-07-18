import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(['Outcome'], axis=1)
    y = df.Outcome
    return df, X, y

df, X, y = load_data()

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion="entropy", max_depth=2)
model.fit(X_train, y_train)

# Prediction function
def predict_diabetes(features):
    prediction = model.predict([features])
    return "The person is likely to have diabetes" if prediction[0] == 1 else "The person is not likely to have diabetes"

# Streamlit app
st.markdown(
    """
    <style>
    .header {
        font-family: 'Arial Black', Gadget, sans-serif;
        font-size: 40px;
        color:purple;
    }
    .subheader {
        font-family: 'Comic Sans MS', cursive, sans-serif;
        font-size: 30px;
    }
    .text {
        font-family: 'Verdana', Geneva, sans-serif;
        font-size: 16px;
    }
    .result {
        font-family: 'Arial', sans-serif;
        font-size: 25px;
        font-weight: bold;
    }
    .card {
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-type1 {
        background-color: #E6F3FF;
        color: #003366;
    }
    .card-type1 .subheader {
        color: #003366;
    }
    .card-type2 {
        background-color: #FFF0E6;
        color: #663300;
    }
    .card-type2 .subheader {
        color: #663300;
    }
    .card-gestational {
        background-color: #F0FFE6;
        color: #336600;
    }
    .card-gestational .subheader {
        color: #336600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
page = st.sidebar.selectbox("", ["Prediction", "Education", "Data Visualization"])

if page == "Prediction":
    st.title('DiabInsight')
    st.markdown('<h1 class="header">Predict Diabetes</h1>', unsafe_allow_html=True)
    
    # Input fields
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose', min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input('Age', min_value=0, max_value=120, value=30)

    # Prediction button
    if st.button('Predict'):
        features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        result = predict_diabetes(features)
        st.markdown(f'<p class="result">{result}</p>', unsafe_allow_html=True)

elif page == "Education":
    st.markdown('<h1 class="header">Diabetes Education</h1>', unsafe_allow_html=True)
    
    # Type 1 Diabetes
    st.markdown(
        """
        <div class="card card-type1">
            <h2 class="subheader">Type 1 Diabetes</h2>
            <p class="text">Type 1 Diabetes is a chronic condition in which the pancreas produces little or no insulin. Insulin is a hormone needed to allow sugar (glucose) to enter cells to produce energy.</p>
            <h3 class="subheader">Causes</h3>
            <p class="text">The exact cause of Type 1 Diabetes is unknown. However, in most people with Type 1 Diabetes, the body's immune system mistakenly attacks and destroys insulin-producing beta cells in the pancreas. Genetic and environmental factors, such as viruses, might trigger the disease.</p>
            <h3 class="subheader">Risk Factors</h3>
            <ul class="text">
                <li>Family history: Having a parent or sibling with Type 1 Diabetes increases the risk.</li>
                <li>Genetic factors: Certain genes may be associated with an increased risk.</li>
                <li>Age: Although Type 1 Diabetes can appear at any age, it appears at two noticeable peaks. The first peak occurs in children between 4 and 7 years old, and the second is in children between 10 and 14 years old.</li>
            </ul>
            <h3 class="subheader">Prevention</h3>
            <p class="text">There is no known way to prevent Type 1 Diabetes. Current research is focused on finding ways to prevent the disease in those at high risk.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Type 2 Diabetes
    st.markdown(
        """
        <div class="card card-type2">
            <h2 class="subheader">Type 2 Diabetes</h2>
            <p class="text">Type 2 Diabetes is a chronic condition that affects the way your body metabolizes sugar (glucose). With Type 2 Diabetes, your body either resists the effects of insulin or doesn't produce enough insulin to maintain normal glucose levels.</p>
            <h3 class="subheader">Causes</h3>
            <p class="text">Type 2 Diabetes develops when the body becomes resistant to insulin or when the pancreas is unable to produce enough insulin. The exact reason for this is unknown, but genetic and environmental factors, such as being overweight and inactive, seem to be contributing factors.</p>
            <h3 class="subheader">Risk Factors</h3>
            <ul class="text">
                <li>Overweight or obesity: Excess weight is a primary risk factor for Type 2 Diabetes.</li>
                <li>Age 45 or older: The risk of Type 2 Diabetes increases with age.</li>
                <li>Family history of diabetes: Having a parent or sibling with Type 2 Diabetes increases the risk.</li>
                <li>Physical inactivity: The less active you are, the greater your risk of Type 2 Diabetes.</li>
                <li>History of gestational diabetes: Having gestational diabetes during pregnancy or giving birth to a baby weighing more than 9 pounds increases your risk.</li>
                <li>Ethnicity: Certain races and ethnicities, including Black, Hispanic, Native American, and Asian American, are at higher risk.</li>
            </ul>
            <h3 class="subheader">Prevention</h3>
            <ul class="text">
                <li>Maintaining a healthy weight.</li>
                <li>Being physically active.</li>
                <li>Eating a balanced, healthy diet.</li>
                <li>Quitting smoking.</li>
                <li>Regular blood sugar monitoring if you have risk factors.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Gestational Diabetes
    st.markdown(
        """
        <div class="card card-gestational">
            <h2 class="subheader">Gestational Diabetes</h2>
            <p class="text">Gestational Diabetes is a type of diabetes that can develop during pregnancy (gestation) in women who don't already have diabetes. It affects how your cells use sugar (glucose), leading to high blood sugar that can affect your pregnancy and your baby's health.</p>
            <h3 class="subheader">Causes</h3>
            <p class="text">During pregnancy, the placenta produces hormones that impair insulin function, leading to an increase in blood sugar. If the pancreas can't produce enough insulin to overcome the effect of these hormones, gestational diabetes results.</p>
            <h3 class="subheader">Risk Factors</h3>
            <ul class="text">
                <li>Age: Women older than 25 are at increased risk.</li>
                <li>Family or personal history: Your risk increases if you have prediabetes, a family history of diabetes, or if you've had gestational diabetes in a previous pregnancy.</li>
                <li>Weight: Being overweight before pregnancy increases the risk.</li>
                <li>Race: For reasons that aren't clear, women who are Black, Hispanic, American Indian, or Asian American are more likely to develop gestational diabetes.</li>
            </ul>
            <h3 class="subheader">Prevention</h3>
            <ul class="text">
                <li>Maintaining a healthy weight before pregnancy.</li>
                <li>Eating a balanced, healthy diet.</li>
                <li>Staying physically active before and during pregnancy.</li>
                <li>Regular prenatal care and following your doctor's advice on managing blood sugar levels during pregnancy.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "Data Visualization":
    st.markdown('<h1 class="header">Data Visualization</h1>', unsafe_allow_html=True)
    
    # Correlation heatmap
    st.markdown('<h2 class="subheader">Correlation Heatmap</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Feature correlation with outcome
    st.markdown('<h2 class="subheader">Feature Correlation with Diabetes Outcome</h2>', unsafe_allow_html=True)
    outcome_correlation = df.corr()['Outcome'].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    outcome_correlation.drop('Outcome').plot(kind='bar', ax=ax)
    plt.title('Correlation of Features with Diabetes Outcome')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
