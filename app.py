import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the Super Learner Model
model = joblib.load('super_learner_model.pkl')

# Define features and their levels
levels = {
    "Level 1": [
        "ag+1:629e", "feeling.nervous", "trouble.in.concentration",
        "having.trouble.in.sleeping", "social.media.addiction",
        "having.nightmares", "change.in.eating", "feeling.tired"
    ],
    "Level 2": [
        "sweating", "breathing.rapidly", "anger", "close.friend",
        "introvert", "feeling.negative", "avoids.people.or.activities",
        "blamming.yourself"
    ],
    "Level 3": [
        "hallucinations", "panic", "hopelessness",
        "suicidal.thought", "popping.up.stressful.memory"
    ]
}

st.title("Mental Health Assessment Chatbot")

responses = {}

# User Inputs
for level, features in levels.items():
    with st.expander(level):
        st.subheader(level)
        for feature in features:
            if feature == "ag+1:629e":
                name = st.text_input("What is your name?")
                age = st.number_input("How old are you?", min_value=0, max_value=120, step=1)
                responses["ag+1:629e"] = age
            else:
                user_response = st.selectbox(
                    f"Do you experience {feature.replace('.', ' ')}?",
                    options=["Select an option", "Yes", "No"],
                    key=feature
                )
                responses[feature] = 1 if user_response == "Yes" else (0 if user_response == "No" else None)

if st.button("Submit"):
    if None in responses.values():
        st.warning("Please answer all the questions before submitting.")
    else:
        st.write(f"Thank you, {name}.")
        input_data = pd.DataFrame([responses])

        prediction = model.predict(input_data)

        loaded_mapping = pd.read_csv('label_encoder_mappings.csv')
        encoder = LabelEncoder()
        encoder.classes_ = loaded_mapping['label'].values

        predicted_label = encoder.inverse_transform([prediction[0]])[0]
        st.write(f"Your responses are: {responses}")
        st.write(f"The model predicts: {predicted_label}")

# Optional: Show label mappings if needed
# st.write("Label Mappings:", loaded_mapping)
