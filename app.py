import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer directly inside the app
MODEL_PATH = "Misinformation_Codes.ipynb/"  # Ensure your model files are inside the 'model/' folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Define label categories
label_mapping = ["False", "Half-True", "Mostly-True", "True", "Barely-True", "Pants-on-Fire"]

# Streamlit UI
st.title("📰 AI-Powered Fake News Detector")
st.write("Enter a news statement below to check its credibility.")

# Input box
statement = st.text_area("Enter a news statement:", "")

if st.button("Detect Fake News"):
    if statement:
        # Tokenize input text
        inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True, max_length=256)

        # Run model inference
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        # Display result
        st.subheader(f"Prediction: {label_mapping[prediction]}")
    else:
        st.warning("Please enter a news statement!")
