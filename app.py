import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load the trained model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("./results")  # Adjust the path based on your model

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
