import streamlit as st
import joblib
import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# ✅ Check if the Logistic Regression Model file exists
if os.path.exists("lr_model.pkl"):
    lr_model = joblib.load("lr_model.pkl")
else:
    st.error("⚠️ Error: `lr_model.pkl` file not found! Please upload the trained model.")

# ✅ Define BERT+LSTM Model Class
class HybridBERTLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=6):
        super(HybridBERTLSTM, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        return self.fc(lstm_out[:, -1, :])

# ✅ Check if BERT+LSTM model file exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_lstm_model = HybridBERTLSTM().to(device)

if os.path.exists("bert_lstm_model.pth"):
    bert_lstm_model.load_state_dict(torch.load("bert_lstm_model.pth", map_location=device))
    bert_lstm_model.eval()
else:
    st.error("⚠️ Error: `bert_lstm_model.pth` file not found! Please upload the trained model.")

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ✅ Streamlit UI
st.title("📰 TrueTell: AI-Powered Misinformation Detector")
st.write("🔍 Enter a statement below to check its credibility.")

# ✅ User Inputs
model_choice = st.selectbox("Choose a model:", ["BERT+LSTM", "TF-IDF + Logistic Regression"])
user_text = st.text_area("📝 Enter a statement to analyze:")

if st.button("🔍 Analyze"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter a valid statement.")
    else:
        if model_choice == "BERT+LSTM":
            inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                output = bert_lstm_model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
            pred_label = torch.argmax(output, dim=1).item()
        else:
            pred_label = lr_model.predict([user_text])[0]

        # ✅ Display Prediction
        labels = ["False", "Half-True", "Mostly-True", "True", "Barely-True", "Pants-on-Fire"]
        st.success(f"✅ **Predicted Verdict:** {labels[pred_label]}")

st.markdown("---")
st.markdown("🔬 Developed with AI & Machine Learning | **TrueTell**")
