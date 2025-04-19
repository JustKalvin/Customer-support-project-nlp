import streamlit as st
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from langdetect import detect_langs
import base64


# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model dan tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./NLP_FINAL_MODEL.h5")

@st.cache_resource
def load_tokenizer():
    with open("./tokenizer.pkl", "rb") as handle:
        return pickle.load(handle)

@st.cache_resource
def load_tfidf():
    data = pd.read_csv('./Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data["instruction"])
    return vectorizer, data

# Load semua komponen
model = load_model()
tokenizer = load_tokenizer()
vectorizer, data = load_tfidf()
translator = Translator()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

encoded_intent = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund"
]


from pathlib import Path

# Load background image as base64
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Panggil fungsi ini dengan nama file gambarnya
set_bg_from_local("NLP_Streamlit_Background.jpeg")


# Preprocessing function
def preprocessing(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# TF-IDF response retrieval
def get_best_response(user_input, predicted_intent):
    filtered_data = data[data['intent'] == predicted_intent]
    if filtered_data.empty:
        return "I'm sorry, I don't understand your request."

    tfidf_matrix = vectorizer.transform(filtered_data["instruction"])
    user_input_tfidf = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)
    best_idx = similarities.argmax()

    return filtered_data.iloc[best_idx]["response"]

# Prediksi chatbot
def predict_text(text):
    try:
        lang_detect = detect_langs(text)
        lang_code = lang_detect[0].lang
        if lang_detect[0].prob < 0.9:
            lang_code = 'en'

        if lang_code != 'en':
            text = translator.translate(text, src=lang_code, dest="en").text

        preprocessed_text = preprocessing(text)
        sequence = tokenizer.texts_to_sequences([preprocessed_text])

        if not sequence or not sequence[0]:
            return "unknown_intent", "I'm a bit confused, please use another sentence."

        x = pad_sequences(sequence)
        prediction = model.predict(x)
        predicted_intent = encoded_intent[np.argmax(prediction[0])]
        response = get_best_response(text, predicted_intent)

        response_translated = translator.translate(response, src='en', dest=lang_code).text

        return predicted_intent, response_translated

    except Exception as e:
        return "unknown_intent", "Sorry, I can't understand what you typed."

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""
if "speech_to_text_done" not in st.session_state:
    st.session_state.speech_to_text_done = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ------------------------- Streamlit UI -------------------------
# Styling
st.markdown("""
    <style>
    .chat-container {
        max-width: 700px;
        margin: auto;
    }
    .chat-user {
        text-align: right;
        color: white;
        background-color: #0078FF;
        padding: 8px;
        border-radius: 10px;
        display: inline-block;
        max-width: 80%;
    }
    .chat-bot {
        text-align: left;
        color: black;
        background-color: #F0F0F0;
        padding: 8px;
        border-radius: 10px;
        display: inline-block;
        max-width: 80%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Ubah font seluruh aplikasi */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    font-size: 16px;
}

.stApp {
    background-color: rgba(255,255,255,0); 
}

.chat-container {
    max-width: 700px;
    margin: auto;
    padding: 20px;
}

/* Efek fade-in untuk bubble */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

.chat-user, .chat-bot {
    padding: 12px 18px;
    border-radius: 20px;
    margin: 10px 0;
    display: inline-block;
    max-width: 80%;
    animation: fadeIn 0.5s ease-in-out;
}

/* Chat bubble user */
.chat-user {
    text-align: right;
    background-color: #0078FF;
    color: white;
    margin-left: auto;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
}

/* Chat bubble bot */
.chat-bot {
    text-align: left;
    background-color: #f9f9f9;
    color: #333;
    margin-right: auto;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
}

/* Input field styling */
input[type="text"] {
    background-color: #1a1a1a
    padding: 10px;
    border: 2px solid #ffffff;
    border-radius: 6px;
    color: #ffffff;
}

/* Tombol rekam lebih modern */
button[kind="primary"] {
    background: linear-gradient(90deg, #0078FF 0%, #00C6FF 100%);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    transition: 0.3s;
}
button[kind="primary"]:hover {
    transform: scale(1.05);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)




st.markdown("<h1 style='text-align: center;'>🤖 AI Customer Support</h1>", unsafe_allow_html=True)

# Input area
user_input = st.text_input("Ask me anything...", key="chat_input", value=st.session_state.get("recognized_text", ""))

# Proses input pengguna (baik diketik atau dari ucapan)
if user_input:
    # Menampilkan pesan pengguna
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"<p class='chat-user'>{user_input}</p>", unsafe_allow_html=True)

    # Prediksi chatbot
    st.session_state.intent, st.session_state.response = predict_text(user_input)
    lines = st.session_state.response.replace("\n", "<br>")

    st.session_state.messages.append({"role": "bot", "content": st.session_state.response})
    st.markdown(f"<p class='chat-bot'>Intent : {st.session_state.intent} <br> {lines}</p>", unsafe_allow_html=True)
