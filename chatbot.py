# chatbot_app.py
import streamlit as st
import random
import json
import pickle
from nltk_utils import tokenize, bag_of_words

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

# Load the trained model
model, all_words, tags = pickle.load(open("model.pkl", "rb"))

# Function to get a response from the model
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    prediction = model.predict([X])[0]
    tag = tags[prediction]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

# Streamlit UI setup
st.set_page_config(page_title="Banking Chatbot", page_icon="🏦")
st.title("🏦 Banking Chatbot")
st.markdown("Ask me anything related to banking services like accounts, loans, cards, etc.")

# Maintain chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("BankBot", response))

# Display chat
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)

st.markdown("---")
st.caption("Built using NLTK, Scikit-learn, and Streamlit")
