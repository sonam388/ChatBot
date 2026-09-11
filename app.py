import streamlit as st
import random
import joblib

model = joblib.load("Chatbot_model.pkl")
vetor = joblib.load("tfidf.pkl")
responce = joblib.load("responses.pkl")

st.title("My ChatBot")
st.write("Ask me Something ...")

if "message" not in st.session_state:
     st.session_state.message = []

for message in st.session_state.message:
     with st.chat_message(message['role']):
          st.write(message['content'])

user_input = st.chat_input("You : ")

if user_input:
     with st.chat_message("user"):
          st.write(user_input)
     st.session_state.message.append({
          "role":"user",
          "content":user_input
     })

     input_vector = vetor.transform([user_input]).toarray()
     predict = model.predict(input_vector)[0]
     if predict in responce:
        reply = random.choice(responce[predict])
        st.success("Bot :" + reply)
     else:
        st.error("Sorry , I don't Understand ")

    #  while st.chat_message("Bot"):
    #      st.write(reply)

     st.session_state.message.append({
         "role":"bot",
         "content":reply
    })