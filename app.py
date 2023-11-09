import os
import tempfile
import time

import streamlit as st
import requests
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.vectorstores import Vectara

load_dotenv()

CUSTOMER_ID = os.getenv("CUSTOMER_ID")
API_KEY = os.getenv("API_KEY")
CORPUS_ID = int(os.getenv("CORPUS_ID", 0))  # Assuming CORPUS_ID should be an integer
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE=os.getenv("OPENAI_API_BASE")

def initialize_vectara():
    vectara = Vectara(
        vectara_customer_id=CUSTOMER_ID,
        vectara_corpus_id=CORPUS_ID,
        vectara_api_key=API_KEY
    )
    return vectara

def get_knowledge_content(vectara, query, threshold=0.5):
    found_docs = vectara.similarity_search_with_score(
        query,
        score_threshold=threshold,
    )
    knowledge_content = ""
    for number, (score, doc) in enumerate(found_docs):
        knowledge_content += f"Document {number}: {found_docs[number][0].page_content}\n"
    return knowledge_content

def call_api(endpoint,prompt,knowledge):
    body = { "model": "codellama/CodeLlama-34b-Instruct-hf","messages": [{"role": "system", "content": f"You are 'Onco Wise', a research tool to fetch the latest research data on diseases. You must state your sources for every answer. It is crucial that you do this. The additional knowledge you need to answer the question is right here: {knowledge}"},{"role": "user", "content": f"{prompt}"} ],"temperature": 0.5}
    headerList = {"Content-Type": "application/json" , "Authorization": f"Bearer {OPENAI_API_KEY}"}
    response = requests.post(url=endpoint,json=body,headers=headerList)
    return response.json()

# with st.sidebar:
#     st.header("Configuration")
#     uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
#     customer_id = st.text_input("Vectara Customer ID", value=CUSTOMER_ID)
#     api_key = st.text_input("Vectara API Key", value=API_KEY)
#     corpus_id = st.text_input("Vectara Corpus ID", value=CORPUS_ID)
#     openai_api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY)
#     openai_api_base = st.text_input("OpenAI API Base", value=OPENAI_API_BASE)
#     submit_button = st.button("Submit")

st.title("Onco Wise")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

vectara_client = initialize_vectara()

#print(CUSTOMER_ID)

if user_input := st.chat_input("Enter your issue:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    knowledge_content = get_knowledge_content(vectara_client, user_input)
    print("__________________ Start of knowledge content __________________")
    print(f"You are 'Onco Wise', a research tool to fetch the latest research data on diseases. You must provide in text citations for your sources for every answer. It is crucial that you do this. The additional knowledge you need to answer the question is right here: {knowledge_content}")
    response = call_api(f"{OPENAI_API_BASE}/chat/completions",user_input,knowledge_content)
    full_response = response['choices'][0]['message']['content']
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
