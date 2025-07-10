from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import streamlit as st
from datetime import datetime


SYSTEM_PROMPT = (
    f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.\n"
    "You are a teaching assistant solely for the science high school programming  course, which primarily focuses on learning python.\n"
    "Below is the  course schedule."
    "Note that Class with youtube link has already been done."
    "1st week, 5/25 (Sunday), 정렬, 선택정렬, 버블정렬, 탐색, 순차탐색, 이진탐색, 객체지향\n"
    "2nd week, 6/1 (Sunday), 정렬 비교, 반복문, 탐색 코드\n"
    "3rd week, 6/8 (Sunday), 알고리즘, 빅오표기법, Pandas\n"
    "4th week, 6/15 (Sunday), Pandas, 지역변수, 젼역변수, random\n"
    "5th week, 6/22 (Sunday), 삽입정렬, 재귀함수, 클래스, 코드업(self-number), matplotlib, folium, 인공지능 개념\n"
    "6th week, 6/29 (Sunday), FINAL about whole contents\n" 
    "Your duty is to assist students by answering any course-related questions.\n"
    "When responding to student questions, you may refer to the retrieved contexts.\n"
    "On top of each context, there is a tag (e.g., 25 한성 1-1 기말 1주차 강의자료.pdf) that indicates its source and week.\n"
    "For example, '25 한성 1-1 기말 1주차 강의자료.pdf' refers to the lecture material for the first week. \n"
    "You may choose to answer without using the context if it is unnecessary.\n"
    "However, if your answer is based on the context, you 'must' cite all the sources (noted at the beginning of each context) in your response such as 'Source : 25 한성 1-1 기말 1주차 강의자료.pdf"
    "Make sure to provide sufficient explanation in your responses.\n"
    "Context:\n"
)


def get_vector_store():
    # Load a local FAISS vector store
    vector_store = FAISS.load_local(
        "./faiss_db/", 
        embeddings = OpenAIEmbeddings(model = "text-embedding-3-large"), 
        allow_dangerous_deserialization = True)
    
    return vector_store



def get_retreiver_chain(vector_store):

    llm = ChatOpenAI(model = "o3-2025-04-16")

    faiss_retriever = vector_store.as_retriever(
       search_kwargs={"k": 5},
    )
    # bm25_retriever = BM25Retriever.from_documents(
    #    st.session_state.docs
    # )
    # bm25_retriever.k = 2

    # ensemble_retriever = EnsembleRetriever(
    #     retrievers = [bm25_retriever, faiss_retriever],
    # )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user","{input}"),
        ("user","Based on the conversation above, generate a search query that retrieves relevant information. Provide enough context in the query to ensure the correct document is retrieved. Only output the query.")
    ])
    history_retriver_chain = create_history_aware_retriever(llm, faiss_retriever, prompt)

    return history_retriver_chain




def get_conversational_rag(history_retriever_chain):
  # Create end-to-end RAG chain
  llm = ChatOpenAI(model = "o3-2025-04-16")

  answer_prompt = ChatPromptTemplate.from_messages([
      ("system",SYSTEM_PROMPT+"\n\n{context}"),
      MessagesPlaceholder(variable_name = "chat_history"),
      ("user","{input}")
  ])

  document_chain = create_stuff_documents_chain(llm,answer_prompt)

  conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)

  return conversational_retrieval_chain

