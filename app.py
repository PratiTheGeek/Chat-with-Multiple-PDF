import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai #for ai model
from langchain.vectorstores import FAISS#vector database
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv() #to see the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)#reading pdf ke saare pages
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text) :
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlalp=1000)
    chunks=text_splitter.split_text(text) 
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings) #text chainks ki embedding store kar li
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""
Answer the question a sdetailed as possible from the provided context,make sure to provide all the details
and if the answer is no, just say it is not available in the context. Don't give a wrong answer\n\n
Context:\n{context}\n
Question:\n{question}\n

Answer:
"""
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"]) #importing prompt templatef rom langchain
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain
#defining function to get user input
def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db=FAISS.load_local("faiss_index",embeddings)#loading faiss database
    docs=new_db.similarity_search(user_question)

    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs=True

    )

    print(response)
    st.write("Reply: ",response["output_text"])

def main():
    st.set_page_config("chat with multiple pdf")
    st.header("Chat with Multiple PDF using Gemini")
    user_question=st.text_input("Ask a question from the PDF Files")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("MEnu: ")
        pdf_docs=st.file_uploader("Upload your PDF Files and click on the generate  button")
        if st.button("Generate"):
            with st.spinner("Processing....."):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.sucess("Done")

if __name__=="__main__":
    main()







