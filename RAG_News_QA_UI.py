import os
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv
import pickle

load_dotenv()

st.title("News Research Tool 📈 📈")

st.sidebar.title("News Articles URLs")

urls=[]

main_placeholder=st.empty()

for i in range(3):
    url= st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

processed_url_clicked= st.sidebar.button("Process URLs")
file_path="vector_store.pkl"

## setting the model 
llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",task="text-generation")
model=ChatHuggingFace(llm=llm)

if processed_url_clicked:
    # load the data
    loader=UnstructuredURLLoader(urls=urls)

    main_placeholder.text("Data Loading...Started ✅✅")

    data=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,chunk_overlap=200)

    main_placeholder.text("Text Splitting...Started ✅✅")

    chunks=text_splitter.split_documents(data)

    embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    main_placeholder.text("creating vector store.... ✅✅")
    vector_store=FAISS.from_documents(chunks,embed_model)

    # storing vector_store create in local
    with open(file_path,"wb") as f:
        pickle.dump(vector_store,f)


    
query= main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):  # save the faiss vectorstore
        with open(file_path,"rb") as f:
            vector_store=pickle.load(f)

            retriever=vector_store.as_retriever(
            search_type="similarity",search_kwargs={"k":3})

            prompt=ChatPromptTemplate([
            ("system", """
            You are a research AI assistant. Answer the user’s question strictly based on the provided context. 
            Include all relevant details from the context, such as numbers, dates, events, or statements. 
            Do not provide information not present in the context. 
            If the context does not contain enough information to answer the question, reply with "I don't know".
            context:{context}"""),
            ("human","{question}")
            ])
            
            retrieved_docs=retriever.invoke(query) # saving retrieved docs for printing the source
            

            def format_docs(retrieved_docs):
                context= "\n\n".join([doc.page_content for doc in retrieved_docs])
                return context
        
            parallel_chain=RunnableParallel(
                {
            "question":RunnablePassthrough(),
            "context":retriever | RunnableLambda(format_docs)
                }
            )

            parser=StrOutputParser()

            main_chain= parallel_chain | prompt | model | parser

            result=main_chain.invoke(query)

            st.markdown("## Answer")

            st.write(result)
            l=[]
            for i,j in enumerate(retrieved_docs):
                # print(f"retrv_docs_{i+1}:{retrieved_docs[i].metadata['source']}")
                l.append(retrieved_docs[i].metadata['source'])
                # print("\n")
                unique_list=list(set(l))
                # print(unique_list)
            if unique_list:
                for k,l in enumerate(unique_list):
                    st.write(f"source: {l}")





