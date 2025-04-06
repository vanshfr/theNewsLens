import os
import streamlit as st
import pickle
import time
import langchain
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

st.title("The News Lens 🔍")
st.sidebar.title("URLs")
urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
submit = st.sidebar.button("Submit to process 🔌")
file_path = "vectorDB.pkl"
main_placeholder = st.empty()

st.markdown(
    """
    <style>
    body {
        background-color: #2b252a; /* Example light blue */
    }
    a {
        text-decoration: none !important;
        color: rgb(97 231 127) !important; /* Change to your desired color */
    }
    a:hover {
        color: rgb(239 247 241) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if submit:
    valid_urls = [url for url in urls if url]
    if not valid_urls:
        main_placeholder.warning("Please enter at least one URL.")
    else:
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        try:
            loader = UnstructuredURLLoader(urls=valid_urls)
            progress_text.text("Data Loading Started...")
            data = loader.load()
            progress_bar.progress(0.2)
            progress_text.text("Done...✅")
            time.sleep(0.5)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            progress_text.text("Almost Done...")
            docs = text_splitter.split_documents(data)
            progress_bar.progress(0.5)
            progress_text.text("Text Splitting Completed...✅")
            time.sleep(0.5)

            embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
            progress_text.text("Please Wait...")
            vectorstore = FAISS.from_documents(docs, embeddings_model)
            progress_bar.progress(0.8)
            progress_text.text("Please Wait...")
            time.sleep(0.5)


            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)
            progress_bar.progress(1.0)
            progress_text.success("URLs processed ✅")
            time.sleep(1)
            progress_bar.empty()  
            progress_text.empty() 
        except Exception as e:
            progress_bar.empty()
            progress_text.error(f"An error occurred during processing: {e}")

query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                model_kwargs={"temperature": 0.9, "max_length": 500},
                task="text-generation"
            )

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)
            with st.spinner("Getting the answer..."):
                result = chain({"question": query}, return_only_outputs=False)

            with st.container():
                st.subheader("Here's what I got..")
                st.write(result["sources"])
                sources = result['source_documents'][0].metadata['source']
                if sources:
                    st.subheader("Sources:") 
                    sources_list = sources.split("\n") 
                    for source in sources_list: 
                        #st.write(source)
                        st.markdown(f"- [Click here to proceed 🔗]({source})")
    
        except FileNotFoundError:
            st.error("Error Encountered. Please try again.")
        except Exception as e:
            st.error(f"An error occurred during query processing: {e}")
    else:
        st.warning("Please submit URLs first.")