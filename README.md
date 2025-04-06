# The News Lens 🔍

![image](https://github.com/user-attachments/assets/bec9aefb-f252-4f14-8594-25d6d40d8fe4)

This project is a Streamlit web application designed to streamline the process of researching information from multiple online sources. Users can provide a list of URLs, and the application will automatically download, process, and index the content using natural language processing techniques. The core functionality allows users to ask questions related to the content of these URLs and receive concise answers along with citations to the original sources, presented as clickable hyperlinks.

[Watch the Demo Video](https://drive.google.com/file/d/19jNEzjWcISo5qIwvfV456zyF1zjSgbMj/view?usp=sharing)

**Key Features:**

* **Automated Content Processing:** Downloads and parses text content from the provided URLs.
* **Intelligent Text Chunking:** Splits large documents into smaller, contextually relevant segments.
* **Semantic Indexing:** Creates a vector database using Hugging Face embeddings for efficient information retrieval.
* **Question Answering with Source Tracking:** Leverages a large language model (Flan-T5) via Langchain to answer user queries based on the indexed content.
* **Hyperlinked Source Citations:** Provides "Click Here" links to the specific URLs where the answer was found, enabling easy verification.
* **Clear User Interface:** Built with Streamlit for an intuitive and user-friendly experience, including progress indicators and clear output.

**How to Use:**

1.  Clone the repository: `git clone [repository URL]`
2.  Navigate to the project directory: `cd [repository directory]`
3.  Create a virtual environment (recommended): `python -m venv venv` and activate it.
4.  Install dependencies: `pip install -r requirements.txt`
5.  Add your HUGGINGFACEHUB_API_TOKEN in `.env` file before starting.
6.  Run the Streamlit application: `streamlit run app.py`
7.  In the Streamlit app, enter the URLs you want to analyze in the sidebar and click "Submit to process."
8.  Once the processing is complete, type your question in the main area and press Enter to get an answer with source links.

**Dependencies:**

* langchain
* langchain-community
* streamlit
* unstructured
* tiktoken (may be implicitly used by Langchain/Transformers)
* faiss-cpu
* python-magic (or libmagic)
* python_dotenv
* transformers
* langchain_huggingface
* sentence-transformers
* accelerate
    *(Refer to `requirements.txt` for specific versions)*

**Potential Future Enhancements:**

* Allowing users to upload files (PDFs, etc.).
* Summarization of the processed content.
* More control over the number of retrieved documents.
* Saving and loading the vector database for different sets of URLs.
