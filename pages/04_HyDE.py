import streamlit as st
from dotenv import dotenv_values
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import libs.tools as tools
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic, LLMChain, HypotheticalDocumentEmbedder
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

#Parameter & variables
tools.set_page_config()
load_dotenv()
images_path = tools.get_images_path()

#Handle the title
def display_page_title():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title('Retrieve information among many PDFs with using of HyDE')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")


#Class to construct documents in the same format as PDFLoader
class Document:
    def __init__(self, page_content='', metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

#Load through Streamlit
def load_pdf_documents(uploaded_file):
    if uploaded_file is not None:
        documents = []
        for pdf in uploaded_file:
            
            reader = PdfReader(pdf)
            meta = reader.metadata
            i = 0
            for page in reader.pages:
                documents.append(Document(page_content=page.extract_text(), metadata={'source': meta.author,'page':i})) 
                i =+ 1
    return documents

#Establish conversation with the LLM
def get_conversation_chain(retriever):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

#Handle the print of the conversation
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def query_embedding():
    multi_llm = OpenAI(n=4, best_of=4)
    embeddings = OpenAIEmbeddings()
    HDE = HypotheticalDocumentEmbedder.from_llm(multi_llm, embeddings, "web_search")
    return HDE 

#

#Vectorize documents into a vectorial dB
def get_retriever(HDE, texts):
        vectorstore = FAISS.from_documents(texts, HDE)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        return retriever


#Split the documents into smaller objects.
def split_text(documents):
        #Define splitter
        splitter = RecursiveCharacterTextSplitter()
        #Split documents
        texts = splitter.split_documents(documents)
        return texts

def main():
    load_dotenv()
    display_page_title()
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("📕📘 Your documents 📙📗")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            documents = load_pdf_documents(pdf_docs)
            #Split documents
            texts = split_text(documents)
            HDE = query_embedding()
            retriever = get_retriever(HDE, texts)
            # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                retriever=retriever
                )

    st.header("Chat with multiple PDFs using HyDE :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()

tools.display_side_bar()