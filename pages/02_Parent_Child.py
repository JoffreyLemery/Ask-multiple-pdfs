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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


tools.set_page_config()
load_dotenv()
images_path = tools.get_images_path()


# def get_pdf_text(pdf_docs):
#     documents = []
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             documents += page
#             print('document : ', documents)
#         print('document : ', documents)
#     return documents

# def load_pdf_documents(pdf_docs):
#     loader = DirectoryLoader('', loader_cls=PyPDFLoader)
#     documents = []

#     for pdf in pdf_docs:
#         with pdf:
#             pdf_file = pdf.read()
#             loader.add_file(pdf.name, pdf_file)

#     return loader.load()

class Document:
    def __init__(self, page_content='', metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def load_pdf_documents(uploaded_file):
    
    if uploaded_file is not None:
        documents = []
        for pdf in uploaded_file:
            
            reader = PdfReader(pdf)
            meta = reader.metadata
            i = 0
            for page in reader.pages:
                print('\npage : ', page, '\n')
                documents.append(Document(page_content=page.extract_text(), metadata={'source': meta.author,'page':i}))
                i =+ 1
    return documents


def get_splitter():
    # This text splitter is used to create the parent documents - The big chunks
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    # This text splitter is used to create the child documents - The small chunks
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

    return parent_splitter, child_splitter


def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings) 
    store = InMemoryStore()
    return store, vectorstore

def retriever_parents(documents, vectorstore, store, child_splitter, parent_splitter):
    big_chunks_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    )
    big_chunks_retriever.add_documents(documents)
    return big_chunks_retriever

def get_conversation_chain(big_chunks_retriever):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=big_chunks_retriever,
        memory=memory
    )
    return conversation_chain


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

def display_page_title():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title('Retrieve information among many PDFs')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")


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
            print('documents : ', documents)

            # get the splitters
            child_splitter, parent_splitter = get_splitter()

            # get the store and vectorstore
            store, vectorstore = get_vectorstore()

            # get the retriever
            big_chunks_retriever = retriever_parents(documents, vectorstore, store, child_splitter, parent_splitter)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                big_chunks_retriever
                )

    st.header("Chat with multiple PDFs using Parent/Child relations :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()

tools.display_side_bar()