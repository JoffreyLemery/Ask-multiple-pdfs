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
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
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
        st.title('Retrieve information among many PDFs with automatic tagging and Self-Query')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")

# Schema for tagging
schema_tagging = {
    "properties": {
        "author": {"type": "string",
                   "description": "Authors of the publication"},
        "key_word": {"type": "string",
                     "description": "key words of the documents, related to the purpose of the document"},
        "language": {"type": "string",
                     "description": "The language of the document",
                     "enum": ["spanish", "english", "french", "german", "italian"]},
        "year": {"type": "integer",
                 "description": "year of the publication of the article"}
    },
    "required": ["author", "key_word", "language", "year"]
}

#Metada for Self_Query - Must be aligned with tagging schema
metadata_field_info = [
    AttributeInfo(
        name="author",
        description="Authors of the publication",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="key_word",
        description="key words of the documents, related to the purpose of the document",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="language",
        description="The language of the document",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="year of the publication of the article",
        type="integer",
    ),

]
document_content_description = "Scientific article"

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
                print('\npage : ', page, '\n')
                documents.append(Document(page_content=page.extract_text(), metadata={'source': meta.author,'page':i}))
                i =+ 1
    return documents

#Establish conversation with the LLM
def get_conversation_chain(big_chunks_retriever, OPENAI_API_KEY):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=big_chunks_retriever,
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

#Return documents with metadata
def metadata_tagging(documents, OPENAI_API_KEY):
    llm_tagging = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    document_transformer = create_metadata_tagger(metadata_schema=schema_tagging, llm=llm_tagging)
    enhanced_documents = document_transformer.transform_documents(documents)
    return enhanced_documents

#Return the Self_Query retriever
def self_query_retriever(vectorstore, document_content_description, metadata_field_info, OPENAI_API_KEY):
    llm_retriever = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    retriever = SelfQueryRetriever.from_llm(
        llm_retriever,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True
        )
    return retriever

#Vectorize documents into a vectorial dB
def get_vectordb(OPENAI_API_KEY, texts):
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, request_timeout=60)
        vectordb = Chroma.from_documents(documents=texts, 
                            embedding=embedding)
        return vectordb

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
            #Tag documents
            documents = metadata_tagging(documents)
            #Split documents
            texts = split_text(documents)
            #Vectorize documents
            vectorstore = get_vectordb(texts)
            retriever = self_query_retriever(vectorstore=vectorstore,document_content_description = document_content_description, metadata_field_info = metadata_field_info )
    
            # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                retriever
                )

    st.header("Chat with multiple PDFs using Self-Queries :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()

tools.display_side_bar()