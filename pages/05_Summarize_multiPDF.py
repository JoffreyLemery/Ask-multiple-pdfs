from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
import streamlit as st
import os
import tempfile
import libs.tools as tools
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
import libs.tools as tools
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from llama_hub.smart_pdf_loader import SmartPDFLoader
from langchain.text_splitter import TokenTextSplitter

tools.set_page_config()

# Set up OpenAI API
load_dotenv()

#Handle the title
def display_page_title():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title('Summarize multi-PDF')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")

def display_page_results():
    st.markdown("""---""")
    st.header('📑 Summarize of documents 📑')

class Document:
    def __init__(self, page_content='', metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def splitter(doc):
    text_splitter = TokenTextSplitter(chunk_size=5000, chunk_overlap=300)
    documents_split = text_splitter.split_documents(doc)
    return documents_split

def prompt_refine_prompt():
    
    prompt_template = """Write a concise and pedagogical summary of the following:
    {text}
    SUMMARY:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a clear and pedagogical summary with key learnings\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary by adding additional information"
        "(only if needed) with detailed context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, add information and refine the original summary."
        "If the context isn't useful, return the original summary."
        "Don't forget to ALWAYS FINISH your sentences."
    
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    
    return prompt, refine_prompt

def chain(llm,refine_prompt, prompt):
    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
    )
    return refine_chain

def to_langchain_document(page_content, metadata):
    document = Document(page_content=page_content, metadata=metadata)
    return document

def loads_pdfs(pdfs):
    docs_list = []
    # llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    # pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
    
    
    for pdf_file in pdfs:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            temp_path = temp_file.name
        # docs = pdf_loader.load_data(temp_path)
        pdf_loader = PyPDFLoader(temp_path)
        doc = pdf_loader.load()
        dict={}
        pdf_name = os.path.basename(pdf_file.name)
        dict.update({'title': pdf_name})
        dict.update({'doc': doc})
        docs_list.append(dict)
        # Delete the temporary file
        os.remove(temp_path)
    return docs_list
        


def main():
    load_dotenv()
    llm = OpenAI(temperature=0.0)
    display_page_title()
    st.write(css, unsafe_allow_html=True)
    batch_size = 3

    st.header("📕📘 Your documents 📙📗")
    pdf_files = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            prompt, refine_prompt = prompt_refine_prompt()
            display_page_results()
            docs = loads_pdfs(pdf_files)
            for i, doc in enumerate(docs):
                documents_split =  splitter(doc['doc'])
                for j in range(0, len(documents_split), batch_size):
                    batch_docs = documents_split[j:j+batch_size]
                    refine_chain = chain(llm, refine_prompt, prompt)
                    refine_outputs = refine_chain({'input_documents': batch_docs})
                    summary = refine_outputs['output_text']
                st.markdown("""---""")
                st.write(f"Summary for PDF {i+1} : ", doc['title'])
                st.write(summary)

if __name__ == '__main__':
    main()

tools.display_side_bar()