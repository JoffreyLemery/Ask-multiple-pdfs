from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
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
                    text+=page.page_content
                    text= text.replace('\t', ' ')
                    i =+ 1
    return documents, text


#Split the documents into smaller objects.
def split_text(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=3000,
            chunk_overlap=300
        )
        texts = text_splitter.create_documents([text])
        return texts

def prompt_refine_prompt():
    
    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary with key learnings\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with detailed context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary"
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    
    return prompt, refine_prompt

def chain(llm,refine_prompt, texts, prompt):
    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
    
    )
    return refine_chain

def loads_pdfs(pdfs):
    texts_list = []
    for pdf_file in pdfs:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        text=""
        dict={}
        pdf_name = os.path.basename(pdf_file.name)
        for page in docs:
            text+=page.page_content
        text= text.replace('\t', ' ')
        dict.update({'title': pdf_name})
        dict.update({'text': text})

        texts_list.append(dict)

        # Delete the temporary file
        os.remove(temp_path)
    return texts_list


def main():
    load_dotenv()
    llm = OpenAI(temperature=0)
    display_page_title()
    st.write(css, unsafe_allow_html=True)

    st.header("📕📘 Your documents 📙📗")
    pdf_files = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            prompt, refine_prompt = prompt_refine_prompt()
            display_page_results()
            texts_list = loads_pdfs(pdf_files)
            print("\n\nDocs : ", texts_list,"\n\n")
            for i, text in enumerate(texts_list):
                texts =  split_text(text['text'])
                refine_chain = chain(llm, refine_prompt, texts, prompt)
                refine_outputs = refine_chain({'input_documents': texts})
                summary = refine_outputs['output_text']
                st.markdown("""---""")
                st.write(f"Summary for PDF {i+1} : ", text['title'])
                st.write(summary)

if __name__ == '__main__':
    main()

tools.display_side_bar()