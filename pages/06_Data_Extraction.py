from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
import langchain
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
import libs.tools as tools
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import PyPDFLoader
import tempfile
import os
from htmlTemplates import css, bot_template, user_template

tools.set_page_config()

# Set up OpenAI API
load_dotenv()

#Handle the title
def display_page_title():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title('Data Extraction')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")

def loads_pdfs(pdfs):
    docs_list = []
    for pdf_file in pdfs:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        dict={}
        pdf_name = os.path.basename(pdf_file.name)
        dict.update({'title': pdf_name})
        dict.update({'doc': docs})

        docs_list.append(dict)

        # Delete the temporary file
        os.remove(temp_path)
    return docs_list


def create_schema(properties):
    schema = {
        "properties": {},
        "required": []
    }

    for prop in properties:
        st.markdown("""- * - * -""")
        enum_values = []
        enum = False
        prop_type = st.selectbox(f"Select type for property '{prop}'", ["string", "integer", "array"])
        
        if prop_type == "array":
            item_type = st.selectbox(f"Select item type for array property '{prop}'", ["string", "integer"])
            item_description = st.text_input(f"Enter item description for array property '{prop}'")
            is_enum = prop_type = st.selectbox(f"Is there fixed values'", ["Yes", "No"], key=f"is_enum_{prop}")
            if is_enum == 'Yes':
                custom_enum_array = st.text_input("Enter available choice in the array (comma-separated)", key=f"custom_enum_array_{prop}")
                if custom_enum_array:
                    if st.button(f"Process_choices_{prop}"):
                        custom_enum_array = [prop.strip() for prop in custom_enum_array.split(",")]
                        enum_values = custom_enum_array + enum_values

                schema["properties"][prop] = {
                    "type": "array",
                    "items": {
                        "type": item_type,
                        "description": item_description,
                        "enum": enum_values
                    }
                }
            else:
                schema["properties"][prop] = {
                    "type": "array",
                    "items": {
                        "type": item_type,
                        "description": item_description
                    }
                }
        else:
            description = st.text_input(f"Enter description for property '{prop}'")
            is_enum = prop_type = st.selectbox(f"Is there fixed values'", ["Yes", "No"], key=f"is_enum_{prop}")
            if is_enum == 'Yes':
                custom_enum = st.text_input("Enter available choice (comma-separated)", key=f"custom_enum_{prop}")
                if custom_enum:
                    if st.button(f"Process_choices_{prop}"):
                        custom_enum = [prop.strip() for prop in custom_enum.split(",")]
                        enum_values = custom_enum + enum_values

                    schema["properties"][prop] = {
                        "type": prop_type,
                        "description": description,
                        "enum": enum_values
                    }
                else:
                    schema["properties"][prop] = {
                        "type": prop_type,
                        "description": description
                    }

        requiered = st.selectbox(f"Select if '{prop}' is requiered '{prop}'", ["Yes", "No"], key=f"required_{prop}")
        
        if requiered == 'Yes':
            schema["required"].append(prop)
    
    return schema

def splitter(doc):
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents_split = text_splitter.split_documents(doc)
    return documents_split

def sum_dictionaries(dictionaries):
    result = {}

    for dictionary in dictionaries:
        for key, values in dictionary.items():
            if key in result:
                if isinstance(result[key], set):
                    if isinstance(values, list):
                        result[key].update(values)
                    else:
                        result[key].add(values)
                else:
                    if isinstance(values, list):
                        result[key] = set(result[key] + values)
                    else:
                        result[key] = {result[key]} | {values}
            else:
                if isinstance(values, list):
                    result[key] = set(values)
                else:
                    result[key] = {values}

    return result

def main():
    load_dotenv()
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    batch_size = 3 

    display_page_title()
    st.write(css, unsafe_allow_html=True)
    with st.expander("Data to extract"):
        st.subheader("Additional properties 📈 ")
        st.markdown("""---""")

        # Example usage
        default_properties = ["author", "key_word", "language", "year"]
        st.write("Available default properties:", default_properties)
        avalaible_properties = default_properties
        # Allow user to add custom properties
        custom_properties = st.text_input("Enter custom properties (comma-separated)")
        if custom_properties:
            if st.button("Process"):
                custom_properties = [prop.strip() for prop in custom_properties.split(",")]
                st.write("Custom properties:", custom_properties)
                avalaible_properties = custom_properties + default_properties
        properties = st.multiselect("Select properties for the schema", avalaible_properties, default=avalaible_properties)
        st.markdown("""---""")
        st.subheader("Description of properties 📉")
        st.markdown("""---""")
        schema = create_schema(properties)
        st.json(schema)
    with st.expander("Upload documents"):
        pdf_files = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                docs_list = loads_pdfs(pdf_files)
                results = {}
                for i, doc in enumerate(docs_list):
                    documents_split =  splitter(doc['doc'])
                    extraction_chain = create_extraction_chain(schema, llm)
                    temp_result = []
                    for j in range(0, len(documents_split), batch_size):
                        
                        batch_docs = documents_split[i:i+batch_size]
                        batch_informations = extraction_chain.run(input=batch_docs)
                        for info in batch_informations:
                            temp_result.append(info)

                    dict_result = sum_dictionaries(temp_result)
                    results.update({doc['title']:dict_result})
                    st.markdown("""---""")
                    st.write(f"Data for PDF {i+1} : ", doc['title'], "\nData : ", results[doc['title']])


if __name__ == '__main__':
    main()

tools.display_side_bar()