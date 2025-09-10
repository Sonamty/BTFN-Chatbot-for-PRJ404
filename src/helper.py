import os
# os.chdir("../")
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

# Extract text from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf", #load all the pdf file
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata ={"source": src}
            )
        )
    return minimal_docs

# split the documents into smaller chunks
def text_splits(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20, #understand the context with this
        length_function=len
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks


def download_embeddings():
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
       #model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    return embeddings