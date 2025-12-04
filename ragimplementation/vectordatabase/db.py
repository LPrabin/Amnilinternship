from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader ,TextLoader, WebBaseLoader
import os
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever 
from langchain_chroma import Chroma


load_dotenv()

def load_pdfs(path: str) -> List[Document]:

    print (f"Loading PDFs from {path}")
    loader = PyPDFLoader(path)
    return loader.load()

def load_text(path: str) -> List[Document]:
    print (f"loading from text {path}")
    loader = TextLoader(path)
    return loader.load()

def load_web(path: str) -> List[Document]:
    print (f"loading from web{path}")
    loader = WebBaseLoader(path)
    return loader.load()




 #load based on type    
def load(source: str) -> List[Document]:
    all_documents: List[Document] = []
    if not os.path.isdir(source):
        raise ValueError(f"Source '{source}' is not a valid directory.")

    print(f"Loading documents from directory: {source}")
    for file_name in os.listdir(source):
        file_path = os.path.join(source, file_name)
        if os.path.isfile(file_path):
            print(f"Processing file: {file_path}")
            if file_name.lower().endswith(".pdf"):
                all_documents.extend(load_pdfs(file_path))
            elif file_name.lower().endswith((".txt", ".md", ".csv")):
                all_documents.extend(load_text(file_path))
        
            else:
                print(f"Skipping unsupported file type: {file_path}")
    return all_documents


def chunk_documents(docs: List[Document]) -> List[Document]:
    print("Chunking documents using semantic chunker- google embeddings model ")
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"Generated {len(chunks)} chunks from {len(docs)} documents")
    return chunks


def create_retriver(docs: List[Document]) -> BaseRetriever:
    print("creating hybrid retriver")
    # bm25_retriver = BM25Retriever.from_documents(docs)
    # bm25_retriver.k = 5

    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    vectorstore = Chroma.from_documents(
        documents = docs,
        embedding= embeddings,
        persist_directory="chroma_db",
        collection_name="hybrid_retriver"
    )
    vector_retriver = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    #ensemble
    
    return vector_retriver

def main(directroy: str):


    try:
        docs = load(directroy)
        chunks = chunk_documents(docs)
        retriver = create_retriver(chunks)

        query = "Human Brain Perspective and Neurophysiological Motivation"
        print(f"\n Query: {query}")

        result = retriver.invoke(query)
        print(f"\n Retrived: {len(result)} chunks.")

        for i, doc in enumerate(result):
            contentpreview = doc.page_content.replace('\n', ' ')
            print(f"{contentpreview}------")

    except Exception as e:
        print(f"Error: {e}")
    


if __name__ == "__main__":
 
    main("documents")