import os
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
load_dotenv()

PERSIST_DIRECTORY = "chroma_db"

if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

class RAGService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    def _get_vectorstore(self, collection_name: str):
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )

    def create_notebook(self, name: str):
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        client.get_or_create_collection(name)

    def list_notebooks(self) -> List[str]:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        return [c.name for c in client.list_collections()]

    def delete_notebook(self, name: str):
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            client.delete_collection(name)
        except Exception as e:
            print(f"Error deleting collection {name}: {e}")
            pass

    def add_document(self, notebook_name: str, file_path: str, original_filename: str):
        # Load
        if original_filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        docs = loader.load()
        
        # Add metadata
        for doc in docs:
            doc.metadata["source_name"] = original_filename
            doc.metadata["notebook"] = notebook_name

        # Chunk
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        print(f"Number of chunks: {len(chunks)}")

        # Add to Chroma
        vectorstore = self._get_vectorstore(notebook_name)
        vectorstore.add_documents(chunks)

        
    
    def list_resources(self, notebook_name: str):
        # let have persistent resource directory to update, delete from 
        
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            collection = client.get_collection(notebook_name)
            # Get all metadata (limit to avoid crash if huge, but for demo it's fine)
            result = collection.get(include=["metadatas"])
            metadatas = result["metadatas"]
            resources = set()
            for m in metadatas:
                if "source_name" in m:
                    resources.add(m["source_name"])
            return list(resources)
        except Exception:
            return []

    def delete_resource(self, notebook_name: str, resource_name: str):
        vectorstore = self._get_vectorstore(notebook_name)
        # Delete by metadata
        # LangChain Chroma wrapper doesn't expose delete by metadata easily?
        # Use client
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(notebook_name)
        collection.delete(where={"source_name": resource_name})

    @tool
    def query_notebook(self, notebook_name: str, query: str):
        """
        Searches the specified notebook in the vector database for relevant context 
        related to the query.

        Args:
            notebook_name: The name of the notebook collection to search.
            query: The user's question or search term.

        Returns:
            A formatted string containing the answer and source documents.
        """
        vectorstore = self._get_vectorstore(notebook_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        
        docs = retriever.invoke(query)
        
        

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        template = """("system", "You are an expert assistant that can use tools to answer questions. If you need information from a , use the `retrieve_from_notebook` tool. Otherwise, answer directly."),

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.format_messages(template)

        agent = create_retrieval_agent(llm, retriever)
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        

        chain = (
            {"context": lambda x: format_docs(docs), "question": lambda x: query}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(query)
        return {"answer": answer, "sources": [d.metadata.get("source_name", "unknown") for d in docs],"top3docs": [d.page_content for d in docs]}


tools = [
    query_notebook,
]