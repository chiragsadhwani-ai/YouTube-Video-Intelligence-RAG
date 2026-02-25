import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

class LangChainAssistant:
    def __init__(self):
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initializes the HuggingFace Endpoint via LangChain."""
        endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=self.hf_token
        )
        return ChatHuggingFace(llm=endpoint)

    def create_vector_store(self, text: str):
        """LangChain pipeline: Split text and index into FAISS."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        return FAISS.from_texts(chunks, self.embeddings)

    def run_qa_chain(self, query: str, vector_db):
        """Executes a Retrieval-Augmented Generation (RAG) chain."""
        docs = vector_db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        template = "Context: {context}\n\nQuestion: {question}\nAnswer:"
        prompt = PromptTemplate.from_template(template)
        
        # Simple Chain execution
        formatted_prompt = prompt.format(context=context, question=query)
        return self.llm.invoke(formatted_prompt)