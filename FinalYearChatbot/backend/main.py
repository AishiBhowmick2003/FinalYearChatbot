from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS

# Initialize FastAPI
app = FastAPI()

# Load Environment Variables (Hugging Face API Key)
HUGGINGFACE_API_KEY ="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" 
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment variables.")

# Define Pydantic Request Model
class QueryRequest(BaseModel):
    query: str

# Load FAISS Vector Store
DB_FAISS_PATH =r"C:\Users\bhowm\OneDrive\Documents\FinalYearChatbot\chatbot_model\db_faiss"  # Path to your FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index: {e}")

# Load LLM (initialize once)
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    task="text-generation",
    temperature=0.5,
    model_kwargs={"max_length": 1024},
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
)

# Create Conversational Chain
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    memory=memory,
)

# API Endpoint for Chatbot Query
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        response = qa_chain.invoke({"question": request.query})
        return {"response": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint
@app.get("/")
def root():
    return {"message": "✅ RAG Chatbot API is running!"}



