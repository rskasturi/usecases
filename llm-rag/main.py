import os
import torch
import intel_extension_for_pytorch as ipex
import langchain
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.schema import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.cache import InMemoryCache
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Device Availability
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
if device.type == "xpu":
    torch.xpu.empty_cache()
    print(f"Using device: {torch.xpu.get_device_name()}")
else:
    print("Using CPU")

# Paths
ROOT_DIRECTORY = '<PWD_Path>' #Make sure to provide pwd path Ex: ~/path_to_folder/
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
pdf_file_path = f"{ROOT_DIRECTORY}/data/attention.pdf"
store = LocalFileStore("./db_cache/cache/")

# Embedding Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

# Hugging Face Model
MODEL_ID = "NousResearch/Llama-2-7b-chat-hf" #Can also try with different models Ex: "NousResearch/Hermes-2-Pro-Mistral-7B", 

# Generation config parameters
MAX_LENGTH = 4096
#MAX_LENGTH = 8096
TEMPERATURE = 0.1
DO_SAMPLE = False
REPETITION_PENALTY = 1.15
RETURN_FULL_TEXT = False 

# Session state management
session_store = {}
session_id = input("Session ID (press enter for default_session): ") or "default_session"

if session_id not in session_store:
    session_store[session_id] = ChatMessageHistory()

# Getting embeddings
def get_embeddings(device_type):
    if "instructor" in EMBEDDING_MODEL_NAME:
        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            embed_instruction="Represent the document for retrieval:",
            query_instruction="Represent the question for retrieving supporting documents:",
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
        )

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Loading the PDF
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()
if not docs:
    raise ValueError("No documents loaded from the PDF.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=240, chunk_overlap=50)
document_chunks = text_splitter.split_documents(docs)
documents = [Document(page_content=chunk.page_content) for chunk in document_chunks if chunk.page_content]

# Get embeddings
embeddings = get_embeddings(device)
embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, store, namespace=EMBEDDING_MODEL_NAME)

# DB creation
db = Chroma.from_documents(
    documents,
    embedder,
    persist_directory=PERSIST_DIRECTORY,
    client_settings=CHROMA_SETTINGS,
)

def hybrid_retrievers():
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder, client_settings=CHROMA_SETTINGS)
    db_retriever = db.as_retriever()
    
    # BM25 Retriever
    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(retrievers=[db_retriever, sparse_retriever], weights=[0.5, 0.5])
    return ensemble_retriever

def llm():
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.to(device)

    # Generation Config
    generation_config = GenerationConfig.from_pretrained(MODEL_ID)

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        device=device,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        #max_new_tokens=512,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        generation_config=generation_config,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    langchain.llm_cache = InMemoryCache()
    return HuggingFacePipeline(pipeline=pipe)

def history_aware_retriever():
    # Contextualize the question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_retriever = create_history_aware_retriever(llm(), hybrid_retrievers(), contextualize_q_prompt)
    return history_retriever

def qa_prompt():
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "You must answer questions strictly using the provided context. If you don't know the answer, say that you don't know. "
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return qa_prompt

def get_session_history(session: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

question_answer_chain = create_stuff_documents_chain(llm(), qa_prompt())
rag_chain = create_retrieval_chain(history_aware_retriever(), question_answer_chain)
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Main loop
while True:
    user_input = input("\nYour question (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    #print(response['answer'])
    print("\nAssistant:", response['answer'].split("Bot:")[-1].strip())
    print("\n\n")
    print("Chat History:", session_history.messages)
