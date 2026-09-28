from fastapi import FastAPI 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from langchain.memory import ConversationSummaryMemory      # Adding Memory for context Retention
from datetime import datetime,timedelta



INACTIVITY_TIMEOUT = timedelta(minutes=5)  # Set timeout for removing inactive users

def remove_inactive_users():
    now = datetime.now()
    inactive_users = [user_id for user_id, details in connected_users.items()
                      if now - datetime.strptime(details["last_active"], "%Y-%m-%d %H:%M:%S") > INACTIVITY_TIMEOUT]
    
    for user_id in inactive_users:
        del connected_users[user_id]
        if user_id in user_memories:
            del user_memories[user_id]  # Remove memory for inactive users
        print(f"Removed inactive user: {user_id}")



import uuid

def generate_user_id():
    return str(uuid.uuid4())  # Generates a unique user_id


app = FastAPI()

# Allow frontend requests (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model to receive messages from frontend
class Message(BaseModel):
    text: str
    user_id : str

# Absolute paths for embedding storage
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")

# Tool 1.2
vectorstore2 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool2"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever2 = vectorstore2.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool2 = create_retriever_tool(retriever=retriever2,                           
                                       name="Udyami_Yojna_scheme1_SC_ST",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.3
vectorstore3 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool3"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever3 = vectorstore3.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool3 = create_retriever_tool(retriever=retriever3,                           
                                       name="Udyami_Yojna_scheme2_Extremely_Backward_Class",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.4
vectorstore4 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool4"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever4 = vectorstore4.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool4 = create_retriever_tool(retriever=retriever4,                           
                                       name="Udyami_Yojna_scheme3_YUVA",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.5
vectorstore5 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool5"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever5 = vectorstore5.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool5 = create_retriever_tool(retriever=retriever5,                           
                                       name="Udyami_Yojna_scheme4_Mahila",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.6
vectorstore6 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool6"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever6 = vectorstore6.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool6 = create_retriever_tool(retriever=retriever6,                           
                                       name="Udyami_Yojna_scheme5_Alpsankhyak",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Direct Gemini Tool
chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                              google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE")

@tool
def direct_llm_answer(query: str) -> str:
    """Directly generates an answer from the LLM."""
    response = chat.invoke(query)
    return response




tools = [retriever_tool2, retriever_tool3, retriever_tool4, retriever_tool5, retriever_tool6, direct_llm_answer]

chat_prompt_template = hub.pull("hwchase17/openai-tools-agent")

# Create tool-calling agent
agent = create_tool_calling_agent(llm=chat, tools=tools, prompt=chat_prompt_template)



# User specific memory storgae

user_memories = {}
connected_users = {}    # store user details


MAX_MEMORY_SIZE = 5  # Clear memory after 5 interactions



def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id] = {
            "memory" : ConversationSummaryMemory(llm=chat, memory_key="chat_history", return_messages=True),
            "interaction_count" : 0
        }
        # Logging User connection

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connected_users[user_id] = {
            "first_seen" : now,
            "last_active" : now,
            "total_messages" : 0
        }
    memory_data = user_memories.get(user_id)

     # Debug print
    print(f"Memory Data for User {user_id}: {memory_data}")
    
    return memory_data if memory_data is not None else {}

@app.post("/chat")
def chat_with_model(msg: Message):
    remove_inactive_users()
    user_id = msg.user_id.strip() if msg.user_id else generate_user_id()

    if user_id not in connected_users:  # New user check
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connected_users[user_id] = {
            "first_seen": now,
            "last_active": now,
            "total_messages": 0
        }

    user_data = get_user_memory(msg.user_id)
    memory = user_data["memory"]
    interaction_count = user_data["interaction_count"]

    # Updating Userlog activity
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connected_users[msg.user_id]["last_active"] = now
    connected_users[msg.user_id]["total_messages"] += 1
    
    # Reset memory if it gets too large
    if interaction_count >= MAX_MEMORY_SIZE:
        print(f'Memory Reset for {msg.user_id}')
        memory.clear()  # Clears stored history
        user_memories[msg.user_id] = {"memory": ConversationSummaryMemory(llm=chat, memory_key="chat_history", return_messages=True),
                                      "interaction_count" : 0}
        memory = user_memories[msg.user_id]["memory"]
        
    agent_executor = AgentExecutor(                                         # Reinitialized agent executor
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            memory=memory
        )
        

    response = agent_executor.invoke({"input": msg.text})
    user_memories[msg.user_id]["interaction_count"] += 1  # Increase interaction count

    # Print logs of all connected users
    print("\n--- Connected Users Log ---")
    for uid, details in connected_users.items():
        print(f"User ID: {uid}, First Seen: {details['first_seen']}, Last Active: {details['last_active']}, Total Messages: {details['total_messages']}")
    print("---------------------------\n")
    
    return {
        "user_id" : user_id,
        "response": response.get("output", "No response generated"),
        "intermediate_steps": response.get("intermediate_steps", [])
    }




# Version 0.2

""" Frontend Chnages:


Uses localStorage to persist user_id.

Sends an empty user_id initially, letting the backend generate it.

Stores the backend-generated user_id to avoid conflicts."""


#The frontend (React) checks for a stored user_id. If none exists, it sends null(enpty string) to the backend.

#The backend (FastAPI) either assigns a new user_id (if it's a new user) or uses the existing one.

#This user_id helps maintain conversation memory for personalized responses.

#The backend removes inactive users after 5 minutes of inactivity.


# see user handling page for more details