from fastapi import FastAPI 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain.tools.retriever import create_retriever_tool 

from langchain.tools import tool

from langchain.agents import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent 

import os
from datetime import datetime, timedelta
import uuid
from langchain.memory import ConversationBufferWindowMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from lingua import Language, LanguageDetectorBuilder
from langchain_community.embeddings import JinaEmbeddings

from langchain_ollama import ChatOllama



shared_embeddings = JinaEmbeddings(
    model_name="jina-embeddings-v2-base-de",
    jina_api_key=""
)



INACTIVITY_TIMEOUT = timedelta(minutes=4)

def remove_inactive_users():
    now = datetime.now()
    inactive_users = [user_id for user_id, details in connected_users.items()
                    if now - datetime.strptime(details["last_active"], "%Y-%m-%d %H:%M:%S") > INACTIVITY_TIMEOUT]
    for user_id in inactive_users:
        del connected_users[user_id]
        print(f"Removed inactive user: {user_id}")

def generate_user_id():
    return str(uuid.uuid4())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str
    user_id : str


def detect_language_and_set(msg_text):
    import joblib
    

    # Load the saved classifier
    model_path = os.path.join(BASE_DIR, "Embeddings", "en_hinglish_classifier.pkl")
    pipeline = joblib.load(model_path)

    # Predict using the classifier
    langs = pipeline.predict([msg_text])[0]
    print(f"Detected Language using classifier: {langs}")

    proba = pipeline.predict_proba([msg_text])[0]
    confidence = max(proba)
    print(f"Confidence Score : {confidence}")

    # If it's not Hinglish or low confidence, fallback to lingua
    if langs != 'hi_en' or confidence < 0.6:
        # Set up lingua
        lang_codes = {
            Language.ENGLISH: 'en',
            Language.HINDI: 'hi'
        }
        languages = list(lang_codes.keys())
        detector = LanguageDetectorBuilder.from_languages(*languages).build()

        # Detect language using lingua
        detected_lang = detector.detect_language_of(msg_text)
        print(f"Lingua detected language: {detected_lang}")

        best_lang = lang_codes.get(detected_lang, 'en')  # Default to 'en' if not found
    else:
        best_lang = 'hi'

    return best_lang



    
# def clarification(message: Message):
#     best_lang = detect_language_and_set(message.text)
    


# models/helpline_schema.py
from pydantic import BaseModel
from typing import Optional

class HelplineLog(BaseModel):
    name: Optional[str]
    application_id: Optional[str]
    mobile_number: Optional[str]
    issue: Optional[str]
    district : Optional[str]


# Step 1: Parser and prompt setup
log_parser = PydanticOutputParser(pydantic_object=HelplineLog)
format_instructions = log_parser.get_format_instructions()

prompt = PromptTemplate(
    template="""
Extract the following information from the user input:
- Name
- Application ID
- Mobile Number
- District
- Issue




Only extract what is present. Leave missing fields as null.
User Query: {user_input}

{format_instructions}
""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions}
)






BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")



# tool 001
vectorstore001 = FAISS.load_local(
    folder_path=os.path.join(EMBEDDINGS_DIR, "tool001"),
    embeddings=shared_embeddings,
    index_name="index",  # If your FAISS index filename is `index.faiss`
    allow_dangerous_deserialization=True  # required if index was saved with pickle
)
retriever001 = vectorstore001.as_retriever(search_type="similarity", search_kwargs={'k': 1})
retriever_tool001 = create_retriever_tool(retriever=retriever001,                           
                                    name="Udyami_Yojna_head",
                                    description=(
        f'''retriever_tool001 = create_retriever_tool(
    retriever=retriever001,
    name="Udyami_Yojna_head",
    description=(
        "You are an expert assistant for the Udyami Yojna scheme. "
        "Only answer questions that match the ones listed below. "
        "Do not answer anything beyond these questions, as other tools may handle them.\n\n"

        ✅ If the user greets you (e.g., 'hi', 'hello', 'namaste'), respond politely with a greeting and say depending on the input(if hindi respond in hindi, if english respond in english):\n"
        "' Hello and Namaste! I am the Udyami Yojna chatbot. You can ask me questions related to the Udyami Yojna scheme,and its sections.'\n\n"
        
        "⚠️ Allowed Questions (English):\n"
        "1. What is Mukhyamantri Udyami Yojna?\n"
        "2. What is the MMUY and BLUY portal link?\n"
        "3. What are the sections in Udyami Yojna?\n"
        "4. What are the 'khand' in Udyami Yojna?\n"
        "5. Who is involved in the leadership of Udyami Yojna?\n"
        "6. What are the departmental roles in Udyami Yojna?\n"
        "7. Which officials are involved in Udyami Yojna?/Officials involved in Udyami Yojna / Who is Nitish Kumar / Who is Nitish Mishra\n\n"
        "8. What State is Udyami Yojna for / which state started Udyami Yojna ?\n\n"
        
        "⚠️ Allowed Questions (Hinglish):\n"
        "1. mukhyamantri udyami yojna kya hai\n"
        "2. MMUY aur BLUY ke portal ka link kya hai?\n"
        "3. udyami yojna ke sections kya hai\n"
        "4. udyami yojna ke khand kya hai\n"
        "5. udyami yojna me leadership kaun hai\n"
        "6. udyami yojna me departmental roles kya hai\n"
        "7. udyami yojna me kaun kaun officials involved hai / Nitish Mishra Kaun hai / Nitish Kumar kaun hai ? 
        "8. Udymai Yojna kis state ke liye hai ? / kaun se state mein Udyami Yojna shuru hua ?
        
        "📌 Additional Instructions:\n"
        "- Do not answer names of officials unless explicitly asked.\n"
        "- Only respond to the exact scope above.\n"
        "- Let other tools handle unrelated or broader questions."

        If someone from other state ask if they can get loan, answer them directly without using other tools - "Only Permanent Residents of Bihar are eligible for this loan" or if language is hindi then "केवल बिहार के स्थायी निवासी ही इस ऋण के लिए पात्र हैं"

    )
)
'''))



# tool10
vectorstore10 = FAISS.load_local(
    folder_path=os.path.join(EMBEDDINGS_DIR, "tool10"),
    embeddings=shared_embeddings,
    index_name="index",  # If your FAISS index filename is `index.faiss`
    allow_dangerous_deserialization=True  # required if index was saved with pickle
)
retriever10 = vectorstore10.as_retriever(search_type="similarity", search_kwargs={'k': 20})
retriever_tool10 = create_retriever_tool(retriever=retriever10,                           
                                    name="Udyami_Yojna_section1_MMUY",
                                    description="You are an expert assistant for the Udyami Yojna scheme section1 MMUY. Using the information retrieved from your knowledge base, provide complete and accurate answers related to the Mukhyamantri Udyami Yojna, including but not limited to: scheme overview, projects/enterprises included(Only the projects which are explicitly mentioned in the documents are eligible, others are not), eligibility criteria (and questions like age limit), required documents, step-by-step application and selection process, financial assistance and benefits, fund disbursement, training and installment procedures, loan repayment guidelines, and any important conditions or restrictions. Also, accurately determine whether individuals from specific occupations, backgrounds, or categories (e.g., farmers, students, government employees, etc.) are eligible to apply, providing a clear explanation including any relevant conditions such as caste category, age, educational qualifications, or occupation-based restrictions. Summarize all relevant details concisely without omitting key points.It also deals with questions like if a certain individual can apply. List the names in a list format.")

# tool_table / tool11
vectorstore11 = FAISS.load_local(
    folder_path=os.path.join(EMBEDDINGS_DIR, "tool11"),
    embeddings=shared_embeddings,
    index_name="index",  # If your FAISS index filename is `index.faiss`
    allow_dangerous_deserialization=True  # required if index was saved with pickle
)
retriever11 = vectorstore11.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.7})
retriever_tool11 = create_retriever_tool(retriever=retriever11,                           
                                    name="MMUY_Project_list_with_fund",
                                    description="Use this tool when asked more information about projects to retrieve detailed information about various names, machinery specifications, quantities, production capacity per hour, estimated electricity load, shed preparation cost, cost of machinery, working capital, and total project cost.A project/industry/business is available ")




# Direct Gemini Tool (tool 9)
chat = ChatOllama(
    model = "qwen2.5:3b"
)
                                        # low for factual Output
# tool12

@tool
def helpline_query_logger(text: str) -> dict:
    """
    Use this tool when user's matter is not resolved, or when user types "Report an Issue","issue darj karo".
    Give this Format as example for them to see - "Enter your Name, ID/Mobile No. , District , Issue"
    If any information is still Null, only ask for that field (nothing else), saying Please Enter
    and logs it only when all are explicitly provided,
    Once all required message is available and message is logged say "Thank You, Your issue has been logged, and you will be contacted soon."
    """
    global current_user_id
    user_id = current_user_id
    if not user_id or user_id not in connected_users:
        return {"query_logged": False, "error": "Invalid or missing user_id"}

    try:
        if connected_users[user_id].get("helpline_logged") is True:
            return {
                "query_logged": True,
                "status": "Already logged",
                "message": "Thank you, your issue has already been logged, and you will be contacted soon."
            }

        chain = prompt | chat | log_parser
        try:
            result: HelplineLog = chain.invoke({"user_input": text})
        except Exception as e:
            return {
                "query_logged": False,
                "status": "Parsing failed. Please use the format: Name, ID/Mobile No., District, Issue (all in one line)",
                "error": str(e)
            }

        # Get or create buffer
        buffer = connected_users[user_id].get("helpline_log_buffer", {})

        # Update only if new non-null values are extracted
        if result.name and result.name.strip():
            buffer["Name"] = result.name.strip()

        if result.application_id and result.application_id.strip():
            buffer["Application ID"] = result.application_id.strip()

        if result.mobile_number and result.mobile_number.strip():
            buffer["Mobile Number"] = result.mobile_number.strip()

        if result.issue and result.issue.strip():
            buffer["Issue"] = result.issue.strip()

            # Validate the issue only if a new one was added
            issue_check_prompt = PromptTemplate.from_template(
                "Does the following text describe a valid user issue or request for help? Reply only 'yes' or 'no'.\n\nIssue: {text}"
            )
            is_valid_issue_chain = issue_check_prompt | chat | StrOutputParser()

            try:
                verdict = is_valid_issue_chain.invoke({"text": result.issue}).strip().lower()
                if verdict.startswith("yes"):
                    buffer["IssueConfirmed"] = True
            except Exception as e:
                print("Issue validation failed:", str(e))

        if result.district and result.district.strip():
            buffer["District"] = result.district.strip()




        # Save buffer
        connected_users[user_id]["helpline_log_buffer"] = buffer

        # Check for required info
        if (
            buffer.get("Name")
            and (buffer.get("Application ID") or buffer.get("Mobile Number"))
            and buffer.get("Issue")
            and buffer.get("District")
            and buffer.get("IssueConfirmed") is True
        ):
            row = {
                "Timestamp": datetime.now().isoformat(),
                "Name": buffer["Name"],
                "Application ID": buffer.get("Application ID"),
                "Mobile Number": buffer.get("Mobile Number"),
                "District" : buffer.get("District"),
                "Issue": buffer["Issue"],
            }

            log_path = "helpline_log.xlsx"
            if os.path.exists(log_path):
                df = pd.read_excel(log_path)
            else:
                df = pd.DataFrame(columns=row.keys())

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_excel(log_path, index=False)

            # Clear buffer after logging
            connected_users[user_id]["helpline_log_buffer"] = {}
            connected_users[user_id]["helpline_logged"] = True

            if "memory" in connected_users[user_id]:
                connected_users[user_id]["memory"].clear()

            return {
                "query_logged": True,
                "logged_name": row["Name"],
                "logged_application_id": row["Application ID"],
                "logged_mobile_number": row["Mobile Number"],
                "logged_issue": row["Issue"],
                "logged_district": row["District"],
                "logged_timestamp": row["Timestamp"],
                "status": "Successfully logged"
            }
            
        else:
            missing = []
            if not buffer.get("Name"):
                missing.append("Name")
            if not (buffer.get("Application ID") or buffer.get("Mobile Number")):
                missing.append("Application ID or Mobile Number")
            if not buffer.get("District"):
                missing.append("District")
            if not buffer.get("IssueConfirmed"):
                missing.append("Issue")

            return {
                "query_logged": False,
                "status": "Waiting for more information",
                "missing_fields": missing,
                "buffer": buffer
            }

    except Exception as e:
        return {
            "query_logged": False,
            "error": str(e)
        }




# tool13
vectorstore13 = FAISS.load_local(
    folder_path=os.path.join(EMBEDDINGS_DIR, "tool13"),
    embeddings=shared_embeddings,
    index_name="index",  # If your FAISS index filename is `index.faiss`
    allow_dangerous_deserialization=True  # required if index was saved with pickle
)
retriever13 = vectorstore13.as_retriever(search_type="similarity", search_kwargs={'k': 20})
retriever_tool13 = create_retriever_tool(retriever=retriever13,                           
                                    name="Udyami_Yojna_section2_BLUY",
                                    description= "You are an expert assistant for the Udyami Yojna scheme section2 BLUY.Only the projects/List of Activities which are explicitly mentioned in the documents are eligible, others are not")






# @tool
# def direct_llm_answer(query: str) -> str:
#     """Directly generates an answer from the LLM and only relevant."""
#     prompt = f"""
#     You are an assistant that only answers queries about government schemes of Bihar, India,
#     and only those queries whose information is not available in other tools.
#     Do not answer anything unrelated to Bihar schemes. If a question is unrelated, politely inform the user.

#     User question: {query}
#     """
#     response = chat.invoke(prompt)
#     return response

    

tools = [ retriever_tool001, retriever_tool10, retriever_tool11, helpline_query_logger, retriever_tool13]


AGENT_INSTRUCTIONS = """
You are a helpful assistant for the Udyami Yojna scheme.

1. If the user's question explicitly mentions either "MMUY" or "BLUY", do not ask for clarification,(Even if the `active_scheme`is none) Route the question to the corresponding sub-scheme tool and provide a direct answer.

2. If `active_scheme` is already set (i.e., not None), do not ask for clarification again. Use the active scheme to answer current and follow-up questions.

3. If the user's question is not about general Udyami Yojna, and it does not mention MMUY or BLUY, and active_scheme is not set, then do not make assumptions. Instead, ask the user:

- If the current language flow is English: ask "Could you please clarify which sub-scheme you're referring to under Udyami Yojna — MMUY or BLUY?"
- other wise: ask "कृपया स्पष्ट करें कि आप उद्यमी योजना के अंतर्गत किस उप-योजना का उल्लेख कर रहे हैं — MMUY या BLUY?"

This clarification should be asked **only once** — never again after scheme is selected.

4. Once any of the keywords are found, you must:
- Identify and store the selected scheme.
- Use it for the current and all future responses.
- Never ask the user to repeat the question.
- If the user had asked a question **before** selecting the scheme, use that previous question now.
- Do not switch schemes unless explicitly changed by the user using a valid keyword.

5. You must always invoke the appropriate tool to answer user queries. Do not answer directly using memory.

6. After checking in all tools,If the answer is not present in the database, say:
    -If the current language flow is English:
        "No such information is available. If you want us to contact you, please report an issue." 
    other wise:
        ""कोई ऐसी जानकारी उपलब्ध नहीं है। यदि आप चाहते हैं कि हम आपसे संपर्क करें, तो कृपया एक समस्या दर्ज करें।"
    —do not say "not available in the database" or just "no".    

7. Normalize semantically similar Hindi or English user inputs that differ only in tone, conjunctions, or auxiliary verbs (like "par", "hai", "please", etc.), and treat them as equivalent while determining intent.
    
8. You must always format your answers clearly. If the response contains more than one item, reason, step, or condition, use bullet points or numbered lists — never provide long, dense paragraphs. Always post-process the tool output if needed to apply this formatting.

Important: These rules override all other instructions and must be followed strictly at all times


"""

agent_prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "chat_history", "active_scheme"],
    template=AGENT_INSTRUCTIONS + """

Active Scheme: {active_scheme}

Chat History:
{chat_history}

User Input: {input}

{agent_scratchpad}
""",

)


from langchain.agents import initialize_agent, AgentType

agent = create_tool_calling_agent(
    llm=chat,
    tools=tools,
    prompt=agent_prompt_template
)

connected_users = {}
current_user_id = None  # global variable


@app.post("/chat")
def chat_with_model(msg: Message):
    global current_user_id
    remove_inactive_users()

    if not msg.user_id or not msg.user_id.strip():
        user_id = generate_user_id()
    else:
        user_id = msg.user_id.strip()

    current_user_id = user_id

    if user_id not in connected_users:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connected_users[user_id] = {
            "first_seen": now,
            "last_active": now,
            "total_messages": 0,
            "memory": ConversationBufferWindowMemory(k=4, return_messages=True, memory_key="chat_history", input_key="input"),
            "helpline_log_buffer": {},
            "active_scheme": None,
            "last_question": None,
            "last_bot_message": None
        }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connected_users[user_id]["last_active"] = now
    connected_users[user_id]["total_messages"] += 1

    best_lang = detect_language_and_set(msg.text)
    lang_map = {'en': 'English', 'hi': 'Hindi', 'hi_en': 'Hindi'}
    prompt1 = f"Please answer the following question in {lang_map[best_lang]}. User's question: {msg.text}"
    user_input_lower = msg.text.strip().lower()

    clarification_msgs = [
        "could you please clarify which sub-scheme you're referring to under udyami yojna — mmuy or bluy?",
        "कृपया स्पष्ट करें कि आप उद्यमी योजना के अंतर्गत किस उप-योजना का उल्लेख कर रहे हैं — mmuy या bluy?"
    ]

    last_bot_msg = connected_users[user_id].get("last_bot_message")
    run_scheme_logic = (last_bot_msg or "").strip().lower() in clarification_msgs

    if run_scheme_logic:
        last_question = connected_users[user_id].get("last_question")
        # Only process scheme logic if clarification message was last sent
        if user_input_lower in ["mmuy", "bluy"]:
            connected_users[user_id]["active_scheme"] = user_input_lower.upper()
        elif ("bluy" in user_input_lower or "बिहार लघु उद्यमी योजना" in user_input_lower or "for बिहार लघु उद्यमी योजना" in user_input_lower or "for बिहार लघु उद्यमी योजना (bluy)" in user_input_lower):
            connected_users[user_id]["active_scheme"] = "BLUY"
        elif ("mmuy" in user_input_lower or "मुख्यमंत्री उद्यमी योजना" in user_input_lower or "for मुख्यमंत्री उद्यमी योजना" in user_input_lower or "for मुख्यमंत्री उद्यमी योजना (mmuy)" in user_input_lower):
            connected_users[user_id]["active_scheme"] = "MMUY"
            
            
        # ✅ After setting active scheme, re-use last question if available
        if connected_users[user_id]["active_scheme"]:
            last_question = connected_users[user_id].get("last_question")
            if last_question:
                msg.text = last_question + "in" + connected_users[user_id]["active_scheme"]
                connected_users[user_id]["last_question"] = None
            else:
                msg.text = "Give me a brief of this scheme."

            connected_users[user_id]["last_bot_message"] = None
        else:
            clarification_message = clarification_msgs[0] if best_lang != "hi" else clarification_msgs[1]
            connected_users[user_id]["last_bot_message"] = clarification_message
            connected_users[user_id]["last_question"] = msg.text 
            return {
                "user_id": user_id,
                "response": clarification_message,
                "intermediate_steps": [],
                "helpline_log": None
            }

    # Else: No scheme logic – just proceed normally
    
    connected_users[user_id]["last_question"] = msg.text
    



    msg.text = f"Please answer the following question in {lang_map[best_lang]}. User's question: {msg.text}"

    print(msg.text)
    print(connected_users[user_id]["active_scheme"])

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=connected_users[user_id]["memory"],
        return_intermediate_steps=True,
    )

    response = agent_executor.invoke({
        "input": msg.text,
        "active_scheme": connected_users[user_id].get("active_scheme", "")
    })

    # Store the bot response for next round
    connected_users[user_id]["last_bot_message"] = response.get("output", "")

    # Check helpline tool output
    helpline_data = None
    for step in response.get("intermediate_steps", []):
        if hasattr(step[0], 'tool') and step[0].tool == "helpline_query_logger":
            helpline_data = step[1]
            break

    return {
        "user_id": user_id,
        "response": response.get("output", "No response generated"),
        "intermediate_steps": response.get("intermediate_steps", []),
        "helpline_log": helpline_data
    }

@app.get("/ping")
def ping():
    return {"status": "ok", "time": datetime.now().isoformat()}