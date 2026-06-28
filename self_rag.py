from typing import List, TypedDict, Literal  
from pydantic import BaseModel, Field ,field_validator

from dotenv import load_dotenv
load_dotenv(override=True)
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient


def get_secret(key: str) -> str:
    # Pehle environment se try karo
    val = os.getenv(key, "")
    if val:
        return val
    # Phir st.secrets se try karo
    try:
        return st.secrets[key]
    except:
        return ""

os.environ["GROQ_API_KEY"] = get_secret("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = get_secret("TAVILY_API_KEY")
os.environ["MONGODB_URI"] = get_secret("MONGODB_URI")
os.environ["HF_TOKEN"] = get_secret("HF_TOKEN")


# -----------------------------
# LLM + Tools
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ("GROQ_API_KEY")
)
search_tool = TavilySearch(max_results=3)


# -----------------------------
# ✅ Bool Mixin — YAHAN LAGAO (classes se pehle)
# -----------------------------
class BoolMixin:
    @field_validator("*", mode="before")
    @classmethod
    def coerce_bools(cls, v, info):
        field = cls.model_fields.get(info.field_name)
        if field and field.annotation is bool:
            if isinstance(v, str):
                return v.strip().lower() not in ("false", "0", "no", "")
        return v


# -----------------------------
# Global Retriever
# -----------------------------
retriever = None

def set_retriever(new_retriever):
    global retriever
    retriever = new_retriever

# -----------------------------
# Graph State
# -----------------------------
class State(TypedDict):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int
    isuse: Literal["useful", "not_useful"]
    use_reason: str

# -----------------------------
# 1) Decide Retrieval
# -----------------------------
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer reliably, else False."
    )

decide_retrieval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You decide whether retrieval is needed.\n"
        "Return JSON with key: should_retrieve (boolean).\n\n"
        "Guidelines:\n"
        "- should_retrieve=True if answering requires specific facts from documents.\n"
        "- should_retrieve=False for general explanations/definitions or conversational questions.\n"
        "- If unsure, choose True."
    ),
    ("human", "Question: {question}"),
])

should_retrieve_llm = llm.with_structured_output(RetrieveDecision)

def decide_retrieval(state: State):
    if retriever is None:
        return {"need_retrieval": False}
    
    question = state["question"]
    if "Current Question:" in question:
        actual_question = question.split("Current Question:")[-1].strip()
    else:
        actual_question = question
    
    # ← YAHAN update karo
    conversational_keywords = [
        "my name is",
        "my name",
        "i am",
        "i'm",
        "who am i",
        "what is my",
        "do you remember",
        "what did i",
        "tell me my",
        "call me"
    ]
    if any(kw in actual_question.lower() for kw in conversational_keywords):
        return {"need_retrieval": False}
    
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=actual_question)
    )
    return {"need_retrieval": decision.should_retrieve}

def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"

# -----------------------------
# 2) Direct Answer (with Web Search)
# -----------------------------
def generate_direct(state: State):
    question = state["question"]

    if "Current Question:" in question:
        actual_question = question.split("Current Question:")[-1].strip()
    else:
        actual_question = question

    # ← Conversational hai toh web search mat karo
    conversational = [
    "my name is",
    "my name",
    "i am",
    "i'm",
    "who am i",
    "what is my",
    "do you remember",
    "what did i",
    "tell me my",
    "call me",
    "hy i am",
    "hi i am",
    "hello i am",
    "hey i am"
]
    is_conversational = any(kw in actual_question.lower() for kw in conversational)

    web_context = ""
    if not is_conversational:
        try:
            results = search_tool.invoke({"query": actual_question})
            web_results = results.get("results", [])
            web_context = "\n\n".join([
                f"Title: {r.get('title', '')}\nContent: {r.get('content', '')}"
                for r in web_results
            ])
        except Exception as e:
            print(f"❌ Web Search Error: {e}")
            web_context = ""

    out = llm.invoke(
        "You are a helpful assistant with memory.\n"
        "STRICT RULES:\n"
        "1. If user says 'my name is X' — just greet them as X. Do NOT search who X is.\n"
        "2. Use Conversation History for ALL personal questions.\n"
        "3. Web Results sirf current events/facts ke liye use karo.\n"
        "4. NEVER describe who someone is when user introduces themselves.\n\n"
        f"Web Results:\n{web_context if web_context else 'None'}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )

    return {"answer": out.content}

# -----------------------------
# 3) Retrieve
# -----------------------------
def retrieve(state: State):
    if retriever is None:
        return {"docs": []}
    q = state.get("retrieval_query") or state["question"]
    if "Current Question:" in q:
        q = q.split("Current Question:")[-1].strip()
    return {"docs": retriever.invoke(q)}

# -----------------------------
# 4) Relevance Filter
# -----------------------------
import json

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="Return true or false as boolean only."
    )
    @field_validator("is_relevant", mode="before")
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() not in ("false", "0", "no", "")
        return bool(v)


def is_relevant(state: State):
    question = state["question"]
    if "Current Question:" in question:
        actual_question = question.split("Current Question:")[-1].strip()
    else:
        actual_question = question

    relevant_docs: List[Document] = []
    for doc in state.get("docs", []):
        prompt = (
            "You are judging document relevance.\n"
            "Return ONLY a JSON object: {\"is_relevant\": true} or {\"is_relevant\": false}\n"
            "true = document discusses same topic as question.\n"
            "false = document is unrelated.\n"
            "No explanation. Only JSON.\n\n"
            f"Question:\n{actual_question}\n\n"
            f"Document:\n{doc.page_content[:500]}\n\n"
            "JSON:"
        )
        try:
            out = llm.invoke(prompt)
            raw = out.content.strip()
            if "{" in raw and "}" in raw:
                raw = raw[raw.index("{"):raw.rindex("}")+1]
            else:
                relevant_docs.append(doc)  # ✅ parse fail → safe side pe relevant maano
                continue
            parsed = json.loads(raw)
            is_rel = parsed.get("is_relevant", False)
            if isinstance(is_rel, str):
                is_rel = is_rel.strip().lower() == "true"
            if is_rel:
                relevant_docs.append(doc)
        except Exception as e:
            print(f"⚠️ Relevance parse error: {e}")
            relevant_docs.append(doc)  # ✅ error pe bhi relevant maano
            continue

    return {"relevant_docs": relevant_docs}

def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"

# -----------------------------
# 5) Generate from Context
# -----------------------------
rag_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful RAG chatbot.\n\n"
        "Answer the question based ONLY on the provided CONTEXT.\n"
        "Do NOT mention that you are using a context.\n"
        "If context does not have the answer, say 'I don't have that information.'"
    ),
    ("human", "Question:\n{question}\n\nContext:\n{context}"),
])

def generate_from_context(state: State):
    question = state["question"]
    if "Current Question:" in question:
        actual_question = question.split("Current Question:")[-1].strip()
    else:
        actual_question = question

    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()

    if not context:
        return {"answer": "No answer found.", "context": ""}

    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=actual_question,
            context=context
        )
    )
    return {"answer": out.content, "context": context}

# -----------------------------
# 6) No Answer Found → Web Search
# -----------------------------
def no_answer_found(state: State):
    print("🔍 Web search try kar raha hun...")
    question = state["question"]
    if "Current Question:" in question:
        actual_question = question.split("Current Question:")[-1].strip()
    else:
        actual_question = question

    web_context = ""
    try:
        results = search_tool.invoke({"query": actual_question})
        web_results = results.get("results", [])
        if web_results:
            web_context = "\n\n".join([
                f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')}"
                for r in web_results
            ])
            print(f"✅ Web context: {len(web_context)} chars")
        else:
            print("⚠️ Web search: koi result nahi mila")
    except Exception as e:
        print(f"❌ Web Search Error: {e}")
        web_context = ""

    out = llm.invoke(
        "You are a helpful assistant.\n"
        "Answer DIRECTLY using the web results below.\n"
        "DO NOT say you cannot find information.\n\n"
        f"Web Results:\n{web_context if web_context else 'None'}\n\n"
        f"Question: {actual_question}\n\nAnswer:"
    )
    return {
        "answer": out.content,
        "context": web_context,
        "issup": "fully_supported" if web_context else "no_support",
        "isuse": "useful",
        "use_reason": "Web search fallback answer.",
    }

# -----------------------------
# 7) IsSUP verify + revise loop
# -----------------------------
class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

issup_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
        "Return JSON with keys: issup, evidence.\n"
        "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
        "- fully_supported: Every claim explicitly in CONTEXT, no extra interpretation.\n"
        "- partially_supported: Core facts correct but has interpretation not in CONTEXT.\n"
        "- no_support: Key claims not in CONTEXT.\n\n"
        "Evidence: up to 3 direct quotes from CONTEXT.\n"
        "Do not use outside knowledge."
    ),
    (
        "human",
        "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n"
    ),
])

issup_llm = llm.with_structured_output(IsSUPDecision)

def is_sup(state: State):
    question = state["question"]
    if "Current Question:" in question:
        actual_question = question.split("Current Question:")[-1].strip()
    else:
        actual_question = question

    decision: IsSUPDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=actual_question,
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"issup": decision.issup, "evidence": decision.evidence}

# ✅ MAX_RETRIES kam kiya
MAX_RETRIES = 2

def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"
    return "revise_answer"

def accept_answer(state: State):
    return {}

# -----------------------------
# 8) Revise Answer
# -----------------------------
revise_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a STRICT reviser.\n\n"
        "Output ONLY direct quotes from CONTEXT as bullet points:\n"
        "- <direct quote>\n"
        "- <direct quote>\n\n"
        "Rules:\n"
        "- Use ONLY the CONTEXT.\n"
        "- Do NOT add any words besides bullet dashes.\n"
        "- Do NOT explain anything."
    ),
    (
        "human",
        "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}"
    ),
])

def revise_answer(state: State):
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,
    }

# -----------------------------
# 9) IsUSE
# -----------------------------
class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")

isuse_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Judge USEFULNESS of the ANSWER for the QUESTION.\n"
        "Return JSON with keys: isuse, reason.\n"
        "- useful: Answer directly addresses the question.\n"
        "- not_useful: Generic, off-topic, or incomplete.\n"
        "Keep reason to 1 short line."
    ),
    ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
])

isuse_llm = llm.with_structured_output(IsUSEDecision)

def is_use(state: State):
    question = state["question"]
    if "Current Question:" in question:
        actual_question = question.split("Current Question:")[-1].strip()
    else:
        actual_question = question

    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=actual_question,
            answer=state.get("answer", ""),
        )
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}

# ✅ MAX_REWRITE_TRIES kam kiya
MAX_REWRITE_TRIES = 2

def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    if not state.get("relevant_docs"):
        return "no_answer_found"
    return "rewrite_question"

# -----------------------------
# 10) Rewrite Question
# -----------------------------
class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval against internal PDFs."
    )

rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL PDFs.\n\n"
        "Rules:\n"
        "- Keep it short (6-16 words).\n"
        "- Add 2-5 high-signal keywords.\n"
        "- Remove filler words.\n"
        "- Do NOT answer the question.\n"
        "- Output JSON with key: retrieval_query"
    ),
    (
        "human",
        "QUESTION:\n{question}\n\n"
        "Previous retrieval query:\n{retrieval_query}\n\n"
        "Answer (if any):\n{answer}"
    ),
])

rewrite_llm = llm.with_structured_output(RewriteDecision)

def rewrite_question(state: State):
    decision: RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }

# -----------------------------
# Build Graph
# -----------------------------
g = StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("no_answer_found", no_answer_found)
g.add_node("is_sup", is_sup)
g.add_node("accept_answer", accept_answer)
g.add_node("revise_answer", revise_answer)
g.add_node("is_use", is_use)
g.add_node("rewrite_question", rewrite_question)

g.add_edge(START, "decide_retrieval")

g.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {"generate_direct": "generate_direct", "retrieve": "retrieve"},
)

# ✅ generate_direct → END (web search already handle kar leta hai)
g.add_edge("generate_direct", END)
g.add_edge("retrieve", "is_relevant")

g.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_answer_found": "no_answer_found",
    },
)

# ✅ no_answer_found → END (web search already handle kar leta hai)
g.add_edge("no_answer_found", END)
g.add_edge("generate_from_context", "is_sup")

g.add_conditional_edges(
    "is_sup",
    route_after_issup,
    {
        "accept_answer": "accept_answer",
        "revise_answer": "revise_answer",
    },
)

g.add_edge("revise_answer", "is_sup")
g.add_edge("accept_answer", "is_use")

g.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {
        "END": END,
        "rewrite_question": "rewrite_question",
        "no_answer_found": "no_answer_found",
    },
)

g.add_edge("rewrite_question", "retrieve")

mongo_client = MongoClient(os.getenv("MONGODB_URI"))
memory = MongoDBSaver(mongo_client)
app = g.compile(checkpointer=memory)