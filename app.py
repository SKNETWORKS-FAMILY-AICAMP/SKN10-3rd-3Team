import os
import pandas as pd
import chainlit as cl
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain.schema import Document
from langchain_core.messages import HumanMessage

# Load .env
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 전역 객체들 선언 (start 함수 밖으로 뺌)
llm = ChatOpenAI(model_name="gpt-4o-mini")
tavily_tool = TavilySearchResults(
    max_results=5,
    include_answer=True,
    include_raw_content=True,
    include_domains=["m.entertain.naver.com"]
)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 한국어 요약 도우미입니다. 사용자가 제공한 웹 검색 결과를 바탕으로 핵심만 정리해서 답변하세요."),
    ("human", "다음은 검색 결과입니다. {search_content}.다음은 요청사항입니다.{target}이를 바탕으로 요청사항과 관련된 결과만 가지고 자연스럽고 간결하게 요약해 주세요:\n\n")
])

def summarize_search(user_input: str) -> str:
    search_results = tavily_tool.invoke(user_input)
    if not search_results or not isinstance(search_results, list):
        return "검색 결과가 없습니다."
    all_content = "\n\n".join([item.get("content", "") for item in search_results])
    prompt_text = summary_prompt.format(search_content=all_content,target=user_input)
    response = llm.invoke([HumanMessage(content=prompt_text)])
    return response.content.strip()

@cl.on_chat_start
async def start():
    try:
        DATA_PATH = "./data/"
        loader = CSVLoader(
            file_path=DATA_PATH + "중증외상센터 등장인물.csv",
            encoding="utf-8",
            metadata_columns=["character_name", "actor_name"]
        )
        raw_docs = loader.load()
        print("✅ CSV 로딩 성공")

        docs = []
        for doc in raw_docs:
            metadata = doc.metadata
            character_name = metadata.get("character_name")
            actor_name = metadata.get("actor_name")
            content = doc.page_content
            if character_name and actor_name:
                description = content.split(",")[-1].strip()
                new_content = f"{character_name}은(는) 배우 {actor_name}이 연기했습니다. {description}"
                docs.append(Document(page_content=new_content, metadata={"character_name": character_name}))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        split_docs = splitter.split_documents(docs)

        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(split_docs, embedding=embedding)
        retriever = vectorstore.as_retriever()

        rag_prompt = PromptTemplate.from_template("""
당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
아래에 주어진 문맥(Context)을 참고하여 질문에 답변하세요.

문맥에 관련된 정보가 있는 경우, 구체적이고 자세하게 알려주세요.
문맥에 정보가 없으면 '답을 찾을 수 없습니다'라고 답변하세요.

질문: {question}

문맥:
{context}

답변:
""")

        cl.user_session.set("retriever", retriever)
        cl.user_session.set("llm", llm)
        cl.user_session.set("prompt", rag_prompt)
        print("✅ 세션 세팅 완료")

    except Exception as e:
        print(f"❌ 초기화 실패: {e}")

@cl.on_message
async def handle_message(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    llm = cl.user_session.get("llm")
    prompt = cl.user_session.get("prompt")

    user_input = message.content
    await cl.Message(content=f"❓ 질문: **{user_input}**\n먼저 내부 문서에서 찾아볼게요!").send()

    if retriever is None:
        await cl.Message(content="⚠️ retriever가 초기화되지 않았어요. 새로고침 후 다시 시도해 주세요!").send()
        return

    internal_result = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in internal_result]) if internal_result else ""

    answer = None
    if context:
        prompt_text = prompt.format_prompt(context=context, question=user_input).to_string()
        answer = llm.invoke(prompt_text)

    if answer and answer.content.strip() and not _contains_not_found_phrase(answer.content):
        await cl.Message(content="✅ 내부 문서에서 찾은 결과입니다.").send()
        await cl.Message(content=answer.content).send()
    else:
        await cl.Message(content="📭 내부 문서에서는 관련 정보를 찾을 수 없었어요.\n외부 자료를 검색해서 알려드릴게요!").send()
        result = summarize_search(user_input)
        await cl.Message(content=result).send()

def _contains_not_found_phrase(text: str) -> bool:
    text = text.lower()
    not_found_phrases = [
        "정보가 없습니다",
        "찾을 수 없습니다",
        "포함되어 있지 않습니다",
        "제공된 문맥에는",
        "뉴스 웹사이트를 방문하시거나",
        "외부에서 확인해보세요",
        "정확한 내용이 없습니다",
        "이 질문에 대한 문서가 없습니다"
    ]
    return any(phrase in text for phrase in not_found_phrases)


