import os
import pandas as pd
import chainlit as cl
import urllib.parse
from dotenv import load_dotenv
from chainlit.context import get_context
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from urllib.parse import unquote, parse_qs
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Load .env
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

dataset_memory = {}

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

# 📌 분기 로직 포함한 process_docs 함수
def process_docs(file_path: str) -> tuple[list[Document], PromptTemplate]:
    filename = os.path.splitext(os.path.basename(file_path))[0]
    docs = []

    if "등장인물" in filename or "3월이후" in filename:
        loader = CSVLoader(
            file_path=file_path,
            encoding="utf-8",
            metadata_columns=["character_name", "actor_name"]
        )
        raw_docs = loader.load()
        print("✅ 등장인물 or 3월이후 CSV 로딩 성공")

        # 🧠 작품명 자동 추출
        title = filename.replace("_등장인물", "").replace(" 등장인물", "").replace("_", " ").strip()

        docs = []
        for doc in raw_docs:
            metadata = doc.metadata
            character_name = metadata.get("character_name")
            actor_name = metadata.get("actor_name")
            content = doc.page_content.strip() if doc.page_content else ""

            if character_name and actor_name:
                # 👇 description을 전체 내용으로 사용하도록 수정
                description = content.strip()

                new_content = f"""
            드라마 '{title}'에 등장하는 인물 '{character_name}'은(는),
            배우 '{actor_name}'이 연기한 역할입니다.
            {description}
            """.strip()

                # ✅ 문서 확인 로그 추가
                print(f"📄 생성된 문서: {new_content}")

                docs.append(Document(
                    page_content=new_content,
                    metadata={
                        "character_name": character_name,
                        "actor_name": actor_name
                    }
                ))

                

        prompt = PromptTemplate.from_template("""
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

    elif "naver_entertainment_articles" in filename:
        loader = CSVLoader(file_path=file_path, encoding="utf-8", metadata_columns=["title", "content"])
        raw_docs = loader.load()
        print("✅ 네이버 뉴스 CSV 로딩 성공")

        for doc in raw_docs:
            metadata = doc.metadata
            title = metadata.get("title")
            content = metadata.get("content")
            full_content = doc.page_content
            if title and content:
                description = full_content.split(",")[-1].strip()
                new_content = f"{title}은(는) {content}인 내용입니다. {description}"
                docs.append(Document(page_content=new_content, metadata={"title": title}))

        prompt = PromptTemplate.from_template("""
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

    elif "kinolights_ranking_data" in filename:
        loader = CSVLoader(file_path=file_path, encoding="utf-8", metadata_columns=[
            '제목', '종류', '장르', '방영일', '개봉일',
            '채널', '제작국가', '제작연도', '연령등급',
            '오리지널', '배우', '러닝타임', 'URL'
        ])
        raw_docs = loader.load()
        print("✅ Kinolights CSV 로딩 성공")

        for doc in raw_docs:
            metadata = doc.metadata
            title = metadata.get("제목")
            genre = metadata.get("장르")
            content_type = metadata.get("종류")
            country = metadata.get("제작국가")
            production_year = metadata.get("제작연도")
            release_date = metadata.get("개봉일")
            broadcast_date = metadata.get("방영일")
            channel = metadata.get("채널")
            age_rating = metadata.get("연령등급")
            is_original = metadata.get("오리지널")
            actors = metadata.get("배우")
            running_time = metadata.get("러닝타임")
            url = metadata.get("URL")
            content = doc.page_content
            description = content.split(",")[-1].strip()

            new_content = f"""
'{title}'은(는) {production_year}년에 {country}에서 제작된 {genre} {content_type}입니다.
주요 출연진으로는 {actors}가 있으며, {channel} 채널을 통해 방영되었습니다.
방영일은 {broadcast_date}, 개봉일은 {release_date}로 예정되어 있으며, 러닝타임은 {running_time}입니다.
연령등급은 {age_rating}이며, 오리지널 여부는 '{is_original}'입니다.
작품에 대한 자세한 정보는 다음 링크에서 확인할 수 있습니다: {url}

작품 요약:
{description}
"""
            docs.append(Document(page_content=new_content, metadata=metadata))

        prompt = PromptTemplate.from_template("""
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

    else:
        loader = CSVLoader(file_path=file_path, encoding="utf-8", metadata_columns=None)
        raw_docs = loader.load()
        docs = raw_docs
        print("✅ 기본 CSV 로딩 성공")

        prompt = PromptTemplate.from_template("""
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

    return docs, prompt

@cl.on_chat_start
async def start():
    try:
        import socket
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)

        dataset_name = dataset_memory.get(ip, "중증외상센터_등장인물")
        cl.user_session.set("dataset_name", dataset_name)
        print(f"📦 [Chainlit] {ip} → {dataset_name} 로드 완료")

        DATA_PATH = "./data/"
        file_path = os.path.join(DATA_PATH, f"{dataset_name}.csv")
        print(f"📁 불러올 파일 경로: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 파일이 존재하지 않아요: {file_path}")

        docs, rag_prompt = process_docs(file_path)
        if not docs:
            raise ValueError("📭 문서가 비어있어요! CSV 내용이 잘못되었을 수 있어요.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        split_docs = splitter.split_documents(docs)

        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(split_docs, embedding=embedding)
        retriever = vectorstore.as_retriever()

        cl.user_session.set("retriever", retriever)
        cl.user_session.set("llm", llm)
        cl.user_session.set("prompt", rag_prompt)

        print(f"✅ 세션 세팅 완료: {dataset_name}")

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


from chainlit.server import app as chainlit_app

# 미들웨어: request.query_string 저장
class SaveQueryStringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        query_string = request.scope.get("query_string", b"").decode()
        dataset = None
        if "dataset=" in query_string:
            dataset = unquote(query_string.split("dataset=")[-1].split("&")[0])
        if dataset:
            request.scope["dataset_name"] = dataset

            # 🔥 여기서 전역 dict에 저장 (IP 기준 or 랜덤 ID 기준으로)
            from starlette.middleware.base import RequestResponseEndpoint
            client_ip = request.client.host
            dataset_memory[client_ip] = dataset
            print(f"💾 [미들웨어] {client_ip} → {dataset} 저장 완료")

        response = await call_next(request)
        return response

# FastAPI에 미들웨어 등록
chainlit_app.add_middleware(SaveQueryStringMiddleware)