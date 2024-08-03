from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pydantic import BaseModel
from typing import List
import os

# FastAPI 초기화
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 데이터 모델 정의
class DiaryEntry(BaseModel):
    userId: str
    summarizedDiary: str

class Query(BaseModel):
    userId: str
    question: str
    chatHistory: List[str]  # 채팅 내역 추가

# 환경 변수 또는 직접 설정으로부터 OpenAI API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Embeddings와 ChromaDB 초기화
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
persist_directory = 'db'

# ChromaDB 로드 또는 초기화
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Retriever 정의
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# QA 체인 초기화
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# 일상 대화 및 공감 챗봇을 위한 프롬프트 생성 함수
def create_prompt(conversation: list[str], new_question: str) -> str:
    system_message = "만약 이해할 수 없는 말이면 사과하고 너는 감성적이고 공감을 잘해주는 따뜻한 마음씨를 가진 친구야. 자연스럽게 반말로 답변해줘. 핵심적인 답변이 끝나면 자연스럽게 관련 질문해도 좋고 듣고만 있어도 좋아"
    conversation_context = "\n".join(conversation)
    prompt = f"{system_message}\n{conversation_context}\n사용자: {new_question}\n친구:"
    return prompt

# 일기 항목을 추가하는 엔드포인트
@app.post("/add_diary")
async def add_diary(entry: DiaryEntry):
    try:
        vectordb.add_texts([entry.summarizedDiary], metadatas=[{"userId": entry.userId}])
        vectordb.persist()
        return {"message": "Diary embedding is successed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 챗봇 쿼리를 처리하는 엔드포인트
@app.post("/query")
async def query_api(query: Query):
    try:
        user_id = query.userId

        # 사용자로부터 받은 채팅 내역 사용
        user_conversation = query.chatHistory

        # 새로운 질문을 대화 기록에 추가
        user_conversation.append(f"사용자: {query.question}")

        # 프롬프트 생성
        prompt = create_prompt(user_conversation, query.question)

        # 특정 사용자의 데이터를 검색하도록 retriever 수정
        retriever_with_user_id = vectordb.as_retriever(search_kwargs={"k": 3, "metadata_filter": {"userId": user_id}})

        # gpt 모델로 QA 체인 초기화
        qa_chain_with_user_id = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever_with_user_id,
            return_source_documents=True
        )


        # 모델로부터 응답 얻기
        llm_response = qa_chain({"query": prompt})
        result = llm_response['result']

        # 모델의 응답을 대화 기록에 추가
        user_conversation.append(f"친구: {result}")

        # 소스 문서가 존재하는지 확인
        if "source_documents" in llm_response:
            sources = [doc.metadata.get('source', 'Unknown source') for doc in llm_response["source_documents"]]
        else:
            sources = ["No source documents available"]

        return {"message": result, "sources": sources}
    except Exception as e:
        print(f"Error occurred: {e}")  # 디버깅 출력
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행: uvicorn main:app --reload
