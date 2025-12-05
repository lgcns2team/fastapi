# =========================
# FastAPI + AWS Bedrock Gateway (Simplified)
# =========================

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import boto3
import json
from typing import List
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# -------------------------
# 로깅 설정
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# FastAPI 앱 생성
# -------------------------
app = FastAPI(
    title="Bedrock Gateway API",
    description="AWS Bedrock Knowledge Base + AI Chat (Streaming)",
    version="3.0.0"
)

# -------------------------
# CORS 설정
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Bedrock 클라이언트 초기화
# -------------------------
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
)

bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
)

# -------------------------
# 요청 모델
# -------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = Field(default="anthropic.claude-3-5-sonnet-20240620-v1:0")
    messages: List[Message]
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=1.0)

class KnowledgeBaseRequest(BaseModel):
    query: str = Field(..., description="사용자 질문")
    kb_id: str = Field(..., description="Knowledge Base ID")
    model_arn: str = Field(..., description="Foundation Model ARN")

# -------------------------
# GET /
# -------------------------
@app.get("/")
async def root():
    return {
        "service": "Bedrock Gateway API",
        "version": "3.0.0",
        "status": "operational",
        "endpoints": {
            "/chat": "POST - AI 채팅 (스트리밍)",
            "/chat/knowledge": "POST - Knowledge Base 검색 (스트리밍)",
            "/health": "GET - 헬스 체크",
            "/models": "GET - 모델 목록"
        }
    }

# -------------------------
# GET /health
# -------------------------
@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        # Bedrock 연결 테스트는 비용이 들 수 있으므로 간단하게 처리
        return {
            "status": "healthy",
            "bedrock": "configured",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# -------------------------
# GET /models
# -------------------------
@app.get("/models")
async def list_models():
    """사용 가능한 모델 목록"""
    return {
        "models": [
            {
                "id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "name": "Claude 3.5 Sonnet",
                "provider": "Anthropic",
                "context_window": 200000
            }
        ]
    }

# -------------------------
# POST /chat (스트리밍)
# -------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    """AI 채팅 (스트리밍 응답)"""
    return StreamingResponse(
        stream_chat_response(request),
        media_type="text/event-stream"
    )

async def stream_chat_response(request: ChatRequest):
    """채팅 스트리밍 제너레이터"""
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages
        }

        logger.info(f"Chat request - Model: {request.model}")

        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId=request.model,
            body=json.dumps(body)
        )

        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            
            if chunk['type'] == 'content_block_delta':
                text = chunk['delta'].get('text', '')
                if text:
                    yield f"data: {json.dumps({'type': 'content', 'text': text})}\n\n"
            
            elif chunk['type'] == 'message_stop':
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

# -------------------------
# POST /chat/knowledge (스트리밍)
# -------------------------
@app.post("/chat/knowledge")
async def knowledge_base(request: KnowledgeBaseRequest):
    """Knowledge Base 검색 (스트리밍 응답)"""
    return StreamingResponse(
        stream_knowledge_response(request),
        media_type="text/event-stream"
    )

async def stream_knowledge_response(request: KnowledgeBaseRequest):
    """Knowledge Base 스트리밍 제너레이터"""
    try:
        logger.info(f"KB request - KB ID: {request.kb_id}, Query: {request.query[:50]}...")

        # retrieve_and_generate_stream 호출
        response = bedrock_agent_runtime.retrieve_and_generate_stream(
            input={'text': request.query},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': request.kb_id,
                    'modelArn': request.model_arn
                }
            }
        )

        # EventStream 처리
        citations = []
        full_text = ""
        
        for event in response['stream']:
            # output 이벤트 - 텍스트 스트리밍
            if 'output' in event:
                output_data = event['output']
                
                if 'text' in output_data:
                    text = output_data['text']
                    full_text += text
                    
                    # 스트리밍으로 전송 (ensure_ascii=False로 한글 제대로 출력)
                    yield f"data: {json.dumps({'type': 'content', 'text': text}, ensure_ascii=False)}\n\n"
                    logger.info(f"Sent text chunk: {text[:30]}...")
            
            # citation 이벤트 - 인용 정보
            elif 'citation' in event:
                citation_data = event['citation']
                citations.append(citation_data)
                logger.info(f"Citation received")
        
        # Citations를 메타데이터로 전송
        if citations:
            logger.info(f"Sending {len(citations)} citations")
            yield f"data: {json.dumps({'type': 'citations', 'count': len(citations), 'data': citations}, ensure_ascii=False)}\n\n"
        
        # 완료 신호
        logger.info(f"Stream complete. Total text length: {len(full_text)}")
        yield f"data: {json.dumps({'type': 'done', 'total_length': len(full_text)})}\n\n"

    except Exception as e:
        logger.error(f"KB error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

# -------------------------
# 글로벌 예외 핸들러
# -------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.utcnow().isoformat()
    }

# -------------------------
# 직접 실행
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )