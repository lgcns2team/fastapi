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

class PromptRequest(BaseModel):
    prompt_id: str = Field(..., description="Bedrock Prompt ARN or ID")
    user_query: str = Field(..., description="사용자 질문/입력")
    variables: dict = Field(default_factory=dict, description="추가 프롬프트 변수 (옵션)")

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
            "/chat/prompt": "POST - Bedrock Prompt 호출 (스트리밍)",
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
# POST /chat/prompt (스트리밍)
# -------------------------
@app.post("/chat/prompt")
async def chat_with_prompt(request: PromptRequest):
    """Bedrock Prompt 호출 (스트리밍 응답)"""
    return StreamingResponse(
        stream_prompt_response(request),
        media_type="text/event-stream"
    )

async def stream_prompt_response(request: PromptRequest):
    """Prompt 기반 채팅 스트리밍 제너레이터"""
    try:
        logger.info(f"Prompt request - Prompt ID: {request.prompt_id}, Query: {request.user_query[:50]}...")
        
        # Bedrock Agent client for prompt management
        bedrock_agent = boto3.client(
            service_name='bedrock-agent',
            region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
        )
        
        # ARN 형식인지 확인
        if request.prompt_id.startswith('arn:'):
            prompt_identifier = request.prompt_id
        else:
            # ID만 있는 경우 ARN 생성
            prompt_identifier = f"arn:aws:bedrock:{os.getenv('AWS_REGION', 'ap-northeast-2')}:125814533785:prompt/{request.prompt_id}"
        
        logger.info(f"Using Prompt ARN: {prompt_identifier}")
        
        # Get prompt information
        try:
            prompt_response = bedrock_agent.get_prompt(
                promptIdentifier=prompt_identifier
            )
            
            logger.info(f"Prompt retrieved: {prompt_response.get('name', 'Unknown')}")
            
            # Extract prompt configuration
            variants = prompt_response.get('variants', [])
            if not variants:
                raise ValueError("Prompt has no variants")
            
            variant = variants[0]
            template_type = variant.get('templateType', 'TEXT')
            
            logger.info(f"Template type: {template_type}")
            
            # Get model ID from prompt or use default
            model_id = prompt_response.get('defaultModelId', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
            
            # Prepare prompt variables
            prompt_variables = {
                "user_query": request.user_query,
                **request.variables
            }
            
            # Process TEXT template
            if template_type == 'TEXT':
                template_config = variant.get('templateConfiguration', {})
                template_text = template_config.get('text', {}).get('text', '')
                
                # Replace variables in template
                formatted_prompt = template_text
                for var_name, var_value in prompt_variables.items():
                    # Match Bedrock variable syntax: {{variable_name}}
                    formatted_prompt = formatted_prompt.replace(f"{{{{{var_name}}}}}", str(var_value))
                
                logger.info(f"Formatted prompt (first 100 chars): {formatted_prompt[:100]}...")
                
                # Get inference configuration from prompt
                inference_config = variant.get('inferenceConfiguration', {})
                
                # Build request body for Claude
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": inference_config.get('maxTokens', 4096),
                    "temperature": inference_config.get('temperature', 1.0),
                    "messages": [
                        {
                            "role": "user",
                            "content": formatted_prompt
                        }
                    ]
                }
                
                # Add stop sequences if defined
                if 'stopSequences' in inference_config:
                    body['stop_sequences'] = inference_config['stopSequences']
                
                logger.info(f"Invoking model: {model_id}")
                
                # Invoke model with streaming
                response = bedrock_runtime.invoke_model_with_response_stream(
                    modelId=model_id,
                    body=json.dumps(body)
                )
                
                # Stream the response
                for event in response['body']:
                    chunk = json.loads(event['chunk']['bytes'])
                    
                    if chunk['type'] == 'content_block_delta':
                        text = chunk['delta'].get('text', '')
                        if text:
                            yield f"data: {json.dumps({'type': 'content', 'text': text}, ensure_ascii=False)}\n\n"
                    
                    elif chunk['type'] == 'message_stop':
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break
            
            # Process CHAT template
            elif template_type == 'CHAT':
                template_config = variant.get('templateConfiguration', {})
                chat_config = template_config.get('chat', {})
                messages = chat_config.get('messages', [])
                system_prompts = chat_config.get('system', [])
                
                logger.info(f"CHAT template - Messages: {len(messages)}, System prompts: {len(system_prompts)}")
                
                # Log raw messages for debugging
                for i, msg in enumerate(messages):
                    logger.info(f"Raw message {i}: role={msg.get('role')}, content={msg.get('content')}")
                
                # Get inference configuration from prompt
                inference_config = variant.get('inferenceConfiguration', {})
                
                # Process messages and replace variables
                formatted_messages = []
                for msg in messages:
                    role = msg.get('role', 'user')
                    content_blocks = msg.get('content', [])
                    
                    # Process each content block
                    formatted_content = []
                    for block in content_blocks:
                        if 'text' in block:
                            text = block['text']
                            # Replace variables
                            for var_name, var_value in prompt_variables.items():
                                text = text.replace(f"{{{{{var_name}}}}}", str(var_value))
                            # Only add non-empty text
                            if text.strip():
                                formatted_content.append({"type": "text", "text": text})
                    
                    if formatted_content:
                        # Simplify for Claude API - just extract text
                        content_text = " ".join([c['text'] for c in formatted_content if 'text' in c])
                        if content_text.strip():  # Only add messages with non-empty content
                            formatted_messages.append({
                                "role": role,
                                "content": content_text
                            })
                
                # If no valid messages or last message is not user, add user_query as final user message
                if not formatted_messages or formatted_messages[-1].get('role') != 'user':
                    formatted_messages.append({
                        "role": "user",
                        "content": request.user_query
                    })
                # If there are messages but first user message is just a variable placeholder, replace it
                elif formatted_messages and not formatted_messages[0].get('content', '').strip():
                    formatted_messages[0]['content'] = request.user_query
                
                logger.info(f"Formatted {len(formatted_messages)} messages")
                for i, msg in enumerate(formatted_messages):
                    logger.info(f"Final message {i}: role={msg['role']}, content={msg['content'][:50] if msg['content'] else 'EMPTY'}...")
                
                # Build request body for Claude
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": inference_config.get('maxTokens', 4096),
                    "temperature": inference_config.get('temperature', 1.0),
                    "messages": formatted_messages
                }
                
                # Add system prompts if any
                if system_prompts:
                    system_text = []
                    for sys_prompt in system_prompts:
                        if 'text' in sys_prompt:
                            text = sys_prompt['text']
                            # Replace variables in system prompt
                            for var_name, var_value in prompt_variables.items():
                                text = text.replace(f"{{{{{var_name}}}}}", str(var_value))
                            system_text.append(text)
                    
                    if system_text:
                        body['system'] = " ".join(system_text)
                        logger.info(f"Added system prompt: {body['system'][:100]}...")
                
                # Add stop sequences if defined
                if 'stopSequences' in inference_config:
                    body['stop_sequences'] = inference_config['stopSequences']
                
                logger.info(f"Invoking model: {model_id}")
                
                # Invoke model with streaming
                response = bedrock_runtime.invoke_model_with_response_stream(
                    modelId=model_id,
                    body=json.dumps(body)
                )
                
                # Stream the response
                for event in response['body']:
                    chunk = json.loads(event['chunk']['bytes'])
                    
                    if chunk['type'] == 'content_block_delta':
                        text = chunk['delta'].get('text', '')
                        if text:
                            yield f"data: {json.dumps({'type': 'content', 'text': text}, ensure_ascii=False)}\n\n"
                    
                    elif chunk['type'] == 'message_stop':
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break
            
            else:
                raise ValueError(f"Unsupported template type: {template_type}")
        
        except bedrock_agent.exceptions.ResourceNotFoundException:
            error_msg = f"Prompt not found: {request.prompt_id}"
            logger.error(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        
    except Exception as e:
        logger.error(f"Prompt error: {str(e)}")
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