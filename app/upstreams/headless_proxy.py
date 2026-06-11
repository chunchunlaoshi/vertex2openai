"""
batchGraphql 直连代理上游通道

基于 Agent Platform Studio Express Mode 的 batchGraphql 协议实现。
无需无头浏览器，直接通过 Cookie + SAPISIDHASH 鉴权调用 batchGraphql 端点。

请求格式：
  POST cloudconsole-pa.clients6.google.com/.../batchGraphql?key=...
  Body: { requestContext, querySignature, operationName, variables }

响应格式：
  流式返回多个 JSON 对象，每个包含 results[].data.candidates[].content.parts
"""

import json
import time
import uuid
import httpx
import traceback
from typing import Any, Optional, List, Dict, AsyncGenerator
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from runtime_state import app_state
import config as app_config

from cookie_auth import (
    build_headers,
    BATCH_GRAPHQL_URL,
    STREAM_GENERATE_QUERY_SIGNATURE,
    STREAM_GENERATE_OPERATION_NAME,
)


# ========== requestContext 模板 ==========

def _build_request_context(project_id: str) -> dict:
    """
    构建 batchGraphql 的 requestContext

    保持与 Studio 页面发出的请求格式一致。
    """
    return {
        "clientVersion": "boq_cloud-boq-clientweb-vertexaistudio_20260609.06_p0",
        "pagePath": "/agent-platform/studio/multimodal",
        "pageViewId": int(time.time() * 1000) % (10**15),
        "trackingId": str(int(time.time() * 1000000) % (10**17)),
        "backendOverrides": {},
        "clientSessionId": str(uuid.uuid4()).upper(),
        "projectId": project_id,
        "selectedPurview": {"projectId": project_id},
        "jurisdiction": "global",
        "localizationData": {"locale": "zh_CN", "timezone": "Asia/Hong_Kong"}
    }


# ========== OpenAI → batchGraphql 消息格式转换 ==========

def _convert_messages_to_contents(messages: list) -> tuple:
    """
    将 OpenAI messages 转换为 Vertex AI contents 格式
    
    Returns:
        (contents_list, system_instruction_text_or_None)
    """
    contents = []
    system_parts = []
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue
        
        # OpenAI role → Gemini role
        gemini_role = "user" if role == "user" else "model"
        
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            # 多模态消息 - 内容可能是 Pydantic 模型或 dict
            for item in content:
                # 如果是 Pydantic BaseModel，转成 dict
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item_type == "image_url":
                        url = item.get("image_url", {})
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        if url.startswith("data:"):
                            try:
                                header, encoded = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                parts.append({
                                    "inlineData": {"mimeType": mime_type, "data": encoded}
                                })
                            except Exception:
                                parts.append({"text": "[图片解析失败]"})
                elif isinstance(item, str):
                    parts.append({"text": item})
        
        if parts:
            contents.append({"role": gemini_role, "parts": parts})
    
    system_text = "\n".join(system_parts) if system_parts else None
    return contents, system_text


def _build_batch_graphql_body(
    project_id: str,
    model_name: str,
    request: OpenAIRequest,
) -> dict:
    """
    构建完整的 batchGraphql 请求体
    """
    contents, system_text = _convert_messages_to_contents(request.messages)
    
    # 模型完整路径
    model_path = f"projects/{project_id}/locations/global/publishers/google/models/{model_name}"
    
    # 生成配置
    gen_config = {
        "temperature": request.temperature if request.temperature is not None else 1,
        "topP": request.top_p if request.top_p is not None else 0.95,
        "maxOutputTokens": request.max_tokens if request.max_tokens is not None else 65535,
    }
    
    # 思考模式
    if any(kw in model_name for kw in ("gemini-3", "gemini-2.5")):
        gen_config["thinkingConfig"] = {
            "thinkingLevel": "MEDIUM",
            "includeThoughts": True
        }
    
    # 安全设置 - 全部关闭
    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    ]
    
    # variables
    variables = {
        "contents": contents,
        "model": model_path,
        "generationConfig": gen_config,
        "safetySettings": safety_settings,
    }
    
    # 系统提示
    if system_text:
        variables["systemInstruction"] = {"parts": [{"text": system_text}]}
    
    # stop sequences
    if request.stop:
        gen_config["stopSequences"] = request.stop if isinstance(request.stop, list) else [request.stop]
    
    # 搜索工具
    if hasattr(request, 'model') and request.model.endswith("-search"):
        variables["tools"] = [{"googleSearch": {}}]
    
    body = {
        "requestContext": _build_request_context(project_id),
        "querySignature": STREAM_GENERATE_QUERY_SIGNATURE,
        "operationName": STREAM_GENERATE_OPERATION_NAME,
        "variables": variables,
    }
    
    return body


# ========== batchGraphql 流式响应解析 ==========

async def _iter_json_objects(response) -> AsyncGenerator[dict, None]:
    """
    从 batchGraphql 流式响应中逐个提取完整 JSON 对象
    
    响应是连续的 JSON 对象流（非 JSON Array），需要手动切分。
    """
    buffer = ""
    async for chunk in response.aiter_text():
        if not chunk:
            continue
        buffer += chunk
        
        while True:
            start = buffer.find('{')
            if start == -1:
                buffer = ""
                break
            
            brace_count = 0
            in_string = False
            escape = False
            end = -1
            
            for i in range(start, len(buffer)):
                c = buffer[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break
            
            if end == -1:
                buffer = buffer[start:]
                break
            
            json_str = buffer[start:end + 1]
            buffer = buffer[end + 1:]
            
            try:
                yield json.loads(json_str)
            except json.JSONDecodeError:
                pass


def _extract_from_results(obj: dict):
    """
    从 batchGraphql 响应对象中提取文本/图片/错误
    
    Yields: (event_type, data)
      event_type: "text" | "thought" | "image" | "finish" | "error"
      data: str (text/thought/image) 或 dict (error)
    """
    # 检查顶层错误
    if "error" in obj:
        yield ("error", obj["error"])
        return
    
    results = obj.get("results", [])
    for result in results:
        # 检查结果级错误
        if "errors" in result:
            for err in result["errors"]:
                yield ("error", err)
            continue
        
        data = result.get("data")
        if not data:
            continue
        
        candidates = data.get("candidates", [])
        for candidate in candidates:
            content_obj = candidate.get("content") or {}
            parts = content_obj.get("parts") or []
            
            for part in parts:
                text = part.get("text", "")
                if text:
                    if part.get("thought", False):
                        yield ("thought", text)
                    else:
                        yield ("text", text)
                
                inline_data = part.get("inlineData")
                if inline_data:
                    mime_type = inline_data.get("mimeType", "")
                    b64 = inline_data.get("data", "")
                    if mime_type and b64:
                        image_md = f"![Generated Image](data:{mime_type};base64,{b64})"
                        yield ("image", image_md)
            
            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason in ("STOP", "MAX_TOKENS", "SAFETY"):
                yield ("finish", finish_reason)


# ========== OpenAI SSE 格式化 ==========

def _make_openai_chunk(
    response_id: str,
    model: str,
    content: str = None,
    reasoning_content: str = None,
    finish_reason: str = None,
    role: str = None,
) -> str:
    """构建单个 OpenAI SSE chunk"""
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ========== 认证解析 ==========

def _get_cookie_string() -> str:
    """获取可用的 Cookie 字符串"""
    return app_config.GOOGLE_COOKIE or app_state.get_google_cookie() or ""


def _get_project_id() -> str:
    """获取可用的 Project ID"""
    return app_config.GOOGLE_PROJECT_ID or app_state.get_project_id() or ""


# ========== 主代理类 ==========

class HeadlessProxyUpstream(BaseUpstream):
    """
    batchGraphql 直连代理
    
    使用 Cookie + SAPISIDHASH 鉴权，
    通过 batchGraphql 端点调用 Agent Platform Studio Express Mode 模型。
    """
    
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        # ===== 1. 验证认证 =====
        cookie_str = _get_cookie_string()
        if not cookie_str:
            return JSONResponse(status_code=401, content={"error": {"message": (
                "未配置 Google Cookie。\n"
                "请在大盘控制台中粘贴 Cookie 和 Project ID，\n"
                "或设置环境变量 GOOGLE_COOKIE 和 GOOGLE_PROJECT_ID。"
            ), "type": "auth_error"}})
        
        project_id = _get_project_id()
        if not project_id:
            return JSONResponse(status_code=400, content={"error": {"message": (
                "未配置 Google Cloud Project ID。\n"
                "请在大盘中填写，或设置环境变量 GOOGLE_PROJECT_ID。\n"
                "可从 Studio URL 中获取：...?project=YOUR_PROJECT_ID"
            ), "type": "config_error"}})
        
        # ===== 2. 构建请求头（每次重新计算 SAPISIDHASH） =====
        headers = build_headers(cookie_str)
        if not headers:
            return JSONResponse(status_code=401, content={"error": {"message": (
                "Cookie 中未找到 SAPISID，无法计算认证头。\n"
                "请确保 Cookie 来自已登录的 console.cloud.google.com 页面。"
            ), "type": "auth_error"}})
        
        # ===== 3. 解析模型名 =====
        model_display = request_obj.model
        base_model_name = model_display
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]
        
        # ===== 4. 构建 batchGraphql 请求体 =====
        body = _build_batch_graphql_body(project_id, base_model_name, request_obj)
        
        # ===== 5. HTTP 客户端配置 =====
        client_kwargs = {
            "timeout": httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=10.0),
            "follow_redirects": True,
            "http1": True,
            "http2": True,
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL
        
        is_stream = request_obj.stream
        response_id = f"chatcmpl-studio-{int(time.time())}"
        
        # ========== 流式处理 ==========
        if is_stream:
            async def stream_generator():
                role_sent = False
                has_content = False
                
                try:
                    # 每次请求重新计算 SAPISIDHASH
                    req_headers = build_headers(_get_cookie_string()) or headers
                    
                    async with httpx.AsyncClient(**client_kwargs) as client:
                        async with client.stream("POST", BATCH_GRAPHQL_URL,
                                                  headers=req_headers, json=body) as response:
                            
                            if response.status_code != 200:
                                error_text = await response.aread()
                                error_msg = error_text.decode('utf-8', errors='replace')[:1000]
                                print(f"❌ [batchGraphql] HTTP {response.status_code}: {error_msg[:200]}")
                                yield _make_openai_chunk(
                                    response_id, model_display,
                                    content=f"[Studio 错误 {response.status_code}] {error_msg[:500]}"
                                )
                                yield _make_openai_chunk(
                                    response_id, model_display, finish_reason="stop"
                                )
                                yield "data: [DONE]\n\n"
                                return
                            
                            # 发送 role chunk
                            if not role_sent:
                                yield _make_openai_chunk(
                                    response_id, model_display, role="assistant"
                                )
                                role_sent = True
                            
                            # 逐个解析 JSON 对象并提取内容
                            async for obj in _iter_json_objects(response):
                                for event_type, data in _extract_from_results(obj):
                                    if event_type == "text":
                                        yield _make_openai_chunk(
                                            response_id, model_display, content=data
                                        )
                                        has_content = True
                                    
                                    elif event_type == "thought":
                                        yield _make_openai_chunk(
                                            response_id, model_display, reasoning_content=data
                                        )
                                        has_content = True
                                    
                                    elif event_type == "image":
                                        yield _make_openai_chunk(
                                            response_id, model_display, content=data
                                        )
                                        has_content = True
                                    
                                    elif event_type == "finish":
                                        fr = "stop" if data == "STOP" else "length" if data == "MAX_TOKENS" else "stop"
                                        yield _make_openai_chunk(
                                            response_id, model_display, finish_reason=fr
                                        )
                                    
                                    elif event_type == "error":
                                        err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                                        print(f"❌ [batchGraphql] API 错误: {err_msg[:200]}")
                                        yield _make_openai_chunk(
                                            response_id, model_display,
                                            content=f"\n[Studio API 错误] {err_msg}"
                                        )
                            
                            # 如果没有发送 finish chunk，补一个
                            if has_content:
                                yield _make_openai_chunk(
                                    response_id, model_display, finish_reason="stop"
                                )
                            elif not has_content:
                                yield _make_openai_chunk(
                                    response_id, model_display, content="[无响应内容]"
                                )
                                yield _make_openai_chunk(
                                    response_id, model_display, finish_reason="stop"
                                )
                            
                            yield "data: [DONE]\n\n"
                
                except Exception as e:
                    print(f"❌ [batchGraphql] 流式异常: {e}")
                    traceback.print_exc()
                    if not role_sent:
                        yield _make_openai_chunk(response_id, model_display, role="assistant")
                    yield _make_openai_chunk(
                        response_id, model_display, content=f"\n[代理错误] {str(e)}"
                    )
                    yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
        # ========== 非流式处理 ==========
        else:
            try:
                req_headers = build_headers(_get_cookie_string()) or headers
                
                async with httpx.AsyncClient(**client_kwargs) as client:
                    response = await client.post(
                        BATCH_GRAPHQL_URL, headers=req_headers, json=body
                    )
                    
                    if response.status_code != 200:
                        return JSONResponse(status_code=response.status_code, content={
                            "error": {"message": response.text[:500], "type": "upstream_error"}
                        })
                    
                    # 解析所有 JSON 对象，收集文本
                    full_text = ""
                    reasoning_text = ""
                    finish_reason = "stop"
                    
                    raw_text = response.text
                    # 简单提取所有 JSON 对象
                    import io
                    
                    class _FakeResponse:
                        """模拟异步迭代器供 _iter_json_objects 使用"""
                        def __init__(self, text):
                            self._text = text
                            self._done = False
                        
                        async def aiter_text(self):
                            yield self._text
                    
                    fake_resp = _FakeResponse(raw_text)
                    async for obj in _iter_json_objects(fake_resp):
                        for event_type, data in _extract_from_results(obj):
                            if event_type == "text":
                                full_text += data
                            elif event_type == "thought":
                                reasoning_text += data
                            elif event_type == "image":
                                full_text += data
                            elif event_type == "finish":
                                if data == "MAX_TOKENS":
                                    finish_reason = "length"
                            elif event_type == "error":
                                err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                                full_text += f"\n[错误] {err_msg}"
                    
                    if not full_text:
                        full_text = " "
                    
                    # 构建 OpenAI 响应
                    message_obj = {"role": "assistant", "content": full_text}
                    if reasoning_text:
                        message_obj["reasoning_content"] = reasoning_text
                    
                    return JSONResponse(content={
                        "id": response_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_display,
                        "choices": [{
                            "index": 0,
                            "message": message_obj,
                            "finish_reason": finish_reason,
                        }],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }
                    })
            
            except Exception as e:
                print(f"❌ [batchGraphql] 非流式异常: {e}")
                traceback.print_exc()
                return JSONResponse(status_code=500, content={
                    "error": {"message": f"batchGraphql proxy error: {str(e)}", "type": "proxy_error"}
                })
