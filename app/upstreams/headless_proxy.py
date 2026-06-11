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
import asyncio
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

# ========== 重试配置 ==========
MAX_RETRIES = 10
RETRY_BACKOFF = [5] * 10  # 每次重试等待秒数

# 可重试的错误关键词（429 限流类）
RETRYABLE_KEYWORDS = [
    "resource exhausted",
    "try again later",
    "429",
    "quota",
    "rate limit",
    "overloaded",
    "temporarily unavailable",
    "internal error",
]

# Cookie 过期/权限失效的错误关键词（不可重试，需要刷新 Cookie）
COOKIE_EXPIRED_KEYWORDS = [
    "permission",
    "denied",
    "aiplatform.endpoints.predict",
    "not authorized",
    "unauthenticated",
    "login required",
    "session expired",
    "invalid credentials",
]

COOKIE_REFRESH_HINT = (
    "\n\n💡 Cookie 可能已过期（PSIDTS 约 1-2 小时有效）。"
    "请重新获取：在电脑浏览器打开 console.cloud.google.com，"
    "按 F12 打开控制台，输入 copy(document.cookie) 回车，"
    "然后到大盘粘贴新 Cookie。"
)


def _is_retryable_error(error_msg: str) -> bool:
    """判断错误是否可重试（429 限流类）"""
    lower = error_msg.lower()
    return any(kw in lower for kw in RETRYABLE_KEYWORDS)


def _is_cookie_expired_error(error_msg: str) -> bool:
    """判断是否为 Cookie 过期/权限失效错误"""
    lower = error_msg.lower()
    return any(kw in lower for kw in COOKIE_EXPIRED_KEYWORDS)


# ========== requestContext 模板 ==========

def _get_experiment_flags() -> str:
    """获取 experimentFlagsBinary，优先使用配置，其次从无头浏览器捕获的凭证包中提取"""
    if app_config.EXPERIMENT_FLAGS:
        return app_config.EXPERIMENT_FLAGS
    
    bundle = app_state.get_auth_bundle()
    if bundle and "body" in bundle and isinstance(bundle["body"], dict):
        req_ctx = bundle["body"].get("requestContext")
        if req_ctx and isinstance(req_ctx, dict):
            flags = req_ctx.get("experimentFlagsBinary")
            if flags:
                return flags
    return ""


def _build_request_context(project_id: str) -> dict:
    """
    构建 batchGraphql 的 requestContext
    
    包含 experimentFlagsBinary，这是 Express Mode 权限的关键标识。
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
        "experimentFlagsBinary": _get_experiment_flags(),
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
        
        gemini_role = "user" if role == "user" else "model"
        
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
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
    """构建完整的 batchGraphql 请求体"""
    contents, system_text = _convert_messages_to_contents(request.messages)
    
    model_path = f"projects/{project_id}/locations/global/publishers/google/models/{model_name}"
    
    gen_config = {
        "temperature": request.temperature if request.temperature is not None else 1,
        "topP": request.top_p if request.top_p is not None else 0.95,
        "maxOutputTokens": request.max_tokens if request.max_tokens is not None else 65535,
    }
    
    if any(kw in model_name for kw in ("gemini-3", "gemini-2.5")):
        gen_config["thinkingConfig"] = {
            "thinkingLevel": "MEDIUM",
            "includeThoughts": True
        }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    ]
    
    variables = {
        "contents": contents,
        "model": model_path,
        "generationConfig": gen_config,
        "safetySettings": safety_settings,
    }
    
    if system_text:
        variables["systemInstruction"] = {"parts": [{"text": system_text}]}
    
    if request.stop:
        gen_config["stopSequences"] = request.stop if isinstance(request.stop, list) else [request.stop]
    
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
    """从 batchGraphql 流式响应中逐个提取完整 JSON 对象"""
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
    """
    if "error" in obj:
        yield ("error", obj["error"])
        return
    
    results = obj.get("results", [])
    for result in results:
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
    return app_config.GOOGLE_COOKIE or app_state.get_google_cookie() or ""

def _get_project_id() -> str:
    return app_config.GOOGLE_PROJECT_ID or app_state.get_project_id() or ""


# ========== 单次请求执行（供重试包装器调用） ==========

async def _execute_stream_request(
    client: httpx.AsyncClient,
    headers: dict,
    body: dict,
    model_display: str,
    response_id: str,
    attempt: int,
):
    """
    执行单次流式请求，返回 (success, events_list, error_msg, is_retryable)
    
    events_list: 成功时的 SSE 事件列表
    error_msg: 失败时的错误消息
    is_retryable: 该错误是否可重试
    """
    events = []
    has_content = False
    retryable_error = None
    
    try:
        async with client.stream("POST", BATCH_GRAPHQL_URL,
                                  headers=headers, json=body) as response:
            
            # HTTP 级别错误
            if response.status_code != 200:
                error_text = await response.aread()
                error_msg = error_text.decode('utf-8', errors='replace')[:1000]
                
                # 401/403 = Cookie 过期
                if response.status_code in (401, 403) or _is_cookie_expired_error(error_msg):
                    print(f"🔑 [Studio] HTTP {response.status_code} Cookie 过期/权限错误 (尝试 {attempt+1})")
                    return False, [], error_msg + COOKIE_REFRESH_HINT, False
                
                is_retryable = response.status_code in (429, 503, 500) or _is_retryable_error(error_msg)
                print(f"{'⚠️' if is_retryable else '❌'} [Studio] HTTP {response.status_code} (尝试 {attempt+1}): {error_msg[:150]}")
                return False, [], error_msg, is_retryable
            
            # 解析流式响应
            async for obj in _iter_json_objects(response):
                for event_type, data in _extract_from_results(obj):
                    if event_type == "text":
                        events.append(_make_openai_chunk(response_id, model_display, content=data))
                        has_content = True
                    
                    elif event_type == "thought":
                        events.append(_make_openai_chunk(response_id, model_display, reasoning_content=data))
                        has_content = True
                    
                    elif event_type == "image":
                        events.append(_make_openai_chunk(response_id, model_display, content=data))
                        has_content = True
                    
                    elif event_type == "finish":
                        fr = "stop" if data == "STOP" else "length" if data == "MAX_TOKENS" else "stop"
                        events.append(_make_openai_chunk(response_id, model_display, finish_reason=fr))
                    
                    elif event_type == "error":
                        err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        
                        # Cookie 过期/权限失效 → 不可重试，立即返回并提示刷新
                        if _is_cookie_expired_error(err_msg) and not has_content:
                            print(f"🔑 [Studio] Cookie 过期/权限错误: {err_msg[:150]}")
                            return False, [], err_msg + COOKIE_REFRESH_HINT, False
                        
                        # 429 限流类 → 可重试
                        if _is_retryable_error(err_msg) and not has_content:
                            print(f"⚠️ [Studio] 429/限流 (尝试 {attempt+1}): {err_msg[:150]}")
                            return False, [], err_msg, True
                        
                        # 其他 API 错误，加入事件流
                        print(f"❌ [Studio] API 错误: {err_msg[:200]}")
                        events.append(_make_openai_chunk(
                            response_id, model_display,
                            content=f"\n[Studio API 错误] {err_msg}"
                        ))
            
            return True, events, None, False
    
    except Exception as e:
        err_msg = str(e)
        is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()
        print(f"{'⚠️' if is_retryable else '❌'} [Studio] 异常 (尝试 {attempt+1}): {err_msg[:150]}")
        return False, [], err_msg, is_retryable


# ========== 主代理类 ==========

class HeadlessProxyUpstream(BaseUpstream):
    """
    batchGraphql 直连代理
    
    使用 Cookie + SAPISIDHASH 鉴权，
    通过 batchGraphql 端点调用 Agent Platform Studio Express Mode 模型。
    支持自动重试 429 限流错误。
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
        
        # ===== 2. 构建请求头 =====
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
        
        # ===== 4. HTTP 客户端配置 =====
        client_kwargs = {
            "timeout": httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=10.0),
            "follow_redirects": True,
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL
        
        is_stream = request_obj.stream
        response_id = f"chatcmpl-studio-{int(time.time())}"
        start_time = time.time()
        
        # 打印请求日志
        msg_count = len(request_obj.messages)
        print(f"→ [Studio] {base_model_name} | {msg_count} 条消息 | {'流式' if is_stream else '非流式'}")
        
        # ========== 流式处理（带自动重试） ==========
        if is_stream:
            async def stream_generator():
                nonlocal start_time
                
                for attempt in range(MAX_RETRIES + 1):
                    # 每次重试重新构建请求（新的 requestContext + 新的 SAPISIDHASH）
                    body = _build_batch_graphql_body(project_id, base_model_name, request_obj)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    
                    async with httpx.AsyncClient(**client_kwargs) as client:
                        success, events, error_msg, is_retryable = await _execute_stream_request(
                            client, req_headers, body, model_display, response_id, attempt
                        )
                    
                    if success:
                        # 成功 - 发送 role chunk + 所有事件 + DONE
                        elapsed = time.time() - start_time
                        print(f"✅ [Studio] {base_model_name} | {len(events)} 块 | {elapsed:.1f}s")
                        
                        yield _make_openai_chunk(response_id, model_display, role="assistant")
                        for event in events:
                            yield event
                        
                        # 确保有 finish chunk
                        has_finish = any('"finish_reason":' in e and '"stop"' in e or '"length"' in e for e in events)
                        if not has_finish:
                            yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                        
                        yield "data: [DONE]\n\n"
                        return
                    
                    elif is_retryable and attempt < MAX_RETRIES:
                        # 可重试错误 - 等待后重试
                        wait_sec = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else RETRY_BACKOFF[-1]
                        print(f"🔄 [Studio] {wait_sec}s 后重试 ({attempt+2}/{MAX_RETRIES+1})...")
                        await asyncio.sleep(wait_sec)
                        start_time = time.time()  # 重置计时
                        continue
                    
                    else:
                        # 不可重试 或 重试次数用尽
                        elapsed = time.time() - start_time
                        if attempt >= MAX_RETRIES:
                            print(f"❌ [Studio] {base_model_name} | 重试 {MAX_RETRIES} 次后仍失败 | {elapsed:.1f}s")
                        else:
                            print(f"❌ [Studio] {base_model_name} | 不可重试错误 | {elapsed:.1f}s")
                        
                        yield _make_openai_chunk(response_id, model_display, role="assistant")
                        yield _make_openai_chunk(
                            response_id, model_display,
                            content=f"[Studio 错误] {error_msg[:500]}"
                        )
                        yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
        # ========== 非流式处理（带自动重试） ==========
        else:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    body = _build_batch_graphql_body(project_id, base_model_name, request_obj)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    
                    async with httpx.AsyncClient(**client_kwargs) as client:
                        response = await client.post(
                            BATCH_GRAPHQL_URL, headers=req_headers, json=body
                        )
                    
                    # HTTP 级别 429
                    if response.status_code in (429, 503, 500):
                        if attempt < MAX_RETRIES:
                            wait_sec = RETRY_BACKOFF[attempt]
                            print(f"⚠️ [Studio] HTTP {response.status_code} (尝试 {attempt+1}), {wait_sec}s 后重试...")
                            await asyncio.sleep(wait_sec)
                            continue
                    
                    if response.status_code != 200:
                        elapsed = time.time() - start_time
                        print(f"❌ [Studio] {base_model_name} | HTTP {response.status_code} | {elapsed:.1f}s")
                        return JSONResponse(status_code=response.status_code, content={
                            "error": {"message": response.text[:500], "type": "upstream_error"}
                        })
                    
                    # 解析响应
                    full_text = ""
                    reasoning_text = ""
                    finish_reason = "stop"
                    api_error = None
                    
                    class _FakeResponse:
                        def __init__(self, text):
                            self._text = text
                        async def aiter_text(self):
                            yield self._text
                    
                    fake_resp = _FakeResponse(response.text)
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
                                if _is_retryable_error(err_msg) and attempt < MAX_RETRIES:
                                    api_error = err_msg
                                    break
                                full_text += f"\n[错误] {err_msg}"
                    
                    # 可重试的 API 错误
                    if api_error and attempt < MAX_RETRIES:
                        wait_sec = RETRY_BACKOFF[attempt]
                        print(f"⚠️ [Studio] 429/限流 (尝试 {attempt+1}): {api_error[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue
                    
                    if not full_text:
                        full_text = " "
                    
                    elapsed = time.time() - start_time
                    text_len = len(full_text)
                    print(f"✅ [Studio] {base_model_name} | {text_len} 字符 | {elapsed:.1f}s")
                    
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
                    err_msg = str(e)
                    is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()
                    
                    if is_retryable and attempt < MAX_RETRIES:
                        wait_sec = RETRY_BACKOFF[attempt]
                        print(f"⚠️ [Studio] 异常 (尝试 {attempt+1}): {err_msg[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue
                    
                    elapsed = time.time() - start_time
                    print(f"❌ [Studio] {base_model_name} | 异常 | {elapsed:.1f}s: {err_msg[:150]}")
                    traceback.print_exc()
                    return JSONResponse(status_code=500, content={
                        "error": {"message": f"batchGraphql proxy error: {err_msg}", "type": "proxy_error"}
                    })
            
            # 所有重试用尽
            elapsed = time.time() - start_time
            print(f"❌ [Studio] {base_model_name} | 重试 {MAX_RETRIES} 次后仍失败 | {elapsed:.1f}s")
            return JSONResponse(status_code=429, content={
                "error": {"message": "请求被限流，已重试多次仍失败。请稍后再试。", "type": "rate_limit_error"}
            })
