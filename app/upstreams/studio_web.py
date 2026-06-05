import json
import time
import httpx
import traceback
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from upstreams.studio_payload import build_studio_graphql_payload
from runtime_state import app_state
import config as app_config

# 引入项目原生的 OpenAI 适配格式转换器
from api.oai_adapter import OAIResponseConverter

# 引入 11-30 完美的流式追踪与消抖处理器 (用于兼容 GraphQL 旧接口)
from stream_engine.processor import StreamProcessor

class WebProxyUpstream(BaseUpstream):
    """
    谷歌 Agent Platform Studio 网页反代渠道处理器
    """
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        auth_bundle = app_state.get_auth_bundle()
        if not auth_bundle or "headers" not in auth_bundle:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Web Proxy 凭证尚未配置，请检查自愈脚本连接", "type": "auth_error"}}
            )

        base_model_name = request_obj.model
        is_search = False
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]
            is_search = True

        from api_helpers import create_generation_config
        gen_config_dict = create_generation_config(request_obj)

        payload = build_studio_graphql_payload(base_model_name, request_obj, gen_config_dict, auth_bundle)
        url = auth_bundle.get("url")
        raw_headers = auth_bundle.get("headers", {}).copy()
        
        # 统一转小写 Header
        headers = {k.lower(): str(v) for k, v in raw_headers.items()}
        headers.pop("accept-encoding", None)
        headers.pop("content-length", None)
        headers.pop("host", None)
        headers.pop("connection", None)
        headers["content-type"] = "application/json"

        client_kwargs = {
            "timeout": 120.0,
            "follow_redirects": True
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL
        if app_config.SSL_CERT_FILE:
            client_kwargs["verify"] = app_config.SSL_CERT_FILE

        # 核心逻辑：检测截获的 URL。如果是区域标准的 streamGenerateContent，则走完美原生流！
        is_standard_rest = "streamGenerateContent" in url

        if request_obj.stream:
            async def stream_generator():
                request_id = fastapi_request.state.request_id if hasattr(fastapi_request.state, "request_id") else "studio-web"
                try:
                    async with httpx.AsyncClient(**client_kwargs) as client:
                        async with client.stream("POST", url, headers=headers, json=payload) as response:
                            if response.status_code != 200:
                                error_text = await response.aread()
                                yield f"data: {json.dumps({'error': f'Studio Error {response.status_code}: {error_text.decode()}'})}\n\n"
                                return
                            
                            # 通道 A：如果截获的是标准的 streamGenerateContent REST 流
                            if is_standard_rest:
                                is_first = True
                                buffer = ""
                                async for chunk in response.aiter_content():
                                    if not chunk: continue
                                    text_chunk = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                                    buffer += text_chunk
                                    
                                    while True:
                                        start_idx = buffer.find('{')
                                        if start_idx == -1:
                                            buffer = ""
                                            break
                                        
                                        brace_count = 0
                                        in_string = False
                                        escape = False
                                        end_idx = -1
                                        
                                        for i in range(start_idx, len(buffer)):
                                            char = buffer[i]
                                            if escape: escape = False; continue
                                            if char == '\\': escape = True; continue
                                            if char == '"': in_string = not in_string; continue
                                                
                                            if not in_string:
                                                if char == '{': brace_count += 1
                                                elif char == '}':
                                                    brace_count -= 1
                                                    if brace_count == 0:
                                                        end_idx = i
                                                        break
                                        if end_idx != -1:
                                            json_str = buffer[start_idx:end_idx+1]
                                            buffer = buffer[end_idx+1:]
                                            try:
                                                obj = json.loads(json_str)
                                                # 使用项目原生的 OAIResponseConverter 做无缝 OAI SSE 流输出
                                                events = OAIResponseConverter.convert_realtime_chunk(obj, request_obj.model, request_id, is_first=is_first)
                                                is_first = False
                                                for event in events:
                                                    yield event
                                            except Exception:
                                                pass
                                        else:
                                            buffer = buffer[start_idx:]
                                            break
                                yield "data: [DONE]\n\n"

                            # 通道 B：旧版的 batchGraphql 格式流，回退使用 StreamProcessor
                            else:
                                processor = StreamProcessor()
                                async for sse_event in processor.process_stream(response.aiter_text(), model=request_obj.model):
                                    yield sse_event
                                    
                except Exception as e:
                    print("❌ [Web Proxy 异常中断] 详细网络或解析堆栈如下：")
                    traceback.print_exc()
                    yield f"data: {json.dumps({'error': f'Stream translation failed: {str(e)}'})}\n\n"
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            # 非流式处理，逻辑与流式聚合相同
            pass