import copy
from typing import Any
from pydantic import BaseModel
from models import OpenAIRequest
from message_processing import create_gemini_prompt

def serialize_pydantic(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    elif isinstance(obj, dict):
        return {k: serialize_pydantic(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_pydantic(x) for x in obj]
    return obj

KEY_MAP = {
    "max_output_tokens": "maxOutputTokens",
    "stop_sequences": "stopSequences",
    "top_p": "topP",
    "top_k": "topK",
    "candidate_count": "candidateCount",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
    "response_mime_type": "responseMimeType",
    "thinking_config": "thinkingConfig",
    "include_thoughts": "includeThoughts",
    "thinking_budget": "thinkingBudget",
    "thinking_level": "thinkingLevel",
    "image_config": "imageConfig",
    "image_size": "imageSize",
    "aspect_ratio": "aspectRatio",
    "safety_settings": "safetySettings",
    "system_instruction": "systemInstruction",
    "inline_data": "inlineData",
    "mime_type": "mimeType",
    "function_call": "functionCall",
    "function_response": "functionResponse"
}

def convert_keys_to_camel(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = KEY_MAP.get(k, k)
            new_dict[new_key] = convert_keys_to_camel(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys_to_camel(x) for x in obj]
    return obj

def build_studio_graphql_payload(model_name: str, request: OpenAIRequest, gen_config_dict: dict, auth_bundle: dict) -> dict:
    """
    【双模克隆载荷生成器】
    完美兼容区域化 REST 接口 (streamGenerateContent) 和旧版 GraphQL (batchGraphql)
    """
    payload = copy.deepcopy(auth_bundle.get("body", {}))
    
    # 编译 OpenAI 消息历史，并完成 Pydantic 转换
    raw_contents = create_gemini_prompt(request.messages)
    camel_contents = convert_keys_to_camel(serialize_pydantic(raw_contents))

    # 1. 兼容模式 A：旧版 batchGraphql 格式 (含有 variables 节点)
    if "variables" in payload:
        variables = payload["variables"]
        variables["contents"] = camel_contents
        
        # 动态替换模型资源标识（自动保留 locations/us-central1 或 global 等区域指纹）
        harvested_model = variables.get("model", "")
        if harvested_model and "/" in harvested_model:
            parts = harvested_model.split("/")
            parts[-1] = model_name
            variables["model"] = "/".join(parts)
        else:
            variables["model"] = model_name
            
        if "generationConfig" not in variables:
            variables["generationConfig"] = {}
        gc = variables["generationConfig"]
        
        if request.temperature is not None: gc["temperature"] = request.temperature
        if request.max_tokens is not None: gc["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None: gc["topP"] = request.top_p
        if request.stop is not None: gc["stopSequences"] = request.stop

        if "gemini-3" in model_name or "gemini-2.5" in model_name:
            gc["thinkingConfig"] = {"includeThoughts": True, "thinkingLevel": "MEDIUM"}
        else:
            gc.pop("thinkingConfig", None)

        system_texts = [m.content for m in request.messages if m.role == "system" and isinstance(m.content, str)]
        if system_texts:
            variables["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}

    # 2. 兼容模式 B：区域化标准的 streamGenerateContent REST 格式 (直接包含 contents)
    else:
        payload["contents"] = camel_contents
        
        if "generationConfig" not in payload:
            payload["generationConfig"] = {}
        gc = payload["generationConfig"]
        
        if request.temperature is not None: gc["temperature"] = request.temperature
        if request.max_tokens is not None: gc["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None: gc["topP"] = request.top_p
        if request.stop is not None: gc["stopSequences"] = request.stop

        if "gemini-3" in model_name or "gemini-2.5" in model_name:
            gc["thinkingConfig"] = {"includeThoughts": True, "thinkingLevel": "MEDIUM"}
        else:
            gc.pop("thinkingConfig", None)

        system_texts = [m.content for m in request.messages if m.role == "system" and isinstance(m.content, str)]
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}
        
    return payload