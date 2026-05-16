import time
from fastapi import APIRouter, Depends, Request
from typing import List, Dict, Any, Set
from auth import get_api_key
from model_loader import get_vertex_models, get_vertex_express_models, refresh_models_config_cache
from credentials_manager import CredentialManager

router = APIRouter()
_last_model_fetch_time = 0  # 增加全局缓存时间戳

@router.get("/v1/models")
async def list_models(fastapi_request: Request, api_key: str = Depends(get_api_key)):
    global _last_model_fetch_time
    
    # 【Bug 修复】：1小时内只向 GitHub 请求一次，彻底避免被封杀和连接超时
    current_time = time.time()
    if current_time - _last_model_fetch_time > 3600:
        await refresh_models_config_cache()
        _last_model_fetch_time = current_time
    
    PAY_PREFIX = "[PAY]"
    EXPRESS_PREFIX = "[EXPRESS] "
    OPENAI_DIRECT_SUFFIX = "-openai"
    OPENAI_SEARCH_SUFFIX = "-openaisearch"
    
    credential_manager_instance: CredentialManager = fastapi_request.app.state.credential_manager
    express_key_manager_instance = fastapi_request.app.state.express_key_manager

    has_sa_creds = credential_manager_instance.get_total_credentials() > 0
    has_express_key = express_key_manager_instance.get_total_keys() > 0

    raw_vertex_models = await get_vertex_models()
    raw_express_models = await get_vertex_express_models()
    
    final_model_list: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()

    def add_model_and_variants(base_id: str, prefix: str):
        suffixes = [""] 
        if "gemini" in base_id.lower():
            suffixes.append(OPENAI_DIRECT_SUFFIX)
            if not base_id.startswith("gemini-2.0"):
                suffixes.extend(["-search", OPENAI_SEARCH_SUFFIX])

        for suffix in suffixes:
            model_id_with_suffix = f"{base_id}{suffix}"
            final_id = f"{prefix}{model_id_with_suffix}" if "-exp-" not in base_id else model_id_with_suffix

            if final_id not in processed_ids:
                final_model_list.append({
                    "id": final_id,
                    "object": "model",
                    "created": int(current_time),
                    "owned_by": "google",
                    "permission": [],
                    "root": base_id,
                    "parent": None
                })
                processed_ids.add(final_id)

    if has_express_key:
        for model_id in raw_express_models: add_model_and_variants(model_id, EXPRESS_PREFIX)
    if has_sa_creds:
        for model_id in raw_vertex_models: add_model_and_variants(model_id, PAY_PREFIX)

    return {"object": "list", "data": sorted(final_model_list, key=lambda x: x['id'])}
