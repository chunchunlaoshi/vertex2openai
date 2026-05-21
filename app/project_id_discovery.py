import httpx
import re
import json
from typing import Dict, Optional
import config

# 全局项目 ID 缓存
PROJECT_ID_CACHE: Dict[str, str] = {}

def _get_proxy_url() -> Optional[str]:
    return config.PROXY_URL

async def discover_project_id(api_key: str) -> str:
    """
    通过 httpx 发现项目 ID，完美兼容 SOCKS5/HTTP 代理并共享 SSL 配置
    """
    if api_key in PROJECT_ID_CACHE:
        print(f"INFO: 使用缓存的项目 ID: {PROJECT_ID_CACHE[api_key]}")
        return PROJECT_ID_CACHE[api_key]
    
    # 故意请求一个不存在的模型以触发错误并提取 Project ID
    error_url = "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.7-pro-preview-05-06:streamGenerateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "test"}]}]
    }
    
    # 适配代理
    proxies = None
    proxy_url = _get_proxy_url()
    if proxy_url:
        if proxy_url.startswith("socks"):
            proxies = {"all://": proxy_url}
        else:
            proxies = {"https://": proxy_url}
            
    client_args = {'timeout': 20}
    if proxies:
        client_args['proxies'] = proxies
    if getattr(config, "SSL_CERT_FILE", None):
        client_args['verify'] = config.SSL_CERT_FILE
        
    async with httpx.AsyncClient(**client_args) as client:
        try:
            # 发起请求
            response = await client.post(error_url, params={"key": api_key}, json=payload)
            response_text = response.text
            
            try:
                error_data = response.json()
                if isinstance(error_data, list) and len(error_data) > 0:
                    error_data = error_data[0]
                
                if "error" in error_data:
                    error_message = error_data["error"].get("message", "")
                    match = re.search(r'projects/(\d+)/locations/', error_message)
                    if match:
                        project_id = match.group(1)
                        PROJECT_ID_CACHE[api_key] = project_id
                        print(f"INFO: 成功发现项目 ID: {project_id}")
                        return project_id
            except json.JSONDecodeError:
                match = re.search(r'projects/(\d+)/locations/', response_text)
                if match:
                    project_id = match.group(1)
                    PROJECT_ID_CACHE[api_key] = project_id
                    print(f"INFO: 从原始响应文本中发现项目 ID: {project_id}")
                    return project_id
            
            raise Exception(f"未能发现项目 ID。状态码: {response.status_code}, 响应: {response_text[:500]}")
            
        except Exception as e:
            print(f"ERROR: 发现项目 ID 失败 (本地网络/证书/代理可能存在限制): {e}")
            raise
