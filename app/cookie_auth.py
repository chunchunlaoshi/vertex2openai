"""
Cookie 认证模块

从 Google Cookie 字符串中提取 SAPISID 并计算 SAPISIDHASH，
实现无需浏览器的直接 API 调用认证。

原理：Google 内部 API 使用 SAPISIDHASH 认证，算法为：
  Authorization: SAPISIDHASH <timestamp>_<SHA1(timestamp + " " + SAPISID + " " + origin)>
  
只需要用户从浏览器中复制出 Cookie 字符串，本模块即可自动：
1. 解析出 SAPISID 值
2. 实时计算 SAPISIDHASH（每次请求重新计算，因为包含时间戳）
3. 构建完整的请求头
4. 构建正确的 API URL
"""

import hashlib
import time
import re
from typing import Optional, Dict


def parse_cookie_value(cookie_str: str, name: str) -> str:
    """从 cookie 字符串中提取指定 cookie 的值"""
    pattern = rf'(?:^|;\s*){re.escape(name)}=([^;]*)'
    match = re.search(pattern, cookie_str)
    return match.group(1).strip() if match else ""


def extract_sapisid(cookie_str: str) -> Optional[str]:
    """
    从 Cookie 中提取 SAPISID
    优先级: SAPISID > __Secure-3PAPISID > __Secure-1PAPISID
    """
    for name in ["SAPISID", "__Secure-3PAPISID", "__Secure-1PAPISID"]:
        val = parse_cookie_value(cookie_str, name)
        if val:
            return val
    return None


def compute_sapisidhash(sapisid: str, origin: str = "https://console.cloud.google.com") -> str:
    """
    计算 Google SAPISIDHASH 认证头
    
    算法: SAPISIDHASH <timestamp>_<SHA1(timestamp + " " + SAPISID + " " + origin)>
    """
    timestamp = int(time.time())
    hash_input = f"{timestamp} {sapisid} {origin}"
    hash_value = hashlib.sha1(hash_input.encode()).hexdigest()
    return f"SAPISIDHASH {timestamp}_{hash_value}"


def build_auth_headers(cookie_str: str) -> Optional[Dict[str, str]]:
    """
    从 Cookie 字符串构建完整的 API 认证请求头
    
    每次调用都会重新计算 SAPISIDHASH（因为包含实时时间戳）
    
    Returns:
        认证请求头字典，若 Cookie 中无 SAPISID 则返回 None
    """
    sapisid = extract_sapisid(cookie_str)
    if not sapisid:
        print("⚠️ [Cookie 认证] Cookie 中未找到 SAPISID 或 __Secure-3PAPISID")
        print("   请确保从 console.cloud.google.com 的已登录页面中复制完整的 Cookie")
        return None
    
    headers = {
        "authorization": compute_sapisidhash(sapisid),
        "cookie": cookie_str,
        "x-goog-authuser": "0",
        "x-origin": "https://console.cloud.google.com",
        "origin": "https://console.cloud.google.com",
        "referer": "https://console.cloud.google.com/",
        "x-same-domain": "1",
        "content-type": "application/json",
    }
    
    return headers


def build_vertex_url(project_id: str, region: str, model_name: str, stream: bool = True) -> str:
    """
    构建 Vertex AI Studio API URL
    
    使用 Google Cloud Console 内部代理域名格式
    """
    action = "streamGenerateContent" if stream else "generateContent"
    url = (
        f"https://{region}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{region}/"
        f"publishers/google/models/{model_name}:{action}"
    )
    if stream:
        url += "?alt=sse"
    return url


def validate_cookie(cookie_str: str) -> dict:
    """
    验证 Cookie 是否有效（包含必要字段）
    
    Returns:
        {"valid": bool, "has_sapisid": bool, "has_sid": bool, "message": str}
    """
    sapisid = extract_sapisid(cookie_str)
    sid = parse_cookie_value(cookie_str, "__Secure-1PSID")
    if not sid:
        sid = parse_cookie_value(cookie_str, "SID")
    
    if sapisid and sid:
        return {"valid": True, "has_sapisid": True, "has_sid": True, 
                "message": "✅ Cookie 有效：包含 SAPISID 和 SID"}
    elif sapisid:
        return {"valid": True, "has_sapisid": True, "has_sid": False,
                "message": "⚠️ Cookie 部分有效：有 SAPISID 但缺少 SID，可能影响稳定性"}
    elif sid:
        return {"valid": False, "has_sapisid": False, "has_sid": True,
                "message": "❌ Cookie 无效：有 SID 但缺少 SAPISID，无法计算认证头"}
    else:
        return {"valid": False, "has_sapisid": False, "has_sid": False,
                "message": "❌ Cookie 无效：未找到必要的认证字段"}
