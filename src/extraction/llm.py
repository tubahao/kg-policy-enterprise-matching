"""LLM interaction utilities — supports OpenAI-compatible APIs (DeepSeek, DashScope, etc.)."""
import requests
import json
import re
import time


def call_llm(
    model,
    user_prompt,
    api_key,
    system_prompt=None,
    max_tokens=1000,
    temperature=0.2,
    base_url=None,
    max_retries=3,
    retry_delay=2,
) -> str:
    """Call an OpenAI-compatible chat completions API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_exception = None
    for attempt in range(max_retries):
        try:
            response = requests.post(base_url, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                # OpenAI-compatible format
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                # DashScope legacy format
                if "output" in data and "text" in data["output"]:
                    return data["output"]["text"]
                raise Exception(f"Unexpected API response format: {data}")
            else:
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text}"
                )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                print(f"  请求失败，{wait_time}秒后重试 (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise Exception(
                    f"API request failed after {max_retries} retries: {str(last_exception)}"
                )

    raise Exception(f"API request failed: {str(last_exception)}")


def extract_json_from_text(text):
    """Extract JSON array from text that may contain markdown code blocks or extra content."""
    # Strip markdown code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_match = re.search(code_block_pattern, text)
    if code_match:
        text = code_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array brackets
    start_idx = text.find("[")
    if start_idx == -1:
        # Try JSON object brackets
        start_idx = text.find("{")
        if start_idx == -1:
            return None
        # Count braces for object
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start_idx : i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
        return None

    # Count brackets for array
    bracket_count = 0
    for i in range(start_idx, len(text)):
        if text[i] == "[":
            bracket_count += 1
        elif text[i] == "]":
            bracket_count -= 1
            if bracket_count == 0:
                json_str = text[start_idx : i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try fixing common issues
                    fixed = re.sub(r",(\s*[\]}])", r"\1", json_str)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
                return None

    return None
