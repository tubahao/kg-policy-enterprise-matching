"""Session 2 — Ontology-constrained classification prompts."""

CLASSIFICATION_SYSTEM_PROMPT = """\
你是一个政策文本分析专家。你的任务是从一篇政策文本中，识别该政策重点扶持或影响的具体细分行业。

规则：
1. 只能从提供的【细分行业列表】中选择，绝对不要编造或修改行业名称。
2. 选择 1~3 个最相关的子行业。如果政策涉及面非常广（如宏观指导性文件），可以选择 0 个并返回空数组 []。
3. 为每个选择提供 0.0~1.0 的置信度（confidence），反映该政策对该行业的扶持力度确定性。
4. 严格按 JSON 格式输出，不要输出任何其他文字。"""


def build_classification_user_prompt(sub_industries: list[str], policy_text: str) -> str:
    """Build the user prompt with the sub-industry taxonomy and policy text."""
    import textwrap

    industry_list = "\n".join(f"- {s}" for s in sub_industries)

    # Truncate policy text to ~6000 chars to stay well within context window
    max_text_len = 6000
    if len(policy_text) > max_text_len:
        truncated_text = policy_text[:max_text_len] + "\n\n[... 政策文本过长，已截断 ...]"
    else:
        truncated_text = policy_text

    return f"""【细分行业列表】
{industry_list}

【政策文本】
{truncated_text}

请输出该政策重点影响的细分行业（JSON 数组格式）：
[{{"sub_industry": "汽车制造业", "confidence": 0.92}}]"""


# ---------------------------------------------------------------------------
# Legacy prompts — retained for reference, no longer used in Session 2 pipeline
# ---------------------------------------------------------------------------

MAIN_SYSTEM_PROMPT = """
You are an advanced AI system specialized in knowledge extraction and knowledge graph generation.
"""

MAIN_USER_PROMPT = """Your task: ..."""  # stub
