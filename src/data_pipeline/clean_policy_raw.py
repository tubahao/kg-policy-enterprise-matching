#!/usr/bin/env python3
"""
Session 2 前置步骤：原始政策数据清洗流水线 (Policy Raw Data Cleaning Pipeline)

实现三个核心策略：
1. 低信息熵节点剪枝 —— 剔除仅含附件占位符的"空壳政策"
2. 原文-解读实体对齐 —— 三级级联匹配 (强规则 → LCS软匹配 → 发布单位+时间约束)
3. 特征级语义融合 —— 原文+解读 [SEP] 拼接，或解读复活空壳节点

输入: data/raw/policy_data.xlsx
输出: data/processed/policies_cleaned.json (符合目标 JSON Schema)
       data/statistics/cleaning_report.json (清洗统计报告)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -------------------------------------------------------
# 1. 常量 & 规则定义
# -------------------------------------------------------

# 解读类标题关键词
INTERPRETATION_KEYWORDS = [
    "解读", "图解", "九问", "一问", "问答", "热点回应", "政策吹风",
    "答记者问", "新闻发布会", "一图", "速览", "文字解读",
    "政策解读", "重点问题回应", "一图读懂",
]

# 空壳政策正则 —— 匹配"附件：xxx.docx"类纯附件内容
ATTACHMENT_SHELL_PATTERNS = [
    re.compile(r"^附件[：:]\s*[\w./]+(\.docx?|\.wps|\.xlsx?|\.zip|\.pdf)", re.IGNORECASE),
    re.compile(r"^附件[：:][\s\S]{0,200}$"),
    re.compile(r"^附件\d+[：:][\s\S]{0,200}$"),
    re.compile(r"^详见附件[\s\S]{0,50}$"),
    re.compile(r"^[\s\S]*附件[：:]\s*\w+\.(doc|docx|pdf|xls|xlsx|wps|zip)[\s\S]{0,50}$"),
]

# 低信息熵长度阈值
LOW_ENTROPY_CHAR_THRESHOLD = 50  # 剔除空白/标点后字符数 < 此值 → 剪枝

# 实体对齐参数
LCS_COVERAGE_THRESHOLD = 0.55      # T2: 最长公共子串覆盖率阈值
SIMILARITY_THRESHOLD = 0.55        # T2: SequenceMatcher 相似度阈值 (兜底)
MAX_DAYS_PROXIMITY = 180           # T3: 发布时间最大间隔(天)


# -------------------------------------------------------
# 2. 数据结构
# -------------------------------------------------------

@dataclass
class PolicyRecord:
    """清洗后的单条政策记录"""
    policy_id: str
    region: str
    title: str
    pub_agency: Optional[str]
    pub_date: Optional[str]
    # 融合后的内容字段
    main_text: str = ""
    interpretation_text: str = ""
    is_main_empty: bool = False
    # 给 Session 2 生成器用的拼接文本
    text_for_llm: str = ""
    # 元数据
    level: Optional[str] = None
    status: str = "active"               # active | pruned_by_low_entropy | pruned_duplicate
    matched_original_id: Optional[str] = None  # 若本条为解读，指向原文 policy_id
    matched_interpretation_id: Optional[str] = None  # 若本条为原文，指向解读 policy_id

    def to_dict(self) -> dict:
        d = {
            "policy_id": self.policy_id,
            "region": self.region,
            "title": self.title,
            "pub_agency": self.pub_agency,
            "pub_date": self.pub_date,
            "fusion_content": {
                "main_text": self.main_text,
                "interpretation_text": self.interpretation_text,
                "is_main_empty": self.is_main_empty,
            },
            "text_for_llm": self.text_for_llm,
            "level": self.level,
            "status": self.status,
        }
        if self.matched_original_id:
            d["matched_original_id"] = self.matched_original_id
        if self.matched_interpretation_id:
            d["matched_interpretation_id"] = self.matched_interpretation_id
        return d


# -------------------------------------------------------
# 3. 文本工具
# -------------------------------------------------------

def _safe_str(val: Any) -> str:
    """Convert value to native Python str, handling NaN and numpy types."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(val, float) and (val != val):
        return ""
    return str(val)


def _safe_optional(val: Any) -> Optional[str]:
    """Convert value to Optional[str], returning None for NaN."""
    try:
        if val is None or pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, float) and (val != val):
        return None
    s = str(val).strip()
    return s if s else None


# -------------------------------------------------------
# 3b. 发布单位填补
# -------------------------------------------------------

# 政务机构后缀
_AGENCY_SUFFIX = (
    r"(?:总局|总署|管理局|委员会|领导小组|办公室|办公厅|税务总局|知识产权局|市场监管总局"
    r"|部|委|局|厅|办|会|署|院|行|社)"
)

# 从标题/内容开头提取发布单位: "{AGENCY}(关于|印发|发布|转发|决定|制定)"
_AGENCY_IN_TITLE = re.compile(
    r"^([一-鿿\s]{3,80}?" + _AGENCY_SUFFIX + r")"
    r"(?:关于|印发|发布|转发|决定|制定了|出台了|印发了|发布了|联合|等\d+部门|（|〔|\()"
)

# 多条机构并排（用空格或中文标点分隔）
_AGENCY_LIST = re.compile(
    r"[一-鿿]{2,18}" + _AGENCY_SUFFIX
)

# 国务院组成部门/直属机构（常见高频机构全称）
_KNOWN_AGENCIES = re.compile(
    r"(?:国家(?:发展和改革委员会|税务总局|知识产权局|市场监督管理总局|"
    r"统计局|医疗保障局|中医药管理局|外汇管理局|文物局|煤矿安全监察局|"
    r"药品监督管理局|国际发展合作署|林业和草原局|铁路局|邮政局|信访局|"
    r"能源局|国防科技工业局|烟草专卖局|移民管理局|海洋局|测绘地理信息局|"
    r"铁路局|民用航空局|矿山安全监察局)"
    r"|财政部|税务总局|科技部|教育部|公安部|民政部|司法部|"
    r"人力资源和社会保障部|自然资源部|生态环境部|住房和城乡建设部|"
    r"交通运输部|水利部|农业农村部|商务部|文化和旅游部|退役军人事务部|"
    r"应急管理部|审计署|中国人民银行|国务院|国务院办公厅|"
    r"工业和信息化部|人力资源社会保障部|国家发展改革委)"
    r"|(?:[一-鿿]{2,10}(?:省|自治区|市|县|区)(?:人民政府(?:办公室|办公厅)?|"
    r"(?:科学技术|工业和信息化|财政|人力资源和社会保[障]|住房和城乡建设|"
    r"交通运输|水利|农业农村|商务|文化和旅游|卫生健康|应急管理|市场监督管理|"
    r"发展和改革|教育|公安|民政|司法|自然资源|生态环境|退役军人事务|审计|统计|"
    r"医疗保障|林业|大数据发展|地方金融监督管理|知识产权|药品监督管理|中医药管理|"
    r"体育|广播电视|新闻出版|粮食和物资储备|能源|国防动员|信访)"
    r"(?:厅|局|委员会|办公室)))"
)

# 解读内容中引用原文发布单位
_AGENCY_IN_INTERP = re.compile(
    r"(?:根据|按照|会同|联合|转发|落实)"
    r"([一-鿿]{2,25}" + _AGENCY_SUFFIX + r")"
    r"(?:发布|印发|制定|出台|下发|通知|文件|意见|办法|规定)"
)


# 法律立法机构模式
_LAW_PATTERN = re.compile(
    r"第[一二三四五六七八九十百千万\d]+届全国人民代表大会(?:常务委员会)?"
    r"第[一二三四五六七八九十百千万\d]+次会议通过"
)

_LAW_PUBLISHER = {
    True: "全国人民代表大会常务委员会",   # 常务委
    False: "全国人民代表大会",            # 全体会议
}


def _extract_law_publisher(text: str) -> Optional[str]:
    """从法律全文中提取立法机构。"""
    if "人民共和国" not in text and "法》" not in text and "促进法" not in text:
        return None
    m = _LAW_PATTERN.search(text)
    if m:
        full_match = m.group()
        if "常务委员会" in full_match:
            return "全国人民代表大会常务委员会"
        return "全国人民代表大会"
    return None


def _extract_from_title_or_content(title: str, content: str) -> Optional[str]:
    """从标题或内容开头提取发布单位。"""
    # 尝试标题
    m = _AGENCY_IN_TITLE.search(title)
    if m:
        agencies = _AGENCY_LIST.findall(m.group(1))
        if agencies:
            return " ".join(agencies)

    if not content:
        return None

    # 尝试内容前 800 字符 (比原先 400 更宽，捕获更深处的机构名)
    content_head = content[:800]

    m = _AGENCY_IN_TITLE.search(content_head)
    if m:
        agencies = _AGENCY_LIST.findall(m.group(1))
        if agencies:
            return " ".join(agencies)

    # 兜底：在内容中直接找已知机构名
    found = _KNOWN_AGENCIES.findall(content_head)
    if found:
        # 去重保序
        seen = set()
        deduped = []
        for a in found:
            if a not in seen:
                seen.add(a)
                deduped.append(a)
        return " ".join(deduped[:3])

    # 法律模式
    law_pub = _extract_law_publisher(content_head)
    if law_pub:
        return law_pub

    return None


def _extract_from_interpretation(text: str) -> Optional[str]:
    """从解读文本中提取被引用的原文发布单位。"""
    if not text:
        return None
    # 解读文本前 1000 字符内查找
    head = text[:1000]
    matches = _AGENCY_IN_INTERP.findall(head)
    if matches:
        return matches[0]
    # 兜底：已知机构全称
    found = _KNOWN_AGENCIES.findall(head)
    if found:
        return found[0]
    # 法律模式
    law_pub = _extract_law_publisher(head)
    if law_pub:
        return law_pub
    return None


def impute_pub_agency(title: str, content: str, interpretation_text: str) -> Optional[str]:
    """
    三级填补策略：
    1. 从标题/正文开头提取（"XXX厅关于印发..."）
    2. 从解读文本中提取引用的原文发布单位
    3. 对独立解读，在内容中搜索被引用机构的痕迹
    若均无法提取，返回 None。
    """
    # S1: 标题 + 正文
    result = _extract_from_title_or_content(title, content)
    if result:
        return result

    # S2: 专用解读文本提取
    if interpretation_text:
        result = _extract_from_interpretation(interpretation_text)
        if result:
            return result

    # S3: 独立解读（内容在 main_text 而非 interpretation_text）→ 尝试解读模式提取
    if not interpretation_text and content:
        result = _extract_from_interpretation(content)
        if result:
            return result

    return None


def clean_text_for_length(text: str) -> str:
    """剔除空白+标点后返回纯字符内容，用于低熵判断。"""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"\s+", "", text)
    cleaned = re.sub(r"[，。、；：！？…—\-\.\,\;\:\!\?\"\'\(\)\[\]【】《》（）\s]", "", cleaned)
    return cleaned


def extract_titles_from_bookmarks(text: str) -> List[str]:
    """从文本中提取所有《》书名号内的标题。"""
    return re.findall(r"《([^》]+)》", text)


def normalize_title_for_match(title: str) -> str:
    """标准化标题用于匹配：去除失效标记、多余空格、书名号。"""
    t = title.strip()
    t = re.sub(r"【已失效】|（已失效）|\[已失效\]", "", t)
    t = re.sub(r"\s+", "", t)
    return t


def strip_bookmarks(title: str) -> str:
    """去除书名号但保留内容。"""
    return title.replace("《", "").replace("》", "")


# -------------------------------------------------------
# 4. 行级分类
# -------------------------------------------------------

def classify_row(title: str, content: str) -> dict:
    """对单行数据做初步分类。"""
    content_str = str(content) if not pd.isna(content) else ""
    title_str = str(title) if not pd.isna(title) else ""

    # 判断是否为解读类
    interp_matches = [kw for kw in INTERPRETATION_KEYWORDS if kw in title_str]
    is_interpretation = len(interp_matches) > 0

    # 判断是否为空壳附件
    is_attachment_shell = False
    content_clean = content_str.replace("\n", "").replace("\r", "").replace(" ", "").strip()
    if len(content_clean) < 200 and ("附件" in content_clean):
        for pat in ATTACHMENT_SHELL_PATTERNS:
            if pat.match(content_clean):
                is_attachment_shell = True
                break
    # 极短内容也标记
    info_chars = clean_text_for_length(content_str)
    if len(info_chars) < LOW_ENTROPY_CHAR_THRESHOLD:
        is_attachment_shell = True

    return {
        "is_interpretation": is_interpretation,
        "is_attachment_shell": is_attachment_shell,
        "interp_keywords": interp_matches,
        "content_len": len(content_str),
        "info_char_len": len(info_chars),
    }


# -------------------------------------------------------
# 5. 三级级联实体对齐
# -------------------------------------------------------

def _t1_strong_rule(interp_title: str, original_titles: List[str]) -> Optional[str]:
    """
    T1 强规则匹配:
    解读标题 = 《原文标题》政策解读 / 关于《原文标题》的解读 / 九问+一图 速览《原文标题》...
    从解读标题中提取《》内的政策名，在原文标题列表中精确查找。
    """
    interp_normalized = normalize_title_for_match(interp_title)

    # 提取解读标题中所有《》内容
    bookmark_titles = extract_titles_from_bookmarks(interp_title)
    if not bookmark_titles:
        bookmark_titles = extract_titles_from_bookmarks(interp_normalized)

    # 同时尝试从 解读《XXX》 这类模式提取
    interpret_of_pattern = re.findall(r"解读[《]([^》]+)[》]", interp_normalized)
    bookmark_titles.extend(interpret_of_pattern)

    # 尝试匹配 "对《XXX》的解读" 模式
    dui_pattern = re.findall(r"对[《]([^》]+)[》]", interp_normalized)
    bookmark_titles.extend(dui_pattern)

    if not bookmark_titles:
        return None

    for extracted in bookmark_titles:
        extracted_norm = normalize_title_for_match(extracted)
        for orig in original_titles:
            orig_norm = normalize_title_for_match(orig)
            # 精确匹配或包含关系
            if extracted_norm == orig_norm:
                return orig
            if len(extracted_norm) > 10 and extracted_norm in orig_norm:
                return orig
            if len(orig_norm) > 10 and orig_norm in extracted_norm:
                return orig
    return None


def _t2_lcs_soft_match(interp_title: str, original_titles: List[str],
                       threshold: float = LCS_COVERAGE_THRESHOLD) -> Optional[str]:
    """
    T2 软文本相似度匹配:
    使用 SequenceMatcher 计算解读标题与每个原文标题的相似度，
    取最高分且超过阈值的结果。
    """
    interp_norm = normalize_title_for_match(interp_title)
    interp_no_book = strip_bookmarks(interp_norm)

    best_match = None
    best_score = 0.0

    for orig in original_titles:
        orig_norm = normalize_title_for_match(orig)
        orig_no_book = strip_bookmarks(orig_norm)

        # 先算 LCS 覆盖率
        sm = SequenceMatcher(None, interp_no_book, orig_no_book)
        lcs_len = 0
        for block in sm.get_matching_blocks():
            lcs_len += block.size
        shorter = min(len(interp_no_book), len(orig_no_book))
        if shorter == 0:
            continue
        lcs_coverage = lcs_len / shorter

        # 再算整体相似度
        sim = SequenceMatcher(None, interp_norm, orig_norm).ratio()

        # 综合得分：LCS 覆盖率 70% + 整体相似度 30%
        combined = 0.7 * lcs_coverage + 0.3 * sim

        if combined > best_score:
            best_score = combined
            best_match = orig

    if best_score >= threshold:
        return best_match
    return None


def _t3_constraint_match(interp_row: pd.Series, original_df: pd.DataFrame) -> Optional[str]:
    """
    T3 兜底约束匹配:
    要求发布单位一致 且 发布时间在 MAX_DAYS_PROXIMITY 天内。
    在满足约束的候选中，用 LCS 取最佳。
    """
    pub_agency = interp_row.get("pub_agency")
    pub_date_str = str(interp_row.get("pub_date", ""))

    if pd.isna(pub_agency) or not pub_date_str:
        return None

    try:
        interp_date = pd.to_datetime(pub_date_str)
    except Exception:
        return None

    candidates = original_df[original_df["pub_agency"] == pub_agency].copy()
    if candidates.empty:
        return None

    interp_title = str(interp_row.get("title", ""))
    interp_norm = normalize_title_for_match(interp_title)
    interp_no_book = strip_bookmarks(interp_norm)

    best_match = None
    best_score = 0.0

    for _, orig_row in candidates.iterrows():
        orig_date_str = str(orig_row.get("pub_date", ""))
        try:
            orig_date = pd.to_datetime(orig_date_str)
        except Exception:
            continue
        if abs((interp_date - orig_date).days) > MAX_DAYS_PROXIMITY:
            continue

        orig_title = str(orig_row.get("title", ""))
        orig_norm = normalize_title_for_match(orig_title)
        orig_no_book = strip_bookmarks(orig_norm)

        sim = SequenceMatcher(None, interp_no_book, orig_no_book).ratio()
        if sim > best_score:
            best_score = sim
            best_match = orig_title

    if best_score >= 0.4:
        return best_match
    return None


def align_interpretations(df: pd.DataFrame) -> Dict[int, Optional[int]]:
    """
    三级级联匹配主函数。

    输入: 原始 DataFrame (已标注分类)
    输出: dict {interp_row_index: original_row_index}，未匹配到的值为 None
    """
    # 拆分为原文池和解读池
    interpretation_mask = df["_is_interpretation"]
    original_mask = ~interpretation_mask

    original_df = df[original_mask].copy()
    interp_df = df[interpretation_mask].copy()

    # T1 和 T2/T3 使用不同的候选池:
    # - T1 强规则: 全量 (精确匹配书名号，标题短也不影响)
    # - T2 LCS: 仅标题 >= 10 字符的原文 (防止极短/泛化标题成为匹配黑洞)
    original_titles_all = original_df["title"].tolist()
    original_df_meaningful = original_df[original_df["title"].apply(
        lambda t: len(strip_bookmarks(normalize_title_for_match(str(t)))) >= 10
    )].copy()
    original_titles_meaningful = original_df_meaningful["title"].tolist()

    alignment: Dict[int, Optional[int]] = {}
    stats = {"t1_matched": 0, "t2_matched": 0, "t3_matched": 0, "unmatched": 0}

    for interp_idx in interp_df.index:
        interp_title = str(interp_df.loc[interp_idx, "title"])
        matched_orig_title = None

        # T1: 强规则匹配 (全量候选)
        result = _t1_strong_rule(interp_title, original_titles_all)
        if result:
            matched_orig_title = result
            stats["t1_matched"] += 1
        else:
            # T2: LCS 软匹配 (仅 meaningful 候选)
            result = _t2_lcs_soft_match(interp_title, original_titles_meaningful)
            if result:
                matched_orig_title = result
                stats["t2_matched"] += 1
            else:
                # T3: 约束匹配 (仅 meaningful 候选)
                result = _t3_constraint_match(interp_df.loc[interp_idx], original_df_meaningful)
                if result:
                    matched_orig_title = result
                    stats["t3_matched"] += 1
                else:
                    stats["unmatched"] += 1

        if matched_orig_title:
            # 从 original_df 反查行索引
            matched_rows = original_df[original_df["title"] == matched_orig_title]
            if not matched_rows.empty:
                alignment[interp_idx] = int(matched_rows.index[0])
            else:
                alignment[interp_idx] = None
        else:
            alignment[interp_idx] = None

    print(f"  实体对齐结果: T1={stats['t1_matched']}, T2={stats['t2_matched']}, "
          f"T3={stats['t3_matched']}, 未匹配={stats['unmatched']}")
    # Convert to native Python ints for JSON safety
    for k in stats:
        stats[k] = int(stats[k])
    return alignment, stats


# -------------------------------------------------------
# 6. 语义融合
# -------------------------------------------------------

def build_fusion_text(main_text: str, interpretation_text: str, is_main_empty: bool) -> str:
    """构建融合文本用于 LLM 抽取。"""
    if is_main_empty and interpretation_text:
        return f"【官方解读】{interpretation_text}"
    if main_text and interpretation_text:
        return f"【政策原文】{main_text}\n【官方解读】{interpretation_text}"
    if main_text:
        return f"【政策原文】{main_text}"
    return ""


# -------------------------------------------------------
# 7. 主流水线
# -------------------------------------------------------

def clean_policy_data(
    input_path: Path,
    output_path: Path,
    stats_path: Path,
) -> List[PolicyRecord]:
    """主清洗流水线。"""

    # ---- 7.1 加载 ----
    print("=" * 60)
    print("政策原始数据清洗流水线")
    print("=" * 60)
    print(f"\n[1/6] 加载原始数据: {input_path}")
    df = pd.read_excel(input_path)
    df.columns = ["region", "title", "pub_date", "pub_agency", "content"]
    total_raw = len(df)
    print(f"  原始记录数: {total_raw}")

    # ---- 7.2 行级分类 ----
    print(f"\n[2/6] 行级分类...")
    classifications = []
    for _, row in df.iterrows():
        cls = classify_row(str(row["title"]), str(row["content"]) if not pd.isna(row["content"]) else "")
        classifications.append(cls)

    df["_is_interpretation"] = [c["is_interpretation"] for c in classifications]
    df["_is_attachment_shell"] = [c["is_attachment_shell"] for c in classifications]
    df["_interp_keywords"] = [c["interp_keywords"] for c in classifications]
    df["_info_char_len"] = [c["info_char_len"] for c in classifications]

    n_interpretation = int(df["_is_interpretation"].sum())
    n_attachment_shell = int(df["_is_attachment_shell"].sum())
    n_normal = int(total_raw - df["_is_interpretation"].sum() - df["_is_attachment_shell"].sum())

    print(f"  原文政策: {n_normal}")
    print(f"  解读文章: {n_interpretation}")
    print(f"  空壳附件: {n_attachment_shell}")

    # ---- 7.3 实体对齐 ----
    print(f"\n[3/6] 三级级联实体对齐 (原文 <-> 解读)...")
    alignment, alignment_stats = align_interpretations(df)

    # ---- 7.4 融合 & 剪枝 ----
    print(f"\n[4/6] 语义融合与剪枝...")
    records: List[PolicyRecord] = []
    next_id = 0

    # 追踪已匹配的原文 → 解读 (一对多: 一篇原文可有多篇解读)
    orig_to_interps: Dict[int, List[int]] = {}   # orig_idx → [interp_idx, ...]
    for interp_idx, orig_idx in alignment.items():
        if orig_idx is not None:
            orig_to_interps.setdefault(orig_idx, []).append(interp_idx)

    processed_orig = set()
    processed_interp = set()

    # 4a. 处理原文
    for idx in df.index:
        if df.loc[idx, "_is_interpretation"]:
            continue

        title = _safe_str(df.loc[idx, "title"])
        content = _safe_str(df.loc[idx, "content"])
        is_shell = bool(df.loc[idx, "_is_attachment_shell"])
        matched_interp_indices = orig_to_interps.get(idx, [])

        # 剪枝判断
        should_prune = False
        resurrected = False
        interpretation_text = ""

        if is_shell and not matched_interp_indices:
            should_prune = True
        elif is_shell and matched_interp_indices:
            # 空壳但有解读 → 复活: 用解读内容替换原文
            interp_contents = [_safe_str(df.loc[i, "content"]) for i in matched_interp_indices]
            content = "\n\n".join(interp_contents)
            interpretation_text = content
            resurrected = True
            for i in matched_interp_indices:
                processed_interp.add(i)
        elif not is_shell and matched_interp_indices:
            # 正常原文 + 有解读 → 融合: 拼接多篇解读
            interp_contents = [_safe_str(df.loc[i, "content"]) for i in matched_interp_indices]
            interpretation_text = "\n\n---\n\n".join(interp_contents)
            for i in matched_interp_indices:
                processed_interp.add(i)

        if should_prune:
            rec = PolicyRecord(
                policy_id=f"P_{next_id:04d}",
                region=_safe_str(df.loc[idx, "region"]),
                title=title,
                pub_agency=_safe_optional(df.loc[idx, "pub_agency"]),
                pub_date=_safe_optional(df.loc[idx, "pub_date"]),
                main_text=content,
                interpretation_text="",
                is_main_empty=True,
                text_for_llm="",
                status="pruned_by_low_entropy",
            )
        else:
            main_empty = is_shell and resurrected
            text_llm = build_fusion_text(content, interpretation_text, main_empty)

            matched_interp_id = None

            rec = PolicyRecord(
                policy_id=f"P_{next_id:04d}",
                region=_safe_str(df.loc[idx, "region"]),
                title=title,
                pub_agency=_safe_optional(df.loc[idx, "pub_agency"]),
                pub_date=_safe_optional(df.loc[idx, "pub_date"]),
                main_text="" if main_empty else content,
                interpretation_text=interpretation_text,
                is_main_empty=main_empty,
                text_for_llm=text_llm,
                status="active",
                matched_interpretation_id=matched_interp_id if interpretation_text else None,
            )

        records.append(rec)
        processed_orig.add(idx)
        next_id += 1

    # 4b. 处理未匹配的解读（保留为独立政策）
    for idx in df.index:
        if not df.loc[idx, "_is_interpretation"]:
            continue
        if idx in processed_interp:
            continue  # 已融合到原文中

        title = _safe_str(df.loc[idx, "title"])
        content = _safe_str(df.loc[idx, "content"])

        rec = PolicyRecord(
            policy_id=f"P_{next_id:04d}",
            region=_safe_str(df.loc[idx, "region"]),
            title=title,
            pub_agency=_safe_optional(df.loc[idx, "pub_agency"]),
            pub_date=_safe_optional(df.loc[idx, "pub_date"]),
            main_text=content,
            interpretation_text="",
            is_main_empty=False,
            text_for_llm=f"【政策解读】{content}",
            status="active",
            matched_original_id=None,
        )
        records.append(rec)
        processed_interp.add(idx)
        next_id += 1

    # ---- 4c. 发布单位填补 ----
    print(f"\n[4.5/6] 发布单位填补...")
    n_imputed = 0
    n_still_null = 0
    for r in records:
        if r.pub_agency is None and r.status == "active":
            imputed = impute_pub_agency(r.title, r.main_text, r.interpretation_text)
            if imputed:
                r.pub_agency = imputed
                n_imputed += 1
            else:
                n_still_null += 1
                r.status = "pruned_by_missing_publisher"  # 仍缺失 → 移除
    print(f"  填补: {n_imputed}, 仍缺失(已剪枝): {n_still_null}")

    # ---- 4d. 政策级别映射 ----
    print(f"\n[4.6/6] 政策级别映射...")
    _LEVEL_MAP = {
        "国家": {"type": "Policy1", "level_index": 1},
        "自治区": {"type": "Policy2", "level_index": 2},
        "柳州": {"type": "Policy3", "level_index": 3},
    }
    for r in records:
        mapped = _LEVEL_MAP.get(r.region)
        if mapped:
            r.level = mapped
        elif r.status == "active":
            r.level = {"type": "Policy3", "level_index": 3}
    # 统计级别分布
    lv1 = sum(1 for r in records if r.status == "active" and r.level and r.level.get("type") == "Policy1")
    lv2 = sum(1 for r in records if r.status == "active" and r.level and r.level.get("type") == "Policy2")
    lv3 = sum(1 for r in records if r.status == "active" and r.level and r.level.get("type") == "Policy3")
    print(f"  Policy1 (国家级): {lv1}, Policy2 (自治区级): {lv2}, Policy3 (市县级): {lv3}")

    # ---- 最终计数 ----
    n_active = sum(1 for r in records if r.status == "active")
    n_pruned_entropy = sum(1 for r in records if r.status == "pruned_by_low_entropy")
    n_pruned_publisher = sum(1 for r in records if r.status == "pruned_by_missing_publisher")
    n_resurrected = sum(1 for r in records if r.status == "active" and r.is_main_empty)
    n_fused = sum(1 for r in records if r.status == "active" and r.interpretation_text)

    print(f"\n  最终记录: {len(records)} (active={n_active}, pruned={n_pruned_entropy + n_pruned_publisher})")
    print(f"  其中复活空壳: {n_resurrected}, 原文-解读融合: {n_fused}")

    # ---- 7.5 输出 JSON ----
    print(f"\n[5/6] 输出清洗后数据: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = [r.to_dict() for r in records]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  输出 {len(output_data)} 条记录")

    # ---- 7.6 统计报告 ----
    print(f"\n[6/6] 生成统计报告: {stats_path}")
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # 详细统计
    content_lens = [len(r.main_text) + len(r.interpretation_text) for r in records if r.status == "active"]

    report = {
        "pipeline": "policy_raw_cleaning",
        "timestamp": datetime.now().isoformat(),
        "input": {
            "file": str(input_path),
            "total_raw_rows": total_raw,
        },
        "classification": {
            "normal_policies": n_normal,
            "interpretations": n_interpretation,
            "attachment_shells": n_attachment_shell,
        },
        "alignment": {k: int(v) for k, v in alignment_stats.items()},
        "output": {
            "total_records": len(records),
            "active": n_active,
            "pruned_by_low_entropy": n_pruned_entropy,
            "pruned_by_missing_publisher": n_pruned_publisher,
            "resurrected_shells": n_resurrected,
            "fused_with_interpretation": n_fused,
            "pub_agency_imputed": n_imputed,
            "pub_agency_still_null": n_still_null,
            "level_distribution": {
                "Policy1": sum(1 for r in records if r.status == "active" and r.level and r.level.get("type") == "Policy1"),
                "Policy2": sum(1 for r in records if r.status == "active" and r.level and r.level.get("type") == "Policy2"),
                "Policy3": sum(1 for r in records if r.status == "active" and r.level and r.level.get("type") == "Policy3"),
            },
            "standalone_interpretations": sum(
                1 for r in records
                if r.status == "active" and not r.interpretation_text
                and r.text_for_llm.startswith("【政策解读】")
            ),
        },
        "content_statistics": {
            "avg_fusion_length": round(sum(content_lens) / len(content_lens), 1) if content_lens else 0,
            "median_fusion_length": sorted(content_lens)[len(content_lens) // 2] if content_lens else 0,
            "min_fusion_length": min(content_lens) if content_lens else 0,
            "max_fusion_length": max(content_lens) if content_lens else 0,
        },
        "pruned_examples": [
            {
                "policy_id": r.policy_id,
                "title": r.title,
                "content_preview": r.main_text[:120],
            }
            for r in records if r.status == "pruned_by_low_entropy"
        ][:10],
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("清洗完成!")
    print("=" * 60)
    print(f"  输入: {total_raw} 条 → 输出: {n_active} active + {n_pruned_entropy + n_pruned_publisher} pruned")
    print(f"  清洗后数据: {output_path}")
    print(f"  统计报告:   {stats_path}")

    return records


# -------------------------------------------------------
# 8. CLI 入口
# -------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="原始政策数据清洗流水线")
    parser.add_argument(
        "--input", type=str, default="data/raw/policy_data.xlsx",
        help="原始政策 Excel 路径"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/policies_cleaned.json",
        help="输出 JSON 路径"
    )
    parser.add_argument(
        "--stats", type=str, default="data/statistics/cleaning_report.json",
        help="统计报告输出路径"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    clean_policy_data(
        input_path=project_root / args.input,
        output_path=project_root / args.output,
        stats_path=project_root / args.stats,
    )


if __name__ == "__main__":
    main()
