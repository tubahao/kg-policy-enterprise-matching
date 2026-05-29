"""
Session 2 — LLM-based multi-label sub-industry classification.

Reads each policy's text_for_llm and the canonical sub-industry list,
calls DeepSeek to classify which 1-3 sub-industries the policy targets.

Output: data/processed/llm_classification_results.json (with checkpoint support)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.llm import call_llm, extract_json_from_text
from src.extraction.config import load_config
from src.extraction.prompts import CLASSIFICATION_SYSTEM_PROMPT, build_classification_user_prompt

DETERMINISTIC_PATH = PROJECT_ROOT / "data" / "processed" / "deterministic_graph_edges.json"
POLICIES_PATH = PROJECT_ROOT / "data" / "processed" / "policies_final.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "llm_classification_results.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "intermediate" / "classification_checkpoint.json"

BATCH_SIZE = 50  # Save checkpoint every N policies


def load_sub_industry_list() -> List[str]:
    with open(DETERMINISTIC_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sub_industry_list"]


def load_policies() -> List[dict]:
    with open(POLICIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint() -> dict:
    """Load existing checkpoint to resume interrupted runs."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_ids": [], "results": []}


def save_checkpoint(completed_ids: List[str], results: List[dict]):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"completed_ids": completed_ids, "results": results}, f,
                  ensure_ascii=False, indent=2)


def classify_policy(
    policy: dict,
    sub_industries: List[str],
    config: dict,
) -> Optional[List[dict]]:
    """Classify a single policy. Returns list of {sub_industry, confidence} or None."""
    text = policy.get("text_for_llm", "")
    if not text or len(text) < 30:
        return []

    system_prompt = CLASSIFICATION_SYSTEM_PROMPT
    user_prompt = build_classification_user_prompt(sub_industries, text)

    llm_cfg = config["llm"]
    cls_cfg = config.get("classification", {})

    try:
        response = call_llm(
            model=llm_cfg["model"],
            user_prompt=user_prompt,
            api_key=llm_cfg["api_key"],
            system_prompt=system_prompt,
            max_tokens=cls_cfg.get("max_tokens", 512),
            temperature=cls_cfg.get("temperature", 0.1),
            base_url=llm_cfg["base_url"],
        )
        result = extract_json_from_text(response)

        if isinstance(result, list):
            # Validate each item has required fields
            validated = []
            for item in result:
                if isinstance(item, dict) and "sub_industry" in item:
                    validated.append({
                        "sub_industry": str(item["sub_industry"]),
                        "confidence": float(item.get("confidence", 0.5)),
                    })
            return validated[:3]  # Cap at 3
        elif isinstance(result, dict) and "sub_industry" in result:
            return [{
                "sub_industry": str(result["sub_industry"]),
                "confidence": float(result.get("confidence", 0.5)),
            }]
        else:
            print(f"  WARNING: Unexpected LLM output format: {str(result)[:200]}")
            return []
    except Exception as e:
        print(f"  ERROR: {e}")
        return None  # None = failed, should retry


def main():
    print("=" * 60)
    print("Session 2 — LLM 受控分类抽取 (Multi-Label Classification)")
    print("=" * 60)

    # Load config
    config = load_config(str(PROJECT_ROOT / "config.toml"))
    if not config:
        print("FATAL: 无法加载 config.toml")
        return

    print(f"\n模型: {config['llm']['model']}")
    print(f"API: {config['llm']['base_url']}")

    # Load data
    print("\n[1/3] 加载数据...")
    sub_industries = load_sub_industry_list()
    print(f"  子行业候选列表: {len(sub_industries)} 个")

    policies = load_policies()
    active = [p for p in policies if p.get("status") == "active"]
    print(f"  Active 政策: {len(active)} 条")

    # Load checkpoint
    ckpt = load_checkpoint()
    completed_ids = set(ckpt["completed_ids"])
    results = ckpt["results"]
    if completed_ids:
        print(f"  从检查点恢复: 已完成 {len(completed_ids)} 条")

    # Filter out completed policies
    remaining = [p for p in active if p["policy_id"] not in completed_ids]
    print(f"  待处理: {len(remaining)} 条")

    if not remaining:
        print("  所有政策已处理完毕!")
    else:
        print(f"\n[2/3] 开始分类 (batch size={BATCH_SIZE})...")
        t_start = time.time()
        failed_ids: List[str] = []

        for i, policy in enumerate(remaining):
            pid = policy["policy_id"]
            title = policy.get("title", "")[:60]

            result = classify_policy(policy, sub_industries, config)

            if result is None:
                failed_ids.append(pid)
                print(f"  [{i+1}/{len(remaining)}] {pid} FAILED — 将重试")
            else:
                results.append({
                    "policy_id": pid,
                    "policy_title": title,
                    "level": policy.get("level", {}).get("type", ""),
                    "targetsSubIndustry": result,
                    "model": config["llm"]["model"],
                    "timestamp": datetime.now().isoformat(),
                })
                if result:
                    labels = ", ".join(f"{r['sub_industry']}({r['confidence']:.2f})" for r in result)
                    print(f"  [{i+1}/{len(remaining)}] {pid} → {labels}")
                else:
                    print(f"  [{i+1}/{len(remaining)}] {pid} → (无特定行业)")

            # Periodic checkpoint
            if (i + 1) % BATCH_SIZE == 0:
                completed_ids.update(r["policy_id"] for r in results)
                save_checkpoint(list(completed_ids), results)
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                print(f"  --- checkpoint saved [{i+1}/{len(remaining)}] "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s ---")

        # Retry failed
        if failed_ids:
            print(f"\n  重试 {len(failed_ids)} 条失败政策...")
            for pid in failed_ids:
                policy = next(p for p in active if p["policy_id"] == pid)
                result = classify_policy(policy, sub_industries, config)
                if result is not None:
                    results.append({
                        "policy_id": pid,
                        "policy_title": policy.get("title", "")[:60],
                        "level": policy.get("level", {}).get("type", ""),
                        "targetsSubIndustry": result,
                        "model": config["llm"]["model"],
                        "timestamp": datetime.now().isoformat(),
                    })
                    print(f"  {pid} RETRY OK")
                else:
                    print(f"  {pid} RETRY STILL FAILED — 跳过")

        # Final save
        completed_ids.update(r["policy_id"] for r in results)
        save_checkpoint(list(completed_ids), results)

        total_time = time.time() - t_start
        print(f"\n  完成! 总耗时: {total_time:.0f}s "
              f"({total_time/len(remaining):.1f}s/policy)")

    # Sort by policy_id for consistent output
    results.sort(key=lambda r: r["policy_id"])

    # Write final output
    print(f"\n[3/3] 写入结果: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Statistics
    with_results = [r for r in results if r["targetsSubIndustry"]]
    avg_targets = sum(len(r["targetsSubIndustry"]) for r in results) / max(len(results), 1)
    avg_conf = sum(
        item["confidence"]
        for r in with_results
        for item in r["targetsSubIndustry"]
    ) / max(sum(len(r["targetsSubIndustry"]) for r in with_results), 1)

    print(f"\n  统计:")
    print(f"    总处理: {len(results)} 条")
    print(f"    有行业靶向: {len(with_results)} 条 ({100*len(with_results)/max(len(results),1):.1f}%)")
    print(f"    无行业靶向: {len(results) - len(with_results)} 条")
    print(f"    平均靶向行业数: {avg_targets:.2f}")
    print(f"    平均置信度: {avg_conf:.3f}")


if __name__ == "__main__":
    main()
