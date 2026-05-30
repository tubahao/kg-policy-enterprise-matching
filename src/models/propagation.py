#!/usr/bin/env python3
"""Session 4 — Step 5 (Revised): 异构 HT-PPR 传播引擎

Heterogeneous Hierarchical-Temporal Personalized PageRank Engine.

核心架构:
    1. 全局统一 ID 空间 — 将 4 类异构图节点映射到连续 [0, N-1]
    2. 异构转移矩阵 — 手工构建含置信度与反向边的全局稀疏矩阵 (scipy.sparse)
    3. 幂迭代 PPR — 在全局矩阵上运行带重启的 PageRank
    4. 独立 Min-Max — Policy / Enterprise 分数各自归一化，不做混合归一化
    5. 企业节点仅 Min-Max 归一化，不施加层级/时间衰减

边权重设计:
    transmitsTo            P→P    forward  w=1.0        157
    rev_transmitsTo        P→P    reverse  w=w_rev      157
    targetsSubIndustry     P→SI   forward  w=confidence 1,719
    rev_targetsSubIndustry SI→P   reverse  w=w_rev×conf 1,719
    belongsTo              E→SI   forward  w=1.0        5,495
    rev_belongsTo          SI→E   reverse  w=w_rev  ★   5,495
    subClassOf             SI→MI  forward  w=1.0           62
    rev_subClassOf         MI→SI  reverse  w=w_rev         62
                                                          ─────
                                             总非零元 ≈ 14,866

参数:
    α (PPR 阻尼):         0.85
    w_rev (反向边权重):   0.3
    γ_h (层级衰减):       0.8
    λ_t (时间衰减率):     0.15

术语净化: 严禁 GraphRAG / Spatio-Temporal, 统一使用 HT-PPR。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import sparse


# ====================================================================
# 全局 ID 映射
# ====================================================================

class GlobalIDMapper:
    """将异构节点映射到全局连续 ID 空间 [0, N-1]，支持双向查询。"""

    def __init__(self, node_counts: Dict[str, int]):
        self.node_types = list(node_counts.keys())
        self.node_counts = node_counts
        self.offsets: Dict[str, int] = {}
        offset = 0
        for nt in self.node_types:
            self.offsets[nt] = offset
            offset += node_counts[nt]
        self.N = offset

    def to_global(self, node_type: str, local_idx: int) -> int:
        return self.offsets[node_type] + local_idx

    def to_local(self, global_idx: int) -> Tuple[str, int]:
        for nt in self.node_types:
            off = self.offsets[nt]
            cnt = self.node_counts[nt]
            if off <= global_idx < off + cnt:
                return nt, global_idx - off
        raise IndexError(f"global_idx {global_idx} out of range [0, {self.N})")

    def slice_for(self, node_type: str) -> slice:
        off = self.offsets[node_type]
        return slice(off, off + self.node_counts[node_type])

    def __repr__(self):
        parts = [f"GlobalIDMapper(N={self.N})"]
        for nt in self.node_types:
            off = self.offsets[nt]
            cnt = self.node_counts[nt]
            parts.append(f"  {nt:20s}: [{off:6d}, {off+cnt-1:6d}]  ({cnt:,})")
        return "\n".join(parts)


# ====================================================================
# HT-PPR Engine
# ====================================================================

class HT_PPR_Engine:
    """异构层级-时间个性化 PageRank 传播引擎。

    双重衰减仅施加于 Policy 节点:
        Hierarchical (离散):  score *= γ_h ^ |level - ref_level|
        Temporal (指数):      score *= exp(-λ_t * |year - ref_year|)

    Enterprise 节点仅做独立 Min-Max 归一化，不参与层级/时间衰减。
    """

    def __init__(
        self,
        alpha: float = 0.85,
        w_rev: float = 0.3,
        hierarchical_decay: float = 0.8,
        temporal_lambda: float = 0.15,
        max_iter: int = 500,
        tol: float = 1e-6,
    ):
        assert 0 < alpha < 1, f"alpha must be in (0,1), got {alpha}"
        assert w_rev >= 0, f"w_rev must be >= 0, got {w_rev}"
        assert 0 < hierarchical_decay < 1
        assert temporal_lambda > 0
        assert max_iter > 0 and tol > 0

        self.alpha = alpha
        self.w_rev = w_rev
        self.gamma_h = hierarchical_decay
        self.lambda_t = temporal_lambda
        self.max_iter = max_iter
        self.tol = tol

        # 构建后缓存
        self._mapper: Optional[GlobalIDMapper] = None
        self._M: Optional[sparse.csc_matrix] = None  # 列归一化转移矩阵

    # ------------------------------------------------------------------
    # Phase 0: 构建全局转移矩阵
    # ------------------------------------------------------------------

    def build_transition_matrix(
        self,
        message_graph_path: str,
    ) -> GlobalIDMapper:
        """从 message_graph.pt 构建异构全局转移矩阵。

        步骤:
            1. 建立全局 ID 映射
            2. 对每条正向边添加 COO 条目, 反向边附 w_rev 权重
            3. targetsSubIndustry 边使用 LLM confidence 作为权重
            4. 列 L1 归一化, 悬空列填充 1/N

        Returns:
            GlobalIDMapper (同时缓存 self._mapper 和 self._M)。
        """
        g = torch.load(message_graph_path, weights_only=False)

        # 1) 全局映射
        node_counts = {nt: g[nt].num_nodes for nt in g.node_types}
        mapper = GlobalIDMapper(node_counts)

        # 2) 收集 COO 三元组
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for et in g.edge_types:
            src_type, rel, dst_type = et
            ei = g[et].edge_index.numpy()  # [2, E]
            num_edges = ei.shape[1]

            # 正向权重
            if rel == "targetsSubIndustry":
                fwd_weights = g[et].confidence.numpy().astype(np.float64)
            else:
                fwd_weights = np.ones(num_edges, dtype=np.float64)

            for j in range(num_edges):
                src_local = int(ei[0, j])
                dst_local = int(ei[1, j])
                src_g = mapper.to_global(src_type, src_local)
                dst_g = mapper.to_global(dst_type, dst_local)

                # 正向: src → dst, 列随机矩阵 M[dst, src] = weight
                rows.append(dst_g)
                cols.append(src_g)
                data.append(float(fwd_weights[j]))

                # 反向: dst → src, M[src, dst] = w_rev × weight
                rows.append(src_g)
                cols.append(dst_g)
                data.append(self.w_rev * float(fwd_weights[j]))

        # 构建 COO 矩阵
        W_coo = sparse.coo_matrix(
            (data, (rows, cols)), shape=(mapper.N, mapper.N), dtype=np.float64
        )

        # 3) 列 L1 归一化 (高效: 直接在 COO data 上操作)
        # 先按列聚合求和
        col_sums = np.bincount(
            np.array(cols), weights=np.array(data), minlength=mapper.N
        ).astype(np.float64)

        dangling_mask = col_sums < 1e-12
        dangling_count = int(dangling_mask.sum())

        # 对非悬空列: data[i] /= col_sums[cols[i]]
        # 构建除数映射
        col_norm = np.where(dangling_mask, 1.0, col_sums)  # avoid div-by-zero
        data_normalized = np.array(data, dtype=np.float64) / col_norm[np.array(cols)]

        # 悬空列数据清零 (靠幂迭代 leak 补偿)
        for i in range(len(data_normalized)):
            if dangling_mask[cols[i]]:
                data_normalized[i] = 0.0

        W_norm = sparse.coo_matrix(
            (data_normalized, (rows, cols)), shape=(mapper.N, mapper.N), dtype=np.float64
        ).tocsc()
        W_norm.eliminate_zeros()

        # 缓存
        self._mapper = mapper
        self._M = W_norm
        self._dangling_mask = dangling_mask  # [N] bool, 用于幂迭代

        print(f"  全局节点数:   {mapper.N:,}")
        print(f"  原始非零元:   {len(data):,}")
        print(f"  归一化后nnz:  {W_norm.nnz:,}")
        print(f"  稀疏度:       {W_norm.nnz / (mapper.N * mapper.N):.6f}")
        print(f"  悬空列数:     {dangling_count}")

        return mapper

    # ------------------------------------------------------------------
    # Phase 1: 幂迭代 PPR
    # ------------------------------------------------------------------

    def compute_pagerank(
        self,
        personalization: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """在缓存的全局转移矩阵上运行幂迭代 PPR。

        Args:
            personalization: [N] 个性化向量 (默认 None → 均匀 1/N)。
            verbose: 是否打印收敛日志。

        Returns:
            [N] 稳态 PPR 概率向量。
        """
        if self._M is None:
            raise RuntimeError("请先调用 build_transition_matrix()")

        N = self._M.shape[0]

        if personalization is None:
            v = np.full(N, 1.0 / N, dtype=np.float64)
        else:
            v = np.asarray(personalization, dtype=np.float64)
            assert v.shape == (N,), f"personalization shape must be ({N},), got {v.shape}"
            v_sum = v.sum()
            if v_sum > 0:
                v = v / v_sum
            else:
                v = np.full(N, 1.0 / N, dtype=np.float64)

        p = v.copy()
        d_mask = getattr(self, "_dangling_mask", np.zeros(N, dtype=bool))

        for iteration in range(self.max_iter):
            Mp = self._M @ p
            # 悬空节点泄漏质量补偿: leaked = Σ p[j] for dangling columns j
            leaked = float(p[d_mask].sum())
            p_new = self.alpha * (Mp + leaked * v) + (1.0 - self.alpha) * v
            delta = np.abs(p_new - p).max()

            if verbose and (iteration < 3 or iteration % 50 == 0 or delta < self.tol):
                print(f"    iter {iteration:4d}: δ={delta:.8f}")

            p = p_new

            if delta < self.tol:
                if verbose:
                    print(f"    [收敛] iter {iteration+1}: δ={delta:.8f} < tol={self.tol}")
                break
        else:
            if verbose:
                print(f"    [警告] 达到 max_iter={self.max_iter}, δ={delta:.8f}")

        return p

    # ------------------------------------------------------------------
    # Phase 3: 双重衰减 (仅 Policy)
    # ------------------------------------------------------------------

    def _apply_hierarchical_decay(
        self,
        scores: np.ndarray,
        levels: np.ndarray,
        ref_level: int = 1,
    ) -> np.ndarray:
        """层级离散衰减: score *= γ_h ^ |level - ref_level|."""
        dist = np.abs(levels.astype(np.float64) - ref_level)
        decay = np.power(self.gamma_h, dist)
        result = scores * decay
        print(f"    层级衰减: γ_h={self.gamma_h}, ref_level={ref_level}")
        print(f"      平均衰减因子: {decay.mean():.4f}, 最大层级差: {int(dist.max())}")
        return result

    def _apply_temporal_decay(
        self,
        scores: np.ndarray,
        years: np.ndarray,
        ref_year: int = 2024,
    ) -> np.ndarray:
        """时间指数衰减: score *= exp(-λ_t * |year - ref_year|)."""
        dist = np.abs(years.astype(np.float64) - ref_year)
        decay = np.exp(-self.lambda_t * dist)
        result = scores * decay

        # 对零分节点不再衰减 (避免全零)
        mask = scores < 1e-14
        result[mask] = 0.0

        print(f"    时间衰减: λ_t={self.lambda_t}, ref_year={ref_year}")
        print(f"      平均衰减因子: {decay[~mask].mean():.4f}, 最大年份差: {int(dist.max())}")
        return result

    @staticmethod
    def _log_z_sigmoid(
        scores: np.ndarray,
        eps: float = 1e-10,
        label: str = "",
    ) -> np.ndarray:
        """Log-Z-Sigmoid 归一化: 压制极值 + 标准化 + 非线性挤压到 (0,1).

        步骤:
            1. log_scores = log(scores + ε)         — 对数平滑, 压制长尾极值
            2. z = (log_scores - mean) / std         — Z-score 标准化
            3. final = 1 / (1 + exp(-z))             — Sigmoid 映射到 (0, 1)

        期望: 均值回归 ~0.5, 分布近似对称。
        """
        if np.all(scores < eps):
            return np.full_like(scores, 0.5)

        log_scores = np.log(np.maximum(scores, eps))
        mean = np.mean(log_scores)
        std = np.std(log_scores)

        if std < 1e-14:
            z_scores = np.zeros_like(log_scores)
        else:
            z_scores = (log_scores - mean) / std

        final = 1.0 / (1.0 + np.exp(-z_scores))

        if label:
            print(f"    [{label}] Log-Z-Sigmoid:")
            print(f"      log(score):  mean={mean:.4f}, std={std:.4f}")
            print(f"      final:       mean={final.mean():.4f}, std={final.std():.4f}, "
                  f"min={final.min():.6f}, max={final.max():.6f}")
        return final

    # ------------------------------------------------------------------
    # 度数补偿 (消除行业规模稀释效应)
    # ------------------------------------------------------------------

    def _compute_enterprise_compensation(
        self,
        enterprise_raw: np.ndarray,
        message_graph_path: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按所属 SubIndustry 企业数进行度数补偿。

        compensated_PPR = raw_PPR × N_subindustry

        物理意义: 将企业 PPR 修正为该产业赛道的"整体宏观政策热度"。
        大行业不再因企业多而稀释，小行业不再因企业少而膨胀。

        Returns:
            (compensated_scores, si_ent_counts_array)
        """
        g = torch.load(message_graph_path, weights_only=False)

        et = ("Enterprise", "belongsTo", "SubIndustry")
        ei = g[et].edge_index.numpy()  # [2, E], src=Enterprise, dst=SubIndustry

        # enterprise local idx → subindustry local idx
        ent_to_si: Dict[int, int] = {}
        for j in range(ei.shape[1]):
            ent_to_si[int(ei[0, j])] = int(ei[1, j])

        # 统计每个 SubIndustry 的企业数
        si_counts: Dict[int, int] = {}
        for si in ent_to_si.values():
            si_counts[si] = si_counts.get(si, 0) + 1

        compensated = enterprise_raw.copy()

        si_ent_counts = np.ones(len(enterprise_raw), dtype=np.float64)
        for e_local, si_local in ent_to_si.items():
            n_si = si_counts.get(si_local, 1)
            compensated[e_local] *= n_si
            si_ent_counts[e_local] = n_si

        # 诊断: 每行业企业数分布
        counts = list(si_counts.values())
        print(f"    度数补偿: {len(si_counts)} 个 SubIndustry, "
              f"企业数范围 [{min(counts)}, {max(counts)}], "
              f"均值={np.mean(counts):.1f}, 中位数={np.median(counts):.0f}")
        print(f"    补偿因子范围: [{min(counts):.0f}, {max(counts):.0f}]")
        print(f"    compensated PPR: min={compensated.min():.8f}, "
              f"max={compensated.max():.8f}, mean={compensated.mean():.8f}")

        return compensated, si_ent_counts

    # ------------------------------------------------------------------
    # Phase 2-4: 分数提取 + 衰减 + 归一化
    # ------------------------------------------------------------------

    def extract_and_decay(
        self,
        ppr_vector: np.ndarray,
        policy_levels: np.ndarray,
        policy_years: np.ndarray,
        message_graph_path: str = "",
        reference_level: int = 1,
        reference_year: int = 2024,
    ) -> Dict[str, np.ndarray]:
        """从全局 PPR 向量提取 Policy/Enterprise 分数并施加后处理。

        Policy:   层级衰减 → 时间衰减 → Log-Z-Sigmoid
        Enterprise: 度数补偿 → Log-Z-Sigmoid (无层级/时间衰减)

        Returns:
            {
                "policy_raw":         [N_policy]  原始 PPR,
                "policy_decayed":     [N_policy]  衰减后 (未归一化),
                "policy_final":       [N_policy]  最终 (0,1),
                "enterprise_raw":     [N_ent]     原始 PPR,
                "enterprise_compensated": [N_ent] 度数补偿后,
                "enterprise_final":   [N_ent]     最终 (0,1),
                "sub_industry_raw":   [N_si]      原始 PPR,
                "major_industry_raw": [N_mi]      原始 PPR,
            }
        """
        mapper = self._mapper
        if mapper is None:
            raise RuntimeError("请先调用 build_transition_matrix()")

        # 切片提取
        p_slice = mapper.slice_for("Policy")
        e_slice = mapper.slice_for("Enterprise")
        si_slice = mapper.slice_for("SubIndustry")
        mi_slice = mapper.slice_for("MajorIndustry")

        policy_raw = ppr_vector[p_slice].copy()
        enterprise_raw = ppr_vector[e_slice].copy()
        si_raw = ppr_vector[si_slice].copy()
        mi_raw = ppr_vector[mi_slice].copy()

        # --- Policy: 双重衰减 → Log-Z-Sigmoid ---
        print(f"\n  [Policy 后处理] N={len(policy_raw)}")
        print(f"    原始 PPR: min={policy_raw.min():.8f}, max={policy_raw.max():.8f}, "
              f"mean={policy_raw.mean():.8f}, std={policy_raw.std():.8f}")

        policy_decayed = policy_raw.copy()
        policy_decayed = self._apply_hierarchical_decay(
            policy_decayed, policy_levels, reference_level
        )
        policy_decayed = self._apply_temporal_decay(
            policy_decayed, policy_years, reference_year
        )

        print(f"    衰减后:   min={policy_decayed.min():.8f}, max={policy_decayed.max():.8f}, "
              f"mean={policy_decayed.mean():.8f}, std={policy_decayed.std():.8f}")

        policy_final = self._log_z_sigmoid(policy_decayed, label="Policy")

        # --- Enterprise: 度数补偿 → Log-Z-Sigmoid ---
        print(f"\n  [Enterprise 后处理] N={len(enterprise_raw)}")
        print(f"    原始 PPR: min={enterprise_raw.min():.8f}, max={enterprise_raw.max():.8f}, "
              f"mean={enterprise_raw.mean():.8f}, std={enterprise_raw.std():.8f}")

        ent_compensated, _ = self._compute_enterprise_compensation(
            enterprise_raw, message_graph_path
        )

        ent_final = self._log_z_sigmoid(ent_compensated, label="Enterprise")

        return {
            "policy_raw": policy_raw,
            "policy_decayed": policy_decayed,
            "policy_final": policy_final,
            "enterprise_raw": enterprise_raw,
            "enterprise_compensated": ent_compensated,
            "enterprise_final": ent_final,
            "sub_industry_raw": si_raw,
            "major_industry_raw": mi_raw,
        }

    # ------------------------------------------------------------------
    # 全管线 (便捷方法)
    # ------------------------------------------------------------------

    def run(
        self,
        message_graph_path: str,
        policy_levels: np.ndarray,
        policy_years: np.ndarray,
        personalization: Optional[np.ndarray] = None,
        reference_level: int = 1,
        reference_year: int = 2024,
    ) -> Dict[str, np.ndarray]:
        """一键运行 HT-PPR 全管线。

        Returns:
            同 extract_and_decay() 的字典结构。
        """
        print("=" * 60)
        print("HT-PPR Engine — Heterogeneous PPR with Dual Decay")
        print(f"  α={self.alpha}, w_rev={self.w_rev}, "
              f"γ_h={self.gamma_h}, λ_t={self.lambda_t}")
        print("=" * 60)

        # Phase 0: 构建转移矩阵
        print("\n[Phase 0] 构建全局异构转移矩阵...")
        mapper = self.build_transition_matrix(message_graph_path)
        print(mapper)

        # Phase 1: PPR 幂迭代
        print(f"\n[Phase 1] 幂迭代 PPR (α={self.alpha}, "
              f"v={'uniform' if personalization is None else 'custom'})...")
        ppr = self.compute_pagerank(personalization=personalization)

        # Phase 2-4: 提取 + 衰减 + 归一化
        print(f"\n[Phase 2-4] 分数提取 + 度数补偿 + Log-Z-Sigmoid...")
        results = self.extract_and_decay(
            ppr, policy_levels, policy_years,
            message_graph_path=message_graph_path,
            reference_level=reference_level,
            reference_year=reference_year,
        )

        # 诊断汇总
        print(f"\n{'='*60}")
        print("最终分数分布 (Log-Z-Sigmoid):")
        for key in ["policy_final", "enterprise_final"]:
            arr = results[key]
            print(f"  {key:25s}  mean={arr.mean():.4f}, std={arr.std():.4f}, "
                  f"min={arr.min():.6f}, max={arr.max():.6f}, "
                  f"median={np.median(arr):.4f}")
        print("=" * 60)

        return results


# ====================================================================
# 数据加载工具
# ====================================================================

def load_policy_attributes(
    message_graph_path: str,
    policies_final_path: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """从 HeteroData + policies_final.json 加载政策层级和年份。

    Returns:
        levels: [N] int32 层级 (1/2/3)
        years:  [N] int32 发布年份
        policy_ids: [N] str  P_XXXX 列表
    """
    g = torch.load(message_graph_path, weights_only=False)
    levels = g["Policy"].level.numpy().astype(np.int32)

    with open(policies_final_path, "r", encoding="utf-8") as f:
        policies = json.load(f)

    active = {p["policy_id"]: p for p in policies if p.get("status") == "active"}
    pid_list = list(g["Policy"].policy_ids)

    years = np.zeros(len(pid_list), dtype=np.int32)
    for i, pid in enumerate(pid_list):
        p = active.get(pid)
        if p is not None:
            pub_date = str(p.get("pub_date", ""))
            if len(pub_date) >= 4:
                try:
                    years[i] = int(pub_date[:4])
                except ValueError:
                    years[i] = 2018
            else:
                years[i] = 2018
        else:
            years[i] = 2018

    return levels, years, pid_list


# ====================================================================
# CLI
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session 4 (Revised) — Heterogeneous HT-PPR Engine"
    )
    parser.add_argument(
        "--message-graph",
        type=str,
        default="data/processed/splits/message_graph.pt",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="data/processed/policies_final.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/ht_ppr",
    )
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--w-rev", type=float, default=0.3)
    parser.add_argument("--hierarchical-decay", type=float, default=0.8)
    parser.add_argument("--temporal-lambda", type=float, default=0.15)
    parser.add_argument("--reference-level", type=int, default=1)
    parser.add_argument("--reference-year", type=int, default=2024)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    message_graph_path = str(project_root / args.message_graph)
    policies_path = str(project_root / args.policies)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载政策属性
    print("[Pre] 加载政策层级与年份...")
    levels, years, policy_ids = load_policy_attributes(
        message_graph_path, policies_path
    )
    print(f"  Policy: {len(levels)}, levels={dict(zip(*np.unique(levels, return_counts=True)))}")
    print(f"  Year range: {years.min()}-{years.max()}")

    # 2. 运行 HT-PPR
    engine = HT_PPR_Engine(
        alpha=args.alpha,
        w_rev=args.w_rev,
        hierarchical_decay=args.hierarchical_decay,
        temporal_lambda=args.temporal_lambda,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    results = engine.run(
        message_graph_path=message_graph_path,
        policy_levels=levels,
        policy_years=years,
        personalization=None,  # 全局均匀 v
        reference_level=args.reference_level,
        reference_year=args.reference_year,
    )

    # 3. 持久化
    print(f"\n[Save] 输出 → {output_dir}")
    for key, arr in results.items():
        fpath = output_dir / f"{key}.npy"
        np.save(fpath, arr)
        print(f"  [OK] {fpath.name:30s}  shape={arr.shape}")

    # 元信息
    meta = {
        "description": "Heterogeneous HT-PPR Engine — PPR + 度数补偿 + Log-Z-Sigmoid",
        "generated": datetime.now().isoformat(),
        "normalization": "Log-Z-Sigmoid (log → Z-score → sigmoid)",
        "enterprise_processing": "degree_compensation (raw_PPR × N_subindustry)",
        "parameters": {
            "alpha": args.alpha,
            "w_rev": args.w_rev,
            "hierarchical_decay_gamma": args.hierarchical_decay,
            "temporal_lambda": args.temporal_lambda,
            "reference_level": args.reference_level,
            "reference_year": args.reference_year,
            "max_iter": args.max_iter,
            "tol": args.tol,
        },
        "policy_scores": {
            "raw_min": float(results["policy_raw"].min()),
            "raw_max": float(results["policy_raw"].max()),
            "decayed_min": float(results["policy_decayed"].min()),
            "decayed_max": float(results["policy_decayed"].max()),
            "final_mean": float(results["policy_final"].mean()),
            "final_std": float(results["policy_final"].std()),
            "final_min": float(results["policy_final"].min()),
            "final_max": float(results["policy_final"].max()),
        },
        "enterprise_scores": {
            "raw_min": float(results["enterprise_raw"].min()),
            "raw_max": float(results["enterprise_raw"].max()),
            "compensated_min": float(results["enterprise_compensated"].min()),
            "compensated_max": float(results["enterprise_compensated"].max()),
            "final_mean": float(results["enterprise_final"].mean()),
            "final_std": float(results["enterprise_final"].std()),
            "final_min": float(results["enterprise_final"].min()),
            "final_max": float(results["enterprise_final"].max()),
        },
        "top_10_policy_ids": [
            policy_ids[i] for i in np.argsort(results["policy_final"])[-10:][::-1]
        ],
    }
    with open(output_dir / "ht_ppr_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] ht_ppr_meta.json")

    print("\n[DONE] HT-PPR 异构传播完成")


if __name__ == "__main__":
    main()
