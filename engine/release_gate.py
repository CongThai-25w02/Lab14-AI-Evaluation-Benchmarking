from typing import Dict, List


class ReleaseGate:
    """
    Auto-Gate: quyết định Release/Rollback dựa trên Delta Analysis giữa 2 phiên bản Agent,
    xét cả 3 trục: Chất lượng (avg_score, hit_rate), Chi phí (avg_tokens) và Hiệu năng (avg_latency).
    Mặc định: chỉ ROLLBACK khi có ít nhất 1 chỉ số vượt ngưỡng cho phép, dù các chỉ số khác cải thiện.
    """

    def __init__(
        self,
        max_quality_drop: float = 0.1,
        max_hit_rate_drop: float = 0.05,
        max_latency_increase_pct: float = 0.10,
        max_cost_increase_pct: float = 0.10,
    ):
        self.max_quality_drop = max_quality_drop
        self.max_hit_rate_drop = max_hit_rate_drop
        self.max_latency_increase_pct = max_latency_increase_pct
        self.max_cost_increase_pct = max_cost_increase_pct

    @staticmethod
    def _pct_change(old: float, new: float) -> float:
        if old == 0:
            return 0.0 if new == 0 else float("inf")
        return (new - old) / old

    def evaluate(self, v1_summary: Dict, v2_summary: Dict) -> Dict:
        v1, v2 = v1_summary["metrics"], v2_summary["metrics"]

        deltas = {
            "avg_score": v2["avg_score"] - v1["avg_score"],
            "hit_rate": v2["hit_rate"] - v1["hit_rate"],
            "avg_latency_pct": self._pct_change(v1["avg_latency"], v2["avg_latency"]),
            "avg_tokens_pct": self._pct_change(v1["avg_tokens"], v2["avg_tokens"]),
        }

        violations: List[str] = []
        if deltas["avg_score"] < -self.max_quality_drop:
            violations.append(
                f"Quality (avg_score) giảm {abs(deltas['avg_score']):.2f} điểm, "
                f"vượt ngưỡng cho phép {self.max_quality_drop:.2f}."
            )
        if deltas["hit_rate"] < -self.max_hit_rate_drop:
            violations.append(
                f"Retrieval (hit_rate) giảm {abs(deltas['hit_rate']) * 100:.1f}%, "
                f"vượt ngưỡng cho phép {self.max_hit_rate_drop * 100:.0f}%."
            )
        if deltas["avg_latency_pct"] > self.max_latency_increase_pct:
            violations.append(
                f"Performance (avg_latency) tăng {deltas['avg_latency_pct'] * 100:.1f}%, "
                f"vượt ngưỡng cho phép {self.max_latency_increase_pct * 100:.0f}%."
            )
        if deltas["avg_tokens_pct"] > self.max_cost_increase_pct:
            violations.append(
                f"Cost (avg_tokens) tăng {deltas['avg_tokens_pct'] * 100:.1f}%, "
                f"vượt ngưỡng cho phép {self.max_cost_increase_pct * 100:.0f}%."
            )

        decision = "ROLLBACK" if violations else "RELEASE"
        return {
            "decision": decision,
            "deltas": deltas,
            "violations": violations,
        }
