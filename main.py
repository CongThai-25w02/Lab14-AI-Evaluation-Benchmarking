import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.release_gate import ReleaseGate
from agent.main_agent import MainAgent

# Giả lập component RAGAS - chưa tích hợp thực tế.
class ExpertEvaluator:
    async def score(self, case, resp):
        # Hit Rate/MRR không còn giả lập ở đây: BenchmarkRunner tự tính bằng
        # RetrievalEvaluator dựa trên expected_retrieval_ids/retrieved_ids thực tế.
        return {
            "faithfulness": 0.9,
            "relevancy": 0.8,
        }

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), LLMJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["retrieval"]["hit_rate"] for r in results) / total,
            "mrr": sum(r["retrieval"]["mrr"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
            "avg_latency": sum(r["latency"] for r in results) / total,
            "avg_tokens": sum(r["tokens_used"] for r in results) / total,
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    print(f"V1: avg_score={v1_summary['metrics']['avg_score']:.2f}, hit_rate={v1_summary['metrics']['hit_rate']:.2f}, "
          f"avg_latency={v1_summary['metrics']['avg_latency']:.3f}s, avg_tokens={v1_summary['metrics']['avg_tokens']:.1f}")
    print(f"V2: avg_score={v2_summary['metrics']['avg_score']:.2f}, hit_rate={v2_summary['metrics']['hit_rate']:.2f}, "
          f"avg_latency={v2_summary['metrics']['avg_latency']:.3f}s, avg_tokens={v2_summary['metrics']['avg_tokens']:.1f}")

    gate = ReleaseGate()
    gate_result = gate.evaluate(v1_summary, v2_summary)
    v2_summary["release_gate"] = gate_result

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if gate_result["decision"] == "RELEASE":
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (RELEASE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (ROLLBACK)")
        for v in gate_result["violations"]:
            print(f"   - {v}")

if __name__ == "__main__":
    asyncio.run(main())
