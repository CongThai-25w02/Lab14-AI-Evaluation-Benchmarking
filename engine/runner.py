import asyncio
import time
from typing import List, Dict
from engine.retrieval_eval import RetrievalEvaluator

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.retrieval_evaluator = RetrievalEvaluator()

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # 1. Gọi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        # 2. Chạy RAGAS metrics
        ragas_scores = await self.evaluator.score(test_case, response)

        # 2b. Chạy Retrieval metrics (Hit Rate / MRR) dựa trên expected_retrieval_ids
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        retrieval_result = {
            "hit_rate": self.retrieval_evaluator.calculate_hit_rate(expected_ids, retrieved_ids),
            "mrr": self.retrieval_evaluator.calculate_mrr(expected_ids, retrieved_ids),
        }

        # 3. Chạy Multi-Judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"]
        )

        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "tokens_used": response.get("metadata", {}).get("tokens_used", 0),
            "ragas": ragas_scores,
            "retrieval": retrieval_result,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass"
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Chạy song song bằng asyncio.gather với giới hạn batch_size để không bị Rate Limit.
        """
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
