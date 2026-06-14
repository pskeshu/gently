"""
Agent Tool Calling Evaluator

Measures tool selection accuracy and parameter correctness.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single test case"""

    test_id: str
    query: str
    expected_tool: str | list[str]
    actual_tool: str | None
    tool_correct: bool
    params_correct: bool
    param_errors: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.tool_correct and self.params_correct and self.error is None


@dataclass
class BenchmarkReport:
    """Summary report for a benchmark run"""

    timestamp: str
    num_cases: int
    num_passed: int
    tool_accuracy: float
    param_accuracy: float
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_ms: float
    results: list[EvalResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "num_cases": self.num_cases,
                "num_passed": self.num_passed,
                "pass_rate": self.num_passed / self.num_cases if self.num_cases > 0 else 0,
                "tool_accuracy": self.tool_accuracy,
                "param_accuracy": self.param_accuracy,
            },
            "tokens": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "avg_per_query": (self.total_input_tokens + self.total_output_tokens)
                / self.num_cases
                if self.num_cases > 0
                else 0,
            },
            "latency": {
                "avg_ms": self.avg_latency_ms,
            },
            "metadata": self.metadata,
            "results": [
                {
                    "test_id": r.test_id,
                    "passed": r.passed,
                    "tool_correct": r.tool_correct,
                    "params_correct": r.params_correct,
                    "expected": r.expected_tool,
                    "actual": r.actual_tool,
                    "errors": r.param_errors,
                }
                for r in self.results
            ],
        }


class AgentEvaluator:
    """
    Evaluates agent tool calling accuracy

    Usage:
        evaluator = AgentEvaluator()
        report = await evaluator.run_benchmark(agent, test_cases)
        print(f"Tool accuracy: {report.tool_accuracy:.1%}")
    """

    def __init__(self, test_cases_path: Path | None = None):
        """
        Parameters
        ----------
        test_cases_path : Path, optional
            Path to test_cases.json. Defaults to bundled test cases.
        """
        if test_cases_path is None:
            test_cases_path = Path(__file__).parent / "test_cases.json"

        with open(test_cases_path) as f:
            data = json.load(f)

        self.test_cases = data["test_cases"]
        self.version = data.get("version", "unknown")

    async def run_benchmark(
        self,
        agent,
        tags: list[str] | None = None,
        max_cases: int | None = None,
    ) -> BenchmarkReport:
        """
        Run benchmark against agent

        Parameters
        ----------
        agent : MicroscopyAgent
            Agent instance to evaluate
        tags : list, optional
            Only run test cases with these tags
        max_cases : int, optional
            Limit number of test cases

        Returns
        -------
        BenchmarkReport
            Evaluation results
        """
        # Filter test cases
        cases = self.test_cases
        if tags:
            cases = [c for c in cases if any(t in c.get("tags", []) for t in tags)]
        if max_cases:
            cases = cases[:max_cases]

        results = []
        for case in cases:
            result = await self._evaluate_case(agent, case)
            results.append(result)
            logger.info(f"[{'PASS' if result.passed else 'FAIL'}] {case['id']}")

        # Compute summary
        num_passed = sum(1 for r in results if r.passed)
        tool_correct = sum(1 for r in results if r.tool_correct)
        param_correct = sum(1 for r in results if r.params_correct)

        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            num_cases=len(results),
            num_passed=num_passed,
            tool_accuracy=tool_correct / len(results) if results else 0,
            param_accuracy=param_correct / len(results) if results else 0,
            total_input_tokens=sum(r.input_tokens for r in results),
            total_output_tokens=sum(r.output_tokens for r in results),
            avg_latency_ms=sum(r.latency_ms for r in results) / len(results) if results else 0,
            results=results,
            metadata={"version": self.version, "tags": tags},
        )

    async def _evaluate_case(self, agent, case: dict) -> EvalResult:
        """Evaluate a single test case"""
        test_id = case["id"]
        query = case["query"]
        expected_tool = case["expected_tool"]
        expected_params = case.get("expected_params", {})

        try:
            # Get tool call from agent (without executing)
            import time

            start = time.perf_counter()

            tool_call = await self._get_tool_call(agent, query)

            latency_ms = (time.perf_counter() - start) * 1000

            if tool_call is None:
                return EvalResult(
                    test_id=test_id,
                    query=query,
                    expected_tool=expected_tool,
                    actual_tool=None,
                    tool_correct=False,
                    params_correct=False,
                    param_errors=["No tool called"],
                    latency_ms=latency_ms,
                )

            actual_tool = tool_call.get("name")
            actual_params = tool_call.get("input", {})

            # Check tool correctness
            if isinstance(expected_tool, list):
                tool_correct = actual_tool in expected_tool
            else:
                tool_correct = actual_tool == expected_tool

            # Check parameter correctness
            param_errors = []
            for key, expected_value in expected_params.items():
                if key == "intent_contains":
                    # Special case: check if intent contains keywords
                    intent = actual_params.get("intent", "").lower()
                    for keyword in expected_value:
                        if keyword.lower() not in intent:
                            param_errors.append(f"intent missing '{keyword}'")
                elif key not in actual_params:
                    param_errors.append(f"missing param: {key}")
                elif actual_params[key] != expected_value:
                    param_errors.append(
                        f"{key}: expected {expected_value}, got {actual_params[key]}"
                    )

            return EvalResult(
                test_id=test_id,
                query=query,
                expected_tool=expected_tool,
                actual_tool=actual_tool,
                tool_correct=tool_correct,
                params_correct=len(param_errors) == 0,
                param_errors=param_errors,
                latency_ms=latency_ms,
                input_tokens=tool_call.get("input_tokens", 0),
                output_tokens=tool_call.get("output_tokens", 0),
            )

        except Exception as e:
            logger.error(f"Error evaluating {test_id}: {e}")
            return EvalResult(
                test_id=test_id,
                query=query,
                expected_tool=expected_tool,
                actual_tool=None,
                tool_correct=False,
                params_correct=False,
                error=str(e),
            )

    async def _get_tool_call(self, agent, query: str) -> dict | None:
        """
        Get the tool call Claude would make for a query

        Uses agent.get_tool_call() which makes a real API call
        but doesn't execute the selected tool (dry-run mode).
        """
        return await agent.get_tool_call(query)


def compare_reports(before: BenchmarkReport, after: BenchmarkReport) -> dict:
    """
    Compare two benchmark reports

    Returns
    -------
    dict
        Comparison showing deltas for each metric
    """
    return {
        "tool_accuracy": {
            "before": before.tool_accuracy,
            "after": after.tool_accuracy,
            "delta": after.tool_accuracy - before.tool_accuracy,
        },
        "param_accuracy": {
            "before": before.param_accuracy,
            "after": after.param_accuracy,
            "delta": after.param_accuracy - before.param_accuracy,
        },
        "tokens": {
            "before": before.total_input_tokens + before.total_output_tokens,
            "after": after.total_input_tokens + after.total_output_tokens,
            "delta": (after.total_input_tokens + after.total_output_tokens)
            - (before.total_input_tokens + before.total_output_tokens),
        },
        "latency_ms": {
            "before": before.avg_latency_ms,
            "after": after.avg_latency_ms,
            "delta": after.avg_latency_ms - before.avg_latency_ms,
        },
        "regressions": [
            r.test_id
            for r in after.results
            if not r.passed and any(br.test_id == r.test_id and br.passed for br in before.results)
        ],
        "improvements": [
            r.test_id
            for r in after.results
            if r.passed and any(br.test_id == r.test_id and not br.passed for br in before.results)
        ],
    }
