"""
The full evaluation suite.

    python -m evals.run_all             # everything that runs without credentials
    python -m evals.run_all --deep      # + LLM judges (needs ANTHROPIC_API_KEY)
    python -m evals.run_all --json report.json

Four sections:

  1. Narration guard  - every number the LLM states must be a number a tool
                        returned. This is the project's core invariant.
  2. Ops scenarios    - controlled operational faults with KNOWN correct
                        diagnoses (evals/scenarios.py). Scored on six
                        independent dimensions rather than one average,
                        because "right answer, wrong reason" is the
                        interesting failure and a mean would hide it.
  3. RAG evaluation   - retrieval recall/precision/MRR and generation
                        decision accuracy over the labelled eval set.
  4. Ragas            - real Ragas metrics when installed (optional extra).

Exit codes: 0 pass, 1 a quality bar failed, 2 the core invariant was
violated. GitHub Actions treats any non-zero as a failed build; the split
exists so a human reading the log knows immediately whether the project's
central claim broke or a threshold merely drifted.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agents.graph import build_graph, run_question
from evals.narration_faithfulness import check_state

# Scenario questions the narration guard runs against. Chosen to span the
# verdict space - a growth case, a reduction case, a low-volume premium cut,
# a pure inventory question, and a question the simulation cannot answer -
# because the narration template and prompt behave differently for each.
NARRATION_CASES = [
    "what if we increase chuck roast production by 15%?",
    "should we scale back ribeye steak by 20 percent",
    "what happens if we make 25% more pulled pork roast",
    "cut ground beef medium back 30%",
    "how many boxes of blade eyes do we have on hand?",
    "what were our top sellers last month",
    "what products do we carry",
    "how do I trim a brisket fat cap",
]

# Regression bars. Set below currently measured values so they catch breakage
# rather than encoding today's numbers as a target.
MIN_RETRIEVAL_HIT_RATE_AT_5 = 0.90
MIN_RETRIEVAL_RECALL_AT_5 = 0.85
MIN_MRR = 0.75
MIN_DECISION_ACCURACY = 0.80

# Operational-scenario bars. `diagnosis` and `decision` are separated
# deliberately: a system that reaches the right verdict while naming the
# wrong constraint has got lucky, and one number covering both would let that
# pass. Set below currently measured values (all at 1.00) so they catch
# breakage rather than encoding today's numbers as a target.
MIN_SCENARIO_PASS_RATE = 0.85
MIN_SCENARIO_DIAGNOSIS_RATE = 0.80
MIN_SCENARIO_DECISION_RATE = 0.80

# Declining an out-of-corpus question is a SEMANTIC judgement. With
# credentials, Claude reads the passages and the bar is absolute: nothing
# out-of-corpus may be answered. Offline, the lexical heuristic provably
# cannot separate one measured case (see KNOWN_OFFLINE_DECLINE_GAPS), so the
# offline check is "no NEW leaks", by name, rather than a softened average -
# which would let a different regression hide inside the same number.
REQUIRED_DECLINE_RATE_LIVE = 1.0


def run_narration_guard(deep: bool = False) -> dict[str, Any]:
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(checkpointer=InMemorySaver())
    results = []
    for i, question in enumerate(NARRATION_CASES):
        state = run_question(question, thread_id=f"eval-{i}", graph=graph)
        result = check_state(state, deep=deep)
        results.append({
            "question": question,
            "verdict": (state.get("recommendation") or {}).get("verdict"),
            "narrated_by": (state.get("recommendation") or {}).get("narrated_by"),
            **result.to_dict(),
        })

    failures = [r for r in results if not r["passed"]]
    return {
        "cases": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "numbers_checked": sum(r["number_guard"]["numbers_checked"] for r in results),
        "violations": [
            {"question": r["question"], "issues": r["issues"]} for r in failures
        ],
        "per_case": results,
    }


def run_ops_scenarios() -> dict[str, Any]:
    """Controlled operational faults with known correct diagnoses."""
    from evals.scenarios import run_all as run_scenarios
    return run_scenarios()


def run_rag_eval(deep: bool = False) -> dict[str, Any]:
    from rag.eval.metrics import aggregate, aggregate_generation
    from rag.eval.run_eval import K_VALUES, evaluate_generation, evaluate_retrieval
    from rag.pipeline import RagPipeline

    pipeline = RagPipeline(use_cache=False)
    retrieval = evaluate_retrieval(pipeline)
    generation = evaluate_generation(pipeline, deep=deep)
    return {
        "retrieval": aggregate(retrieval, ks=K_VALUES),
        "generation": aggregate_generation(generation),
    }


def run_ragas() -> dict[str, Any]:
    from evals.ragas_adapter import evaluate
    return evaluate().to_dict()


def _bars(narration: dict, rag: dict,
          scenarios: dict | None = None) -> tuple[list[str], list[str]]:
    """Returns (critical failures, threshold failures)."""
    critical: list[str] = []
    soft: list[str] = []

    if scenarios:
        grounding = (scenarios.get("by_dimension") or {}).get("grounding")
        if grounding and grounding["passed"] < grounding["total"]:
            # Grounding is the core invariant wearing a different hat: an
            # ungrounded number or an unsupported claim under a controlled
            # fault is the same violation the narration guard exists to catch.
            critical.append(
                f"{grounding['total'] - grounding['passed']} operational "
                f"scenario(s) produced ungrounded numbers or unsupported claims")

        if scenarios["pass_rate"] < MIN_SCENARIO_PASS_RATE:
            soft.append(f"scenario pass rate {scenarios['pass_rate']} < "
                        f"{MIN_SCENARIO_PASS_RATE}")
        for dimension, floor in (("diagnosis", MIN_SCENARIO_DIAGNOSIS_RATE),
                                 ("decision", MIN_SCENARIO_DECISION_RATE)):
            measured = (scenarios.get("by_dimension") or {}).get(dimension)
            if measured and measured["rate"] < floor:
                soft.append(f"scenario {dimension} rate {measured['rate']} < {floor}")

    if narration["failed"]:
        critical.append(
            f"{narration['failed']} narration(s) stated a number or direction "
            f"not supported by tool output - this violates the project's core "
            f"'LLM never computes' rule")

    from agents import llm
    from rag.eval.dataset import KNOWN_OFFLINE_DECLINE_GAPS

    gen = rag.get("generation", {})
    leaked = set(gen.get("hallucinated") or [])
    if llm.is_live():
        if leaked:
            critical.append(
                f"correct_decline_rate {gen.get('correct_decline_rate')} < "
                f"{REQUIRED_DECLINE_RATE_LIVE}: the RAG layer answered a question "
                f"the corpus does not cover ({sorted(leaked)})")
    else:
        new_leaks = leaked - KNOWN_OFFLINE_DECLINE_GAPS
        if new_leaks:
            critical.append(
                f"NEW out-of-corpus questions are being answered offline: "
                f"{sorted(new_leaks)} (known offline gaps: "
                f"{sorted(KNOWN_OFFLINE_DECLINE_GAPS)})")

    ret = rag.get("retrieval", {})
    hit5 = (ret.get("hit_rate_at_k") or {}).get("5")
    rec5 = (ret.get("recall_at_k") or {}).get("5")
    mrr = ret.get("mrr")
    if hit5 is not None and hit5 < MIN_RETRIEVAL_HIT_RATE_AT_5:
        soft.append(f"hit_rate@5 {hit5} < {MIN_RETRIEVAL_HIT_RATE_AT_5}")
    if rec5 is not None and rec5 < MIN_RETRIEVAL_RECALL_AT_5:
        soft.append(f"recall@5 {rec5} < {MIN_RETRIEVAL_RECALL_AT_5}")
    if mrr is not None and mrr < MIN_MRR:
        soft.append(f"MRR {mrr} < {MIN_MRR}")
    accuracy = gen.get("decision_accuracy")
    if accuracy is not None and accuracy < MIN_DECISION_ACCURACY:
        soft.append(f"decision_accuracy {accuracy} < {MIN_DECISION_ACCURACY}")

    return critical, soft


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the full evaluation suite")
    parser.add_argument("--deep", action="store_true",
                        help="include LLM-judge metrics (needs credentials)")
    parser.add_argument("--ragas", action="store_true", help="also run Ragas")
    parser.add_argument("--json", default=None, help="write the full report here")
    args = parser.parse_args(argv)

    from agents import llm
    from rag import tuning

    print("=" * 74)
    print("OPS COPILOT - EVALUATION SUITE")
    print("=" * 74)
    llm_is_live = llm.is_live()
    print(f"LLM      : {'live' if llm_is_live else 'offline (deterministic paths)'}")
    print(f"RAG params: {tuning.current().source}")

    print("\n--- 1. NARRATION GUARD (LLM never computes numbers) ---")
    narration = run_narration_guard(deep=args.deep)
    print(f"  cases            : {narration['cases']}")
    print(f"  numbers checked  : {narration['numbers_checked']}")
    print(f"  passed           : {narration['passed']}/{narration['cases']}")
    for violation in narration["violations"]:
        print(f"  FAIL {violation['question']}")
        for issue in violation["issues"]:
            print(f"       - {issue}")

    print("\n--- 2. RAG EVALUATION ---")
    rag = run_rag_eval(deep=args.deep)
    ret, gen = rag["retrieval"], rag["generation"]
    print(f"  MRR                      : {ret.get('mrr')}")
    print(f"  recall@5 / hit@5         : {(ret.get('recall_at_k') or {}).get('5')} / "
          f"{(ret.get('hit_rate_at_k') or {}).get('5')}")
    if "rerank" in ret:
        print(f"  after rerank             : recall={ret['rerank']['recall']} "
              f"precision={ret['rerank']['precision']}")
    print(f"  decision accuracy        : {gen.get('decision_accuracy')}")
    print(f"  correct decline rate     : {gen.get('correct_decline_rate')}")
    print(f"  numeric grounded rate    : {gen.get('numeric_grounded_rate')}")
    if gen.get("faithfulness") is not None:
        print(f"  faithfulness (judge)     : {gen['faithfulness']}")

    scenarios = run_ops_scenarios()
    print("\n" + "=" * 78)
    print("OPERATIONAL SCENARIOS (controlled faults, known correct diagnoses)")
    print("=" * 78)
    for case in scenarios["cases"]:
        mark = "PASS" if case["passed"] else "FAIL"
        print(f"  [{mark}] {case['name']:<28} {case['fault']}")
        for failure in case["failures"]:
            print(f"         - {failure}")
    print(f"  {scenarios['passed']}/{scenarios['total']} cases "
          f"({scenarios['pass_rate']:.0%})")
    for name, counts in scenarios["by_dimension"].items():
        print(f"    {name:<12} {counts['passed']}/{counts['total']} "
              f"({counts['rate']:.0%})")

    ragas_report = None
    if args.ragas:
        print("\n--- 3. RAGAS ---")
        ragas_report = run_ragas()
        print(json.dumps(ragas_report, indent=2))

    critical, soft = _bars(narration, rag, scenarios)
    print("\n" + "=" * 74)
    if critical:
        print("CRITICAL FAILURES")
        for message in critical:
            print(f"  ! {message}")
    if soft:
        print("THRESHOLD FAILURES")
        for message in soft:
            print(f"  - {message}")
    if not critical and not soft:
        print("ALL CHECKS PASSED")
    print("=" * 74)

    if args.json:
        payload = {"narration_guard": narration, "scenarios": scenarios,
                   "rag": rag, "ragas": ragas_report,
                   "critical_failures": critical, "threshold_failures": soft}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.json}")

    if critical:
        return 2
    if soft:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
