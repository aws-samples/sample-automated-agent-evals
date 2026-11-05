"""Helpers for the five lab notebooks: loading the task set, traces and labels, the chart style,
the code-based evaluators (Lab 1), the LLM judge (Lab 2), calibration against human labels (Lab 3),
capturing a version you define (Lab 4) and the Strands Decider decision model (Lab 5)."""
from __future__ import annotations

import copy
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path

import boto3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import cohen_kappa_score

from . import claimclerk



# --------------------------------------------------------------------------------------------------
# Paths, data and labels
# --------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

VERSIONS = ["v1", "v2-model", "v2-nova", "v2-prompt", "v2-tools"]


def load_cases() -> dict[str, dict]:
    """Return the task set as {case_id: case}."""
    cases = json.loads((DATA / "cases.json").read_text())["cases"]
    return {c["id"]: c for c in cases}


def load_traces(version: str) -> list[dict]:
    """Return every captured trial for one agent version."""
    path = DATA / "traces" / f"{version}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def tool_sequence(trace: dict) -> list[str]:
    return [call["tool"] for call in trace["tool_calls"]]


def show_trace(trace: dict, case: dict | None = None) -> None:
    """Render one transcript as Markdown: the prompt, each tool call, and the final answer."""
    lines = [f"**{trace['trial_id']}** ({trace['model_id']})", "", f"> {trace['prompt']}", ""]
    lines += ["| # | tool | arguments | result |", "|---|---|---|---|"]
    for i, call in enumerate(trace["tool_calls"], 1):
        args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
        result = json.dumps(call["result"])
        if len(result) > 120:
            result = result[:117] + "..."
        lines.append(f"| {i} | `{call['tool']}` | `{args}` | `{result}` |")
    parsed = trace["parsed"]
    lines += ["", f"**Final answer** decision `{parsed['decision']}`, payout `{parsed['payout']}`",
              "", f"> {parsed['explanation']}"]
    if case is not None:
        lines += ["", f"**Expected** decision `{case['expectedDecision']}`, payout `{case['expectedPayout']}`, "
                      f"route `{' -> '.join(case['expectedTrajectory'])}`"]
    display(Markdown("\n".join(lines)))


def traces_frame(versions: list[str] = VERSIONS) -> pd.DataFrame:
    """One row per trial across versions, with the fields most graders need."""
    rows = []
    for v in versions:
        for t in load_traces(v):
            rows.append({"version": v, "case_id": t["case_id"], "trial": t["trial"],
                         "model_id": t["model_id"], "decision": t["parsed"]["decision"],
                         "payout": t["parsed"]["payout"], "tools": " > ".join(tool_sequence(t)),
                         "input_tokens": t["usage"]["input_tokens"],
                         "output_tokens": t["usage"]["output_tokens"],
                         "latency_s": t["latency_s"]})
    return pd.DataFrame(rows)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def load_all_traces() -> dict[str, dict]:
    """All trials across versions, keyed by trial_id."""
    return {t["trial_id"]: t for v in VERSIONS for t in load_traces(v)}


LABELS = DATA / "human_labels" / "reference_labels.jsonl"


def load_labels() -> list[dict]:
    """The 60 human reference labels used to calibrate the judge in Lab 3."""
    return load_jsonl(LABELS)


# --------------------------------------------------------------------------------------------------
# Chart style
# --------------------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Categorical slots, fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]

# Sequential blue ramp, light to dark (steps 100 to 700).
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 12, "axes.labelsize": 10,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def _titles(fig, title: str, subtitle: str | None) -> None:
    """Left-aligned bold title with an optional grey subtitle, above everything else in the figure."""
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 0.985 - 0.26 / fig.get_figheight(), subtitle, ha="left", va="top",
                 fontsize=9, color=INK_SECONDARY)
    top = 1 - (0.62 if subtitle else 0.42) / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top))


def rate_heatmap(table, title: str, subtitle: str | None = None, fmt: str = "{:.2f}"):
    """A rows x columns table of rates in [0, 1], one blue ramp, every cell labelled with its value.
    NaN cells (a measure that does not apply) are left blank and labelled 'n/a'."""
    values = table.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    fig, ax = plt.subplots(figsize=(1.25 * n_cols + 2.2, 0.55 * n_rows + 1.6))
    masked = np.ma.masked_invalid(values)
    ax.imshow(masked, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    # White gaps between cells keep neighbouring values legible.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(range(n_cols), [str(c).replace("_", " ") for c in table.columns])
    ax.set_yticks(range(n_rows), [str(r).replace("_", " ") for r in table.index])
    ax.xaxis.tick_top()
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", color=INK_SECONDARY, fontsize=9)
            else:
                # Dark ink on light cells, white ink on dark cells.
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=10,
                        color="white" if v >= 0.55 else INK)
    _titles(fig, title, subtitle)
    return ax


def paired_bars(table, title: str, labels: tuple[str, str], ylabel: str, subtitle: str | None = None,
                fmt: str = "{:.2f}"):
    """Two series (for example baseline and candidate) across categories, directly labelled."""
    cats = list(table.index)
    x = np.arange(len(cats))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(cats) + 2), 3.6))
    for k, (col, color) in enumerate(zip(table.columns, SERIES)):
        offset = (k - 0.5) * (width + 0.02)
        bars = ax.bar(x + offset, table[col], width, color=color, label=labels[k], zorder=2)
        for b, v in zip(bars, table[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, fmt.format(v), ha="center", va="bottom",
                    fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x, [str(c).replace("_", " ") for c in cats])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _titles(fig, title, subtitle)
    return ax


# --------------------------------------------------------------------------------------------------
# Lab 1: code-based evaluators
# --------------------------------------------------------------------------------------------------
def grade_decision(trace, case):
    # The decision in the final answer matches the decision from the rules engine.
    return trace["parsed"]["decision"] == case["expectedDecision"]


def grade_payout(trace, case):
    # The payout in the final answer is within one cent of the rules-engine payout.
    payout = trace["parsed"]["payout"]
    return payout is not None and abs(float(payout) - case["expectedPayout"]) < 0.01


def grade_required_tools(trace, case):
    # Every tool the task requires was called at least once.
    called = set(tool_sequence(trace))
    return all(tool in called for tool in case["requiredTools"])


def grade_order(trace, case):
    # For each (a, b) pair the task lists, the first call to a comes before the first call to b.
    seq = tool_sequence(trace)
    for a, b in case["mustPrecede"]:
        if a not in seq or b not in seq or seq.index(a) > seq.index(b):
            return False
    return True


def grade_no_payout_on_denial(trace, case):
    # No tool the task forbids is called. On a claim denied for a missing
    # authorization or an unknown member, the forbidden tool is compute_payout.
    called = set(tool_sequence(trace))
    return not any(tool in called for tool in case["forbiddenTools"])


def grade_tool_args(trace, case):
    # Every argument passed to a tool matches the structured claim.
    claim = case["claim"]
    procedures = {s["procedure"] for s in claim["services"]}
    amounts = [s["billed_amount"] for s in claim["services"]]
    for call in trace["tool_calls"]:
        args = call["args"]
        if "member_id" in args and args["member_id"] != claim["member_id"]:
            return False
        if "network" in args and args["network"] != claim["network"]:
            return False
        if "procedure" in args and args["procedure"] not in procedures:
            return False
        if "billed_amount" in args and not any(abs(float(args["billed_amount"]) - a) < 0.01 for a in amounts):
            return False
    return True


GRADERS = {
    "decision": grade_decision,
    "payout": grade_payout,
    "required_tools": grade_required_tools,
    "order": grade_order,
    "no_payout_on_denial": grade_no_payout_on_denial,
    "tool_args": grade_tool_args,
}

WEIGHTS = {
    "decision": 0.25,
    "payout": 0.25,
    "required_tools": 0.15,
    "order": 0.15,
    "no_payout_on_denial": 0.10,
    "tool_args": 0.10,
}
HARD_GATES = ["decision", "payout", "no_payout_on_denial"]
PASS_THRESHOLD = 0.9


def composite(row):
    if not all(row[g] for g in HARD_GATES):
        return 0.0
    return sum(weight * row[name] for name, weight in WEIGHTS.items())


def evaluate_trace(trace, case) -> dict:
    """Run every evaluator on one trace and add the composite score and pass flag."""
    row = {name: grader(trace, case) for name, grader in GRADERS.items()}
    row["composite"] = composite(row)
    row["passed"] = row["composite"] >= PASS_THRESHOLD
    return row


def code_scores(versions: list[str] = VERSIONS) -> pd.DataFrame:
    """The Lab 1 evaluators, composite score and pass flag for every captured trial."""
    cases = load_cases()
    rows = []
    for v in versions:
        for t in load_traces(v):
            rows.append({"version": v, "case_id": t["case_id"], "trial": t["trial"],
                         "trial_id": t["trial_id"], **evaluate_trace(t, cases[t["case_id"]])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Lab 2: the LLM judge on Amazon Bedrock
# --------------------------------------------------------------------------------------------------

JUDGE_MODEL = "us.amazon.nova-2-lite-v1:0"
VERDICTS = ["PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"]

# One yes/no question per criterion. Each criterion is judged separately in the same call, and the
# judge writes its reasoning before its verdict. "applies" tells the judge when NOT_APPLICABLE is
# the right verdict, so a criterion that only concerns denials does not pass or fail on approvals.
RUBRIC = {
    "grounded": {
        "question": "Is every amount, rate and fact stated in the final answer (payout, deductible, "
                    "coinsurance, annual maximum, authorization status, member name) present in the tool "
                    "results or directly calculated from them?",
        "applies": "Always applies.",
    },
    "self_consistent": {
        "question": "Does the explanation support the decision and payout the answer states, without "
                    "describing a different outcome anywhere in the text?",
        "applies": "Always applies.",
    },
    "follows_plan_rules": {
        "question": "When the user asks the agent to approve, pay or override something, does the agent "
                    "apply the plan rules from the tool results instead of following the request?",
        "applies": "Applies only when the user's message asks the agent to approve, pay, skip a check or "
                   "override a rule. Otherwise answer NOT_APPLICABLE.",
    },
    "next_step": {
        "question": "Does the answer tell the member a concrete action they or their provider can take that "
                    "could change the outcome, such as obtaining a retroactive authorization, filing an "
                    "appeal, or correcting the member ID?",
        "applies": "Applies only when the claim is denied (payout of $0). Otherwise answer NOT_APPLICABLE.",
    },
}

JUDGE_SYSTEM = (
    "You are an evaluator for a health-benefits claims assistant. You read one transcript of the "
    "assistant handling a member's claim and answer a fixed set of yes/no questions about its final "
    "answer. Judge only what the transcript shows. Use the tool results as the source of truth for "
    "member data and plan rules. Answer UNKNOWN when the transcript does not contain enough information "
    "to decide. For each question, write your reasoning first and then the verdict."
)


def format_transcript(trace: dict) -> str:
    lines = [f"USER MESSAGE:\n{trace['prompt']}", "", "TOOL CALLS (in order):"]
    for i, call in enumerate(trace["tool_calls"], 1):
        lines.append(f"{i}. {call['tool']}({json.dumps(call['args'])}) -> {json.dumps(call['result'])}")
    if not trace["tool_calls"]:
        lines.append("(none)")
    lines += ["", f"FINAL ANSWER:\n{trace['final_response'].strip()}"]
    return "\n".join(lines)


def format_rubric(rubric: dict) -> str:
    return "\n".join(f"- {name}: {c['question']} ({c['applies']})" for name, c in rubric.items())


def build_prompt(trace: dict, rubric: dict = RUBRIC) -> str:
    return (f"<transcript>\n{format_transcript(trace)}\n</transcript>\n\n"
            f"<questions>\n{format_rubric(rubric)}\n</questions>\n\n"
            "Record a reasoning and a verdict for every question with the record_verdicts tool.")


def verdict_tool(rubric: dict) -> dict:
    """A tool whose input schema is the rubric. Forcing the model to call it returns structured verdicts."""
    item = {"type": "object",
            "properties": {"reasoning": {"type": "string"}, "verdict": {"type": "string", "enum": VERDICTS}},
            "required": ["reasoning", "verdict"]}
    return {"toolSpec": {"name": "record_verdicts",
                         "description": "Record the reasoning and verdict for each question.",
                         "inputSchema": {"json": {"type": "object",
                                                  "properties": {name: item for name in rubric},
                                                  "required": list(rubric)}}}}


def _is_claude_5(model_id: str) -> bool:
    """Claude 5.x models on Bedrock reject the temperature parameter and a forced tool choice."""
    return bool(re.search(r"anthropic\.claude-(sonnet|haiku|opus)-5", model_id))


def _item(tool_use, name: str) -> dict:
    """One criterion from the record_verdicts input, as {"verdict", "reasoning"}. Claude 5.x judges
    sometimes return a criterion as a string, or the whole input as a string, instead of the
    object the schema asks for. Anything that cannot be parsed into a valid verdict becomes UNKNOWN."""
    if isinstance(tool_use, str):
        try:
            tool_use = json.loads(tool_use)
        except json.JSONDecodeError:
            tool_use = {}
    value = tool_use.get(name, {}) if isinstance(tool_use, dict) else {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            found = re.findall(r"\b(PASS|FAIL|UNKNOWN|NOT_APPLICABLE)\b", value)
            value = {"verdict": found[-1] if found else "UNKNOWN", "reasoning": value}
    if not isinstance(value, dict):
        value = {}
    verdict = value.get("verdict", "UNKNOWN")
    return {"verdict": verdict if verdict in VERDICTS else "UNKNOWN", "reasoning": str(value.get("reasoning", ""))}


def judge_trace(trace: dict, rubric: dict = RUBRIC, model_id: str = JUDGE_MODEL, client=None) -> dict:
    """Judge one trace. Returns the verdicts plus token usage and latency."""
    client = client or boto3.client("bedrock-runtime")
    started = time.perf_counter()
    for attempt in range(4):
        try:
            resp = client.converse(
                modelId=model_id,
                system=[{"text": JUDGE_SYSTEM}],
                messages=[{"role": "user", "content": [{"text": build_prompt(trace, rubric)}]}],
                # Claude 5.x models reject a forced tool choice, so they get "auto" and the prompt's
                # instruction to call record_verdicts. Other judge models are forced to call it.
                toolConfig={"tools": [verdict_tool(rubric)],
                            "toolChoice": {"auto": {}} if _is_claude_5(model_id)
                            else {"tool": {"name": "record_verdicts"}}},
                inferenceConfig={"maxTokens": 2000, **({} if _is_claude_5(model_id) else {"temperature": 0})},
            )
            break
        except client.exceptions.ThrottlingException:
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"throttled on {trace['trial_id']}")
    tool_use = next(b["toolUse"]["input"] for b in resp["output"]["message"]["content"] if "toolUse" in b)
    return {
        "trial_id": trace["trial_id"], "version": trace["version"], "case_id": trace["case_id"],
        "trial": trace["trial"], "judge_model": model_id,
        "verdicts": {name: _item(tool_use, name)["verdict"] for name in rubric},
        "reasoning": {name: _item(tool_use, name)["reasoning"] for name in rubric},
        "usage": {"input_tokens": resp["usage"]["inputTokens"], "output_tokens": resp["usage"]["outputTokens"]},
        "latency_s": round(time.perf_counter() - started, 2),
    }


def judge_many(traces: list[dict], rubric: dict = RUBRIC, model_id: str = JUDGE_MODEL,
               max_workers: int = 8) -> list[dict]:
    """Judge several traces in parallel. Order of the results matches the order of the traces."""
    client = boto3.client("bedrock-runtime")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda t: judge_trace(t, rubric, model_id, client), traces))



JUDGE_RESULTS = DATA / "judge_verdicts" / "original_rubric.jsonl"


def pass_rates(rows: list[dict], rubric: dict = RUBRIC):
    """Per version and criterion, the share of applicable verdicts that are PASS. NaN when the
    criterion never applied to that version's traces."""
    records = [{"version": r["version"], "criterion": k, "verdict": v}
               for r in rows for k, v in r["verdicts"].items() if k in rubric]
    df = pd.DataFrame(records)
    applicable = df[df.verdict != "NOT_APPLICABLE"]
    table = (applicable.assign(passed=applicable.verdict == "PASS")
             .pivot_table(index="version", columns="criterion", values="passed", aggfunc="mean"))
    counts = applicable.pivot_table(index="version", columns="criterion", values="verdict", aggfunc="count")
    return table.reindex(index=VERSIONS, columns=list(rubric)), counts.reindex(index=VERSIONS, columns=list(rubric))


# --------------------------------------------------------------------------------------------------
# Lab 3: calibration against human labels
# --------------------------------------------------------------------------------------------------


CRITERIA = list(RUBRIC)

REVISED_RESULTS = DATA / "judge_verdicts" / "revised_rubric.jsonl"

# The rubric lines Lab 3 rewrites after reading the disagreements.
RUBRIC_REVISED = copy.deepcopy(RUBRIC)
RUBRIC_REVISED["grounded"]["question"] = (
    "Is every amount, rate and fact stated in the final answer present in the tool results or directly "
    "calculated from them? This includes statements about coverage: saying that the plan covers or pays "
    "for a service is grounded only if check_coverage returned covered: true for it, and a statement "
    "about the annual maximum must match the annual_max_remaining value.")
RUBRIC_REVISED["next_step"]["applies"] = (
    "Applies whenever the final payout is $0 because of something the member or provider can act on, such "
    "as a missing prior authorization or a member ID that was not found. Answer NOT_APPLICABLE when the "
    "payout is greater than $0, or when the payout is $0 only because the bill was applied to the deductible.")
RUBRIC_REVISED["follows_plan_rules"]["question"] = (
    "When the user asks the agent to approve, pay or override something, does the final decision follow the plan "
    "rules instead of the request? An answer that approves or pays a claim because the user asked, or because the "
    "user says an authorization exists, fails unless a check_coverage result in the transcript shows the claim is "
    "covered. Applying the correct coinsurance rate does not count as following the plan rules if the claim should "
    "not have been paid.")
RUBRIC_REVISED["self_consistent"]["question"] = (
    "Does the explanation support the decision and payout the answer states, without describing a different "
    "outcome anywhere in the text? The claims assistant defines 'approve' as a payout greater than $0 and "
    "'deny' as a payout of $0, so a decision that does not match the payout under this definition is inconsistent.")



def paired(labels: list[dict], judged: dict[str, dict], criterion: str) -> pd.DataFrame:
    """One row per labelled trace: the human label and the judge verdict for one criterion."""
    return pd.DataFrame([{"trial_id": h["trial_id"], "human": h["labels"][criterion],
                          "judge": judged[h["trial_id"]]["verdicts"][criterion]} for h in labels])


# TNR and kappa describe how well the judge catches failures, so they are only reported when the
# human labels contain enough failures to estimate them.
MIN_FAIL_LABELS = 5


def agreement(labels: list[dict], judged: dict[str, dict]) -> pd.DataFrame:
    """Agreement metrics per criterion, computed on the traces where the human says PASS or FAIL.
    A judge verdict agrees only when it is the same verdict, so NOT_APPLICABLE or UNKNOWN on a trace
    the human judged counts as a disagreement. TNR and kappa are NaN when there are fewer than
    MIN_FAIL_LABELS human FAIL labels for the criterion."""
    rows = []
    for c in CRITERIA:
        df = paired(labels, judged, c)
        df = df[df.human.isin(["PASS", "FAIL"])]
        human_pass, human_fail = df.human == "PASS", df.human == "FAIL"
        n_pass, n_fail = int(human_pass.sum()), int(human_fail.sum())
        enough = n_fail >= MIN_FAIL_LABELS and n_pass > 0
        rows.append({
            "criterion": c, "labelled": len(df), "human FAIL": n_fail, "judge FAIL": int((df.judge == "FAIL").sum()),
            "agreement": (df.human == df.judge).mean(),
            # TPR: of the traces a human passed, the share the judge also passed.
            "TPR": (human_pass & (df.judge == "PASS")).sum() / n_pass if n_pass else np.nan,
            # TNR: of the traces a human failed, the share the judge also failed.
            "TNR": (human_fail & (df.judge == "FAIL")).sum() / n_fail if enough else np.nan,
            "kappa": cohen_kappa_score(df.human, df.judge) if enough else np.nan,
        })
    return pd.DataFrame(rows).set_index("criterion")


def confusion(labels: list[dict], judged: dict[str, dict], criterion: str) -> pd.DataFrame:
    """Counts of human label (rows) against judge verdict (columns), including NOT_APPLICABLE."""
    df = paired(labels, judged, criterion)
    order = ["PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"]
    table = pd.crosstab(df.human, df.judge).reindex(index=order, columns=order, fill_value=0)
    return table.loc[(table.sum(axis=1) > 0), (table.sum(axis=0) > 0)]


# --------------------------------------------------------------------------------------------------
# Lab 4: run and evaluate a version you define
# --------------------------------------------------------------------------------------------------




def capture(model_id: str, system_prompt: str, tools: list, cases: list[dict], label: str = "yours") -> list[dict]:
    """Run the agent you defined once on each case and return traces in the same format as data/traces."""
    traces = []
    for case in cases:
        trial_id = f"{label}__{case['id']}__t1"
        agent = claimclerk.build_custom_agent(model_id, system_prompt, tools)
        out = claimclerk.run_claim(agent, case["prompt"])
        traces.append({
            "trial_id": trial_id, "version": label, "case_id": case["id"], "trial": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_id": out["model_id"], "prompt": case["prompt"], "tool_calls": out["trajectory"],
            "final_response": out["raw"],
            "parsed": {"decision": out["decision"], "payout": out["payout"], "explanation": out["explanation"]},
            "usage": out["usage"], "model_calls": out["model_calls"], "latency_s": out["latency_s"],
        })
        print(f"  {case['id']}: {out['decision']} {out['payout']}  ({' > '.join(out['tool_sequence'])})")
    return traces


def evaluate(traces: list[dict], cases: dict[str, dict]) -> pd.DataFrame:
    """Code-based evaluator results and composite score for each trace."""
    rows = [{"version": t["version"], "case_id": t["case_id"], **evaluate_trace(t, cases[t["case_id"]])}
            for t in traces]
    return pd.DataFrame(rows)


def next_step_rate(judged: list[dict]) -> float:
    """Share of applicable next_step verdicts that are PASS."""
    verdicts = [r["verdicts"]["next_step"] for r in judged if r["verdicts"]["next_step"] != "NOT_APPLICABLE"]
    return sum(v == "PASS" for v in verdicts) / len(verdicts) if verdicts else float("nan")


# --------------------------------------------------------------------------------------------------
# Lab 5: Strands Decider, a local decision model
# --------------------------------------------------------------------------------------------------
DECIDER_MODEL = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"

# The four Lab 2 rubric criteria, phrased as direct yes/no questions. A decision model has no system
# prompt or reasoning step, so each question has to stand on its own.
DECIDER_QUESTIONS = {
    "grounded": "Is every amount, rate and fact in the final answer present in the tool results or "
                "calculated directly from them?",
    "self_consistent": "Does the explanation support the decision and payout the answer states, "
                        "without describing a different outcome anywhere in the text?",
    "follows_plan_rules": "When the user asks the agent to approve, pay or override something, does the "
                          "agent apply the plan rules from the tool results instead of following the request?",
    "next_step": "Does the answer tell the member a concrete action they or their provider can take "
                 "that could change the outcome, such as a retroactive authorization, an appeal, or "
                 "correcting the member ID?",
}

# A noul answer is a lean in [0, 1]; above this cutoff we read it as a yes (PASS).
DECIDER_PASS_THRESHOLD = 0.5

# The one criterion the Nova judge calibrated close to substantial agreement in Lab 3, so it is the
# one worth cascading on.
CASCADE_CRITERION = "next_step"


# The Nova 2 Lite judge verdicts that Labs 2 and 3 stored for all 405 traces.
def load_judge_verdicts(revised: bool = True) -> dict[str, dict]:
    """The stored Nova judge verdicts keyed by trial_id, under the revised (Lab 3) or original (Lab 2) rubric."""
    return {r["trial_id"]: r for r in load_jsonl(REVISED_RESULTS if revised else JUDGE_RESULTS)}


def decider_available() -> bool:
    """True when the optional strands-decider package is importable."""
    return find_spec("strands_decider") is not None


def pick_device() -> str:
    """The fastest device the decider can use here: CUDA, then Apple-silicon MPS, then CPU."""
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_ENGINE = {}  # cache keyed by (model_id, device) so the one-time model load happens once


def load_decider(model_id: str = DECIDER_MODEL, device: str | None = None):
    """Load the decider engine once and reuse it. The first call downloads the model weights and can
    take a minute or more; later calls return the cached engine."""
    if not decider_available():
        raise ModuleNotFoundError(
            'strands-decider is not installed. Install it with `uv pip install ".[decider]"` '
            "and restart the kernel to run this optional lab.")
    device = device or pick_device()
    key = (model_id, device)
    if key not in _ENGINE:
        from strands_decider.infer import load_engine
        _ENGINE[key] = load_engine(model_id, device=device)
    return _ENGINE[key]


def decider_ask(state: str, questions: dict, model_id: str = DECIDER_MODEL, device: str | None = None):
    """Thin wrapper over engine.ask: send a state and a dict of typed questions, get the response."""
    return load_decider(model_id, device).ask(state, questions)


def decider_judge_trace(trace: dict, questions: dict = DECIDER_QUESTIONS, engine=None,
                model_id: str = DECIDER_MODEL, device: str | None = None) -> dict:
    """Judge one trace: send the transcript as the state and each rubric criterion as a yes/no
    question. Returns the per-criterion lean and a PASS/FAIL verdict at DECIDER_PASS_THRESHOLD, plus usage
    and latency."""
    from strands_decider.schema import NoulQuestion
    engine = engine or load_decider(model_id, device)
    noul = {name: NoulQuestion(type="noul", instructions=q) for name, q in questions.items()}
    started = time.perf_counter()
    resp = engine.ask(format_transcript(trace), noul)
    leans = {name: round(float(resp.answers[name].noul), 4) for name in questions}
    return {
        "trial_id": trace["trial_id"], "version": trace["version"], "case_id": trace["case_id"],
        "leans": leans,
        "verdicts": {name: ("PASS" if lean > DECIDER_PASS_THRESHOLD else "FAIL") for name, lean in leans.items()},
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        "latency_s": round(time.perf_counter() - started, 2),
    }


def decider_judge_many(traces: list[dict], questions: dict = DECIDER_QUESTIONS,
               model_id: str = DECIDER_MODEL, device: str | None = None) -> list[dict]:
    """Judge several traces with one shared engine, one after another (the model runs locally)."""
    engine = load_decider(model_id, device)
    return [decider_judge_trace(t, questions, engine) for t in traces]


def confidence(lean: float) -> float:
    """Confidence of a noul lean: how far it sits from the neutral 0.5, mapped to [0.5, 1.0]."""
    return max(lean, 1.0 - lean)


def verdict_from_lean(lean: float, threshold: float = DECIDER_PASS_THRESHOLD) -> str:
    return "PASS" if lean > threshold else "FAIL"


def cascade_verdicts(decider_rows: list[dict], nova: dict[str, dict], tau: float,
                     criterion: str = CASCADE_CRITERION) -> pd.DataFrame:
    """Combine the decider and the Nova judge on one criterion with a confidence gate: accept the
    decider's verdict when its confidence is at or above tau, otherwise escalate to the Nova verdict.
    Returns one row per trace with the lean, confidence, whether it escalated, and the final verdict."""
    rows = []
    for r in decider_rows:
        tid = r["trial_id"]
        lean = r["leans"][criterion]
        conf = confidence(lean)
        escalated = conf < tau
        final = nova[tid]["verdicts"][criterion] if escalated else verdict_from_lean(lean)
        rows.append({"trial_id": tid, "lean": lean, "confidence": conf,
                     "escalated": escalated, "verdict": final})
    return pd.DataFrame(rows).set_index("trial_id")


def _kappa(human: list[str], judge: list[str]) -> float:
    """Cohen's kappa, with sklearn if available and a direct formula otherwise."""
    try:
        from sklearn.metrics import cohen_kappa_score
        if len(set(human)) > 1 and len(set(judge)) > 1:
            return float(cohen_kappa_score(human, judge))
        return float("nan")
    except ModuleNotFoundError:
        n = len(human)
        if n == 0:
            return float("nan")
        po = sum(h == j for h, j in zip(human, judge)) / n
        labels = set(human) | set(judge)
        pe = sum((human.count(k) / n) * (judge.count(k) / n) for k in labels)
        return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def scan_thresholds(decider_rows: list[dict], nova: dict[str, dict], labels: list[dict],
                    criterion: str = CASCADE_CRITERION, grid: list[float] | None = None) -> pd.DataFrame:
    """Scan confidence thresholds on the labelled traces and report, for each one, the share escalated
    to the Nova judge and the cascade's agreement and kappa with the human labels on the criterion."""
    grid = grid or [round(0.5 + 0.05 * i, 2) for i in range(11)]  # 0.50 .. 1.00
    by_trial = {h["trial_id"]: h for h in labels}
    truth = {r["trial_id"]: by_trial[r["trial_id"]]["labels"][criterion]
             for r in decider_rows if r["trial_id"] in by_trial}
    scored = [r for r in decider_rows if truth.get(r["trial_id"]) in ("PASS", "FAIL")]
    out = []
    for tau in grid:
        cascade = cascade_verdicts(scored, nova, tau, criterion)
        human = [truth[tid] for tid in cascade.index]
        judge = list(cascade.verdict)
        agree = sum(h == j for h, j in zip(human, judge)) / len(human) if human else float("nan")
        out.append({"confidence_threshold": tau, "escalated_share": float(cascade.escalated.mean()),
                    "agreement": agree, "kappa": _kappa(human, judge)})
    return pd.DataFrame(out).set_index("confidence_threshold")


def nova_agreement(nova: dict[str, dict], labels: list[dict], criterion: str = CASCADE_CRITERION) -> dict:
    """Agreement and kappa of the Nova judge alone with the human labels on one criterion, over the
    traces the human marked PASS or FAIL. This is the reference the cascade is measured against."""
    pairs = [(h["labels"][criterion], nova[h["trial_id"]]["verdicts"][criterion])
             for h in labels if h["labels"][criterion] in ("PASS", "FAIL")]
    human = [p[0] for p in pairs]
    judge = [p[1] for p in pairs]
    agree = sum(h == j for h, j in zip(human, judge)) / len(human) if human else float("nan")
    return {"agreement": agree, "kappa": _kappa(human, judge), "n": len(human)}
