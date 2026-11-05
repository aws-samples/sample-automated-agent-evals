"""
ClaimClerk, the health-benefits claims agent that the labs evaluate, built with Strands Agents
and a model on Amazon Bedrock.

The agent reads a member's claim in natural language, decides whether to approve or deny it,
computes the payout, and explains the decision. It has four tools, and the baseline system
prompt asks the model to call them in this order:

    verify_member -> fetch_policy -> check_coverage -> compute_payout

check_coverage applies the prior-authorization requirement, so an agent that skips it and calls
compute_payout directly can return a plausible payout on a claim that should have been denied.
On claims that need no authorization the same shortcut gives the correct payout, which is why
the labs grade the tool calls as well as the final answer.

build_agent() constructs the baseline (v1) and the four changed versions that the labs compare.
Every tool call is recorded on a per-invocation Trajectory so that graders can check the tool
calls. The only AWS call in this module is the Bedrock model invocation made by Strands.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from strands import Agent, tool
from strands.models import BedrockModel
from strands.tools.executors import SequentialToolExecutor

from .fixtures import get_member, PREAUTH_REQUIRED, COINSURANCE
from .rules import compute_payout as rules_compute_payout

# --- models -----------------------------------------------------------------
# v1 uses Claude Sonnet 5.5, and the cheaper-model versions use Claude Haiku 5.5 or Amazon Nova 2
# Lite. Claude 5.x models on Bedrock reject the temperature parameter, so they run at their default
# sampling and repeated trials can differ. Nova 2 Lite runs at temperature 0.
MODEL_V1 = os.environ.get("CLAIMCLERK_MODEL_V1", "us.anthropic.claude-sonnet-5-5")
MODEL_V2 = os.environ.get("CLAIMCLERK_MODEL_V2", "us.anthropic.claude-haiku-5-5")
MODEL_NOVA = os.environ.get("CLAIMCLERK_MODEL_NOVA", "us.amazon.nova-2-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")


# --- trajectory capture -----------------------------------------------------
@dataclass
class Trajectory:
    """Records the ordered tool calls of a single claim invocation."""
    steps: list[dict] = field(default_factory=list)

    def record(self, tool_name: str, args: dict, result: Any):
        self.steps.append({"tool": tool_name, "args": args, "result": result})

    @property
    def tool_sequence(self) -> list[str]:
        return [s["tool"] for s in self.steps]


# The tools record their calls on the trajectory of the claim being processed. run_claim() sets
# this module-level slot before each claim and clears it afterwards. This works because one
# process handles one claim at a time, so pre-run.sh runs agent versions in parallel as separate
# processes.
_active: Trajectory | None = None


# --- the four tools ---------------------------------------------------------
@tool
def verify_member(member_id: str) -> str:
    """Verify a member exists and return their name and plan.

    Args:
        member_id: the member identifier, e.g. "M-2001".
    """
    m = get_member(member_id)
    out = ({"found": False} if not m
           else {"found": True, "name": m["name"], "plan": m["plan"]})
    if _active is not None:
        _active.record("verify_member", {"member_id": member_id}, out)
    return json.dumps(out)


@tool
def fetch_policy(member_id: str) -> str:
    """Fetch the member's plan financials: deductible status, network
    coinsurance rates, and annual maximum remaining.

    NOTE: this benefits lookup returns no information about prior-authorization
    requirements or the authorizations a member has on file. That information is
    only available from check_coverage.

    Args:
        member_id: the member identifier, e.g. "M-2001".
    """
    m = get_member(member_id)
    if not m:
        out = {"error": "member not found"}
    else:
        out = {
            "plan": m["plan"],
            "in_network_coinsurance": COINSURANCE["in"],
            "oon_coinsurance": COINSURANCE["oon"],
            "deductible_remaining": round(m["deductible_total"] - m["deductible_met"], 2),
            "oon_deductible_remaining": round(m["oon_deductible_total"] - m["oon_deductible_met"], 2),
            "annual_max_remaining": round(m["annual_max"] - m["annual_used"], 2),
        }
    if _active is not None:
        _active.record("fetch_policy", {"member_id": member_id}, out)
    return json.dumps(out)


@tool
def check_coverage(member_id: str, procedure: str, network: str) -> str:
    """Apply coverage rules for a claim BEFORE computing a payout. Returns
    whether the claim is covered, and if not, why. This is where the
    prior-authorization gate is enforced.

    Args:
        member_id: the member identifier.
        procedure: normalized procedure key, e.g. "mri", "office_visit",
            "physical_therapy", "sleep_study", "urgent_care", "specialist".
        network: "in" for in-network or "oon" for out-of-network.
    """
    m = get_member(member_id)
    proc = _normalize(procedure)
    if not m:
        out = {"covered": False, "reason": "member not found"}
    elif proc in PREAUTH_REQUIRED and proc not in m["auths"]:
        out = {"covered": False, "requires_preauth": True, "auth_on_file": False,
               "reason": f"{proc} requires prior authorization and none is on file"}
    else:
        out = {"covered": True,
               "requires_preauth": proc in PREAUTH_REQUIRED,
               "auth_on_file": proc in m["auths"],
               "coinsurance": COINSURANCE.get(network, COINSURANCE["in"]),
               "reason": "covered under plan rules"}
    if _active is not None:
        _active.record("check_coverage", {"member_id": member_id, "procedure": proc, "network": network}, out)
    return json.dumps(out)


@tool
def compute_payout(member_id: str, billed_amount: float, network: str) -> str:
    """Compute the plan's payout for a covered claim: apply the remaining
    deductible, then the coinsurance, then cap by the annual maximum remaining.

    Args:
        member_id: the member identifier.
        billed_amount: the billed amount in USD.
        network: "in" or "oon".
    """
    m = get_member(member_id)
    if not m:
        out = {"error": "member not found", "payout": 0.0}
    else:
        # The arithmetic comes from the rules engine in rules.py, which also computes
        # the expected payouts in data/cases.json, so the tool and the expected
        # answers always agree.
        comp = rules_compute_payout(m, float(billed_amount), network)
        out = {"billed": billed_amount, **comp}
    if _active is not None:
        _active.record("compute_payout", {"member_id": member_id, "billed_amount": billed_amount, "network": network}, out)
    return json.dumps(out)


def _normalize(procedure: str) -> str:
    p = procedure.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "mri_scan": "mri", "magnetic_resonance_imaging": "mri",
        "pt": "physical_therapy", "therapy": "physical_therapy",
        "sleep_test": "sleep_study",
        "ct": "ct_scan", "cat_scan": "ct_scan",
    }
    return aliases.get(p, p)


# --- system prompts -----------------------------------------------------------
_RULES = (
    "You are ClaimClerk, a health-benefits claims assistant. For each member "
    "claim you must decide the outcome, compute the plan's payout, and explain "
    "the decision in one or two sentences.\n\n"
    "DECISION is exactly one of:\n"
    "  approve - the plan pays something (payout > 0), whether in full or in part\n"
    "  deny    - the plan pays nothing (payout = 0), e.g. a required prior "
    "authorization is missing\n"
    "The payout amount itself conveys how much the member still owes; do not use "
    "any other decision label.\n\n"
    "You have four tools: verify_member, fetch_policy, check_coverage, compute_payout.\n\n"
    "Procedure keys you may use: office_visit, specialist, urgent_care, mri, "
    "ct_scan, sleep_study, physical_therapy, surgery. Network is 'in' or 'oon'.\n\n"
    "When you have finished, respond with ONLY a JSON object on the final line:\n"
    '{"decision": "approve|deny", "payout": <number>, "explanation": "<text>"}'
)

# v1, v2-model and v2-nova: the full prompt, which lists the tool order and the
# authorization check.
SYSTEM_V1 = (
    _RULES + "\n\n"
    "PROCEDURE. Follow these steps in order every time:\n"
    "1. verify_member to confirm the member.\n"
    "2. fetch_policy to load their deductible, coinsurance rates and annual maximum.\n"
    "3. check_coverage BEFORE computing anything. If a procedure requires prior "
    "authorization and none is on file, the claim is DENIED with payout 0, so do "
    "not compute a payout in that case.\n"
    "4. compute_payout only for covered claims.\n"
    "Never skip check_coverage. A correct payout reached without verifying "
    "coverage is not acceptable."
)

# v2-prompt and v2-tools: a shorter prompt written to reduce tool calls and cost. It drops
# the tool order and the instruction to always check authorization, which is the kind of
# change a team makes to save cost without re-running its evaluations.
SYSTEM_V2_PROMPT = (
    _RULES + "\n\n"
    "Keep it efficient: call only the tools you actually need and avoid "
    "redundant lookups. Once you know the member, their plan, and the billed "
    "amount, decide and compute the payout. Be concise."
)


# --- build + run ------------------------------------------------------------
ALL_TOOLS = [verify_member, fetch_policy, check_coverage, compute_payout]


def build_agent(version: str = "v1", change: str = "tools", trace_attributes: dict | None = None) -> Agent:
    """Construct a ClaimClerk agent for one version.

    version: "v1" for the baseline, or "v2" for a changed version. change is ignored for v1.
    change:  the change a v2 agent makes to the baseline.
      "model"  Claude Haiku 5.5 in place of Claude Sonnet 5.5, with the same prompt and tools
               (v2-model).
      "nova"   Amazon Nova 2 Lite in place of Claude Sonnet 5.5, with the same prompt and tools
               (v2-nova).
      "prompt" the shorter system prompt on Claude Sonnet 5.5 (v2-prompt).
      "tools"  check_coverage removed, as if someone believed fetch_policy already covered
               it, with Claude Haiku 5.5 and the shorter prompt (v2-tools). Without
               check_coverage the agent has no way to see whether an authorization is on
               file, so it pays out claims that should be denied.
    trace_attributes: OpenTelemetry attributes added to every span, such as session.id.
    """
    if version == "v1":
        model_id, system, tools = MODEL_V1, SYSTEM_V1, ALL_TOOLS
    elif change == "model":
        model_id, system, tools = MODEL_V2, SYSTEM_V1, ALL_TOOLS       # cheaper model
    elif change == "nova":
        model_id, system, tools = MODEL_NOVA, SYSTEM_V1, ALL_TOOLS     # cheaper model, Nova
    elif change == "prompt":
        model_id, system, tools = MODEL_V1, SYSTEM_V2_PROMPT, ALL_TOOLS  # weaker prompt
    else:  # "tools": check_coverage removed
        model_id, system = MODEL_V2, SYSTEM_V2_PROMPT
        tools = [verify_member, fetch_policy, compute_payout]

    return build_custom_agent(model_id, system, tools, trace_attributes)


def supports_temperature(model_id: str) -> bool:
    """Claude 5.x models on Bedrock reject the temperature parameter. Other models accept it."""
    return not re.search(r"anthropic\.claude-(sonnet|haiku|opus)-5", model_id)


def build_custom_agent(model_id: str, system_prompt: str, tools: list | None = None,
                       trace_attributes: dict | None = None) -> Agent:
    """Construct a ClaimClerk agent from any model, system prompt and tool list. The optional
    Lab 4 uses this to build and evaluate a version of your own."""
    params = {"temperature": 0.0} if supports_temperature(model_id) else {}
    model = BedrockModel(model_id=model_id, region_name=REGION, **params)
    # Sequential execution keeps the recorded tool order, and the span order, identical to the
    # order the model chose. trace_attributes carries session.id, which the AgentCore Evaluate
    # API requires on the spans.
    return Agent(model=model, system_prompt=system_prompt, tools=tools if tools is not None else ALL_TOOLS,
                 tool_executor=SequentialToolExecutor(),
                 trace_attributes=trace_attributes or {},
                 callback_handler=None)


_JSON_RE = re.compile(r"\{[^{}]*\"decision\"[^{}]*\}", re.DOTALL)


def run_claim(agent: Agent, claim_text: str) -> dict:
    """Run one claim through the agent and return the structured outcome plus
    the captured trajectory. Sets up the per-invocation trajectory slot."""
    global _active
    _active = Trajectory()
    started = time.perf_counter()
    try:
        result = agent(claim_text)
        text = str(result)
    finally:
        traj = _active
        _active = None
    latency_s = round(time.perf_counter() - started, 2)

    usage = result.metrics.accumulated_usage
    parsed = _parse_output(text)
    return {
        "decision": parsed.get("decision"),
        "payout": parsed.get("payout"),
        "explanation": parsed.get("explanation", text.strip()[:400]),
        "raw": text,
        "trajectory": traj.steps,
        "tool_sequence": traj.tool_sequence,
        "model_id": agent.model.config.get("model_id"),
        "usage": {"input_tokens": usage.get("inputTokens", 0),
                  "output_tokens": usage.get("outputTokens", 0)},
        "model_calls": result.metrics.cycle_count,
        "latency_s": latency_s,
    }


def _parse_output(text: str) -> dict:
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback when there is no JSON object: take the first decision word and the first number.
    dec = re.search(r"\b(approve|deny|partial)\b", text, re.I)
    pay = re.search(r"\$?\s*([\d,]+\.?\d*)", text)
    return {
        "decision": dec.group(1).lower() if dec else None,
        "payout": float(pay.group(1).replace(",", "")) if pay else None,
        "explanation": text.strip()[:400],
    }
