"""
The ClaimClerk rules engine, which decides how every claim should be adjudicated. The agent's
compute_payout tool (claimclerk.py) and the task builder (data/build_cases.py) both import from
this module, so the payouts the tools return and the expected answers in data/cases.json are
computed by the same code.

The agent never calls adjudicate(). It has to reach the same decision and payout by calling
its tools (verify_member, fetch_policy, check_coverage and compute_payout), and adjudicate()
provides the expected answer that the labs grade against.

The members and plans in fixtures.py are fictional. Data of the same shape about real people,
such as deductibles, coverage decisions, prior authorizations and claim payouts, is Protected
Health Information under HIPAA. Adapting this pattern to real member data requires the
matching controls, including a Business Associate Agreement with AWS, encryption in transit
and at rest, access logging, and data minimization. See the AWS HIPAA Eligible Services
reference and the AWS shared responsibility model.
"""
from __future__ import annotations

from .fixtures import get_member, PREAUTH_REQUIRED, COINSURANCE


def requires_preauth(procedure: str) -> bool:
    return procedure in PREAUTH_REQUIRED


def has_auth(member: dict, procedure: str) -> bool:
    return procedure in member.get("auths", set())


def compute_payout(member: dict, billed: float, network: str) -> dict:
    """Pure payout math: remaining deductible, then coinsurance, then annual cap."""
    if network == "oon":
        ded_rem = member["oon_deductible_total"] - member["oon_deductible_met"]
        coins = COINSURANCE["oon"]
    else:
        ded_rem = member["deductible_total"] - member["deductible_met"]
        coins = COINSURANCE["in"]
    after_ded = max(0.0, billed - max(0.0, ded_rem))
    payout = round(after_ded * coins, 2)
    annual_rem = round(member["annual_max"] - member["annual_used"], 2)
    capped = round(min(payout, annual_rem), 2)
    return {
        "deductible_applied": round(min(billed, max(0.0, ded_rem)), 2),
        "coinsurance": coins,
        "payout_before_cap": payout,
        "annual_max_remaining": annual_rem,
        "payout": capped,
        "capped_by_annual_max": capped < payout,
    }


def adjudicate(member_id: str, procedure: str, billed: float, network: str) -> dict:
    """Full adjudication -> {decision, payout, rule}. This is the oracle."""
    m = get_member(member_id)
    if not m:
        return {"decision": "deny", "payout": 0.0, "rule": "member not found"}

    if requires_preauth(procedure) and not has_auth(m, procedure):
        return {"decision": "deny", "payout": 0.0,
                "rule": f"{procedure} requires prior authorization and none is on file "
                        f"for {member_id} -> deny, $0."}

    comp = compute_payout(m, billed, network)
    payout = comp["payout"]
    net_word = "out-of-network 60%" if network == "oon" else "in-network 80%"
    parts = []
    if comp["deductible_applied"] > 0:
        parts.append(f"apply ${comp['deductible_applied']:.0f} remaining deductible")
    parts.append(f"{net_word} coinsurance")
    if comp["capped_by_annual_max"]:
        parts.append(f"capped at ${comp['annual_max_remaining']:.0f} annual maximum remaining")
    if requires_preauth(procedure) and has_auth(m, procedure):
        parts.insert(0, f"{procedure} requires prior authorization and it is on file")
    rule = "; ".join(parts) + f" -> pays ${payout:.2f}."
    return {"decision": "approve" if payout > 0 else "deny", "payout": payout, "rule": rule}
