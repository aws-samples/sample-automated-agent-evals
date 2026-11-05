"""
ClaimClerk fixture data, which stands in for the member and plan systems that a production
claims agent would query.

Every member, plan and prior authorization here is fictional. The tools in claimclerk.py read
only from this module, so a given tool call always returns the same result, and the rules
engine in rules.py can compute the correct answer for every task.

Plan rules, which rules.py applies to compute the expected answers:
  - In-network coinsurance: plan pays 80% after the member's deductible is met.
  - Out-of-network coinsurance: plan pays 60%, separate OON deductible.
  - Some procedures require PRIOR AUTHORIZATION. No auth on file => deny, $0.
  - Every member has an annual benefit maximum; payout is capped by what remains.
"""

# Procedures that require prior authorization before any payout.
PREAUTH_REQUIRED = {"mri", "sleep_study", "physical_therapy", "ct_scan", "surgery"}

# coinsurance the plan pays, by network status
COINSURANCE = {"in": 0.80, "oon": 0.60}

MEMBERS = {
    # member_id -> profile. Amounts in USD.
    "M-2001": {
        "name": "Jordan Avery", "plan": "PPO-Gold",
        "deductible_total": 1500.0, "deductible_met": 1500.0,   # fully met
        "oon_deductible_total": 3000.0, "oon_deductible_met": 3000.0,
        "annual_max": 50000.0, "annual_used": 2000.0,
        "auths": {"physical_therapy"},                          # has a PT auth on file
    },
    "M-2002": {
        "name": "Sam Rivera", "plan": "PPO-Gold",
        "deductible_total": 1500.0, "deductible_met": 1350.0,   # $150 remaining
        "oon_deductible_total": 3000.0, "oon_deductible_met": 3000.0,
        "annual_max": 50000.0, "annual_used": 8000.0,
        "auths": set(),
    },
    "M-2003": {
        "name": "Casey Kim", "plan": "PPO-Gold",
        "deductible_total": 1500.0, "deductible_met": 1500.0,
        "oon_deductible_total": 3000.0, "oon_deductible_met": 3000.0,
        "annual_max": 50000.0, "annual_used": 49880.0,          # only $120 left under annual max
        "auths": set(),
    },
    "M-2004": {
        "name": "Riley Chen", "plan": "PPO-Gold",
        "deductible_total": 1500.0, "deductible_met": 1500.0,
        "oon_deductible_total": 3000.0, "oon_deductible_met": 3000.0,
        "annual_max": 50000.0, "annual_used": 500.0,
        "auths": set(),                                         # NO pre-auth on file
    },
    "M-2005": {
        "name": "Devon Blake", "plan": "PPO-Gold",
        "deductible_total": 1500.0, "deductible_met": 1500.0,   # met
        "oon_deductible_total": 3000.0, "oon_deductible_met": 3000.0,
        "annual_max": 50000.0, "annual_used": 1200.0,
        "auths": {"mri"},                                       # has an MRI auth on file
    },
    "M-2006": {
        "name": "Harper Lee", "plan": "PPO-Gold",
        "deductible_total": 1500.0, "deductible_met": 900.0,    # $600 remaining
        "oon_deductible_total": 3000.0, "oon_deductible_met": 1500.0,  # $1500 OON remaining
        "annual_max": 50000.0, "annual_used": 3000.0,
        "auths": set(),
    },
    "M-2007": {
        "name": "Quinn Ortiz", "plan": "PPO-Gold",
        "deductible_total": 1500.0, "deductible_met": 1500.0,
        "oon_deductible_total": 3000.0, "oon_deductible_met": 3000.0,
        "annual_max": 50000.0, "annual_used": 49700.0,          # $300 left under annual max
        "auths": set(),
    },
    "M-2008": {
        "name": "Alex Morgan", "plan": "PPO-Gold",
        "deductible_total": 1500.0, "deductible_met": 1500.0,
        "oon_deductible_total": 3000.0, "oon_deductible_met": 3000.0,
        "annual_max": 50000.0, "annual_used": 6000.0,
        "auths": {"physical_therapy"},                          # PT auth on file
    },
}


def get_member(member_id: str):
    return MEMBERS.get(member_id)
