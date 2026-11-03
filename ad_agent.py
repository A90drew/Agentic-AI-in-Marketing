"""
Ad Optimization Agent
=====================
Allocates a fixed daily budget across Search / Social / Display to maximize
total conversions, then compares the result to an equal split baseline.

Run:  python ad_agent.py            (creates ad_data.csv on first run)
      python ad_agent.py --regenerate --days 30 --seed 7
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# SETTINGS (change these to experiment)
CHANNELS = ["Search", "Social", "Display"]
TOTAL_DAILY_BUDGET = 300.0     # dollars per day, split across all channels

SHIFT_PCT = 0.15               # top channel gains 15% of its own budget per day
MAX_CHANGE = 0.20              # no channel moves more than +/-20% in a day
MIN_SHARE = 0.10               # every channel keeps >= 10% of the total (never 0%)
MAX_SHARE = 0.60               # no channel may take more than 60% of the total
LOOKBACK_DAYS = 5              # days of recent results the agent learns from
MIN_GAP = 0.10                 # ignore efficiency gaps under 10% (likely noise)
RETURNS_EXPONENT = 0.9         # <1 = diminishing returns when spend rises; 1.0 = linear

MIN_BUDGET = MIN_SHARE * TOTAL_DAILY_BUDGET
MAX_BUDGET = MAX_SHARE * TOTAL_DAILY_BUDGET

CSV_PATH = Path("ad_data.csv")
LOG_PATH = Path("decision_log.csv")

# The agent's role and rules, written as a system prompt. The agent below is
# rule-based (no API key needed); this text documents exactly what it does.
SYSTEM_PROMPT = f"""
ROLE: You are an ad budget optimization agent for a small advertiser.
GOAL: Maximize total conversions from a fixed ${TOTAL_DAILY_BUDGET:.0f}/day budget split across {", ".join(CHANNELS)}.
INPUTS: Daily channel metrics only (date, channel, spend, impressions, clicks, conversions). Never use personal or user-level data.
OUTPUT: A daily budget per channel plus a plain, factual reason for every change (no hype).
RULES:
- Score each channel by conversions per dollar (1 / CPA) over the last {LOOKBACK_DAYS} days.
- Move {SHIFT_PCT:.0%} of the top channel's budget to it from the weakest channel that can still give.
- Never change a channel by more than {MAX_CHANGE:.0%} in a day.
- Keep every channel between ${MIN_BUDGET:.0f} and ${MAX_BUDGET:.0f} per day.
- If the best and worst channels are within {MIN_GAP:.0%} of each other, hold the budget.
- Log a reason for every decision.
""".strip()



# generate mock data and load it
def generate_mock_data(n_days=21, seed=42, start="2026-10-01"):
    """Create mock daily data for 3 channels at an equal historical spend."""
    rng = np.random.default_rng(seed)
    equal_spend = TOTAL_DAILY_BUDGET / len(CHANNELS)
    rows = []
    for day, date in enumerate(pd.date_range(start, periods=n_days)):
        # Hidden "true" behavior of each channel on this day.
        profiles = {
            "Search":  dict(cpm=15.0, ctr=0.030, cvr=0.042),
            # Social improves between day 6 and day 11 (e.g., a creative refresh).
            "Social":  dict(cpm=8.0, ctr=0.012,
                            cvr=float(np.interp(day, [6, 11], [0.030, 0.075]))),
            # Display slowly gets worse.
            "Display": dict(cpm=4.0, ctr=0.005,
                            cvr=float(np.interp(day, [0, n_days - 1], [0.026, 0.020]))),
        }
        for channel in CHANNELS:
            p = profiles[channel]
            impressions = int(equal_spend / p["cpm"] * 1000 * rng.normal(1.0, 0.05))
            clicks = int(rng.binomial(impressions, p["ctr"]))
            conversions = int(rng.binomial(clicks, p["cvr"]))
            rows.append({
                "date": date.strftime("%Y-%m-%d"), "channel": channel,
                "spend": equal_spend, "impressions": impressions,
                "clicks": clicks, "conversions": conversions,
            })
    return pd.DataFrame(rows)


def load_data(path):
    """Read the CSV and check it has what the agent needs."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "channel", "spend", "impressions", "clicks", "conversions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    unknown = set(df["channel"]) - set(CHANNELS)
    if unknown:
        raise ValueError(f"CSV has channels not in CHANNELS: {sorted(unknown)}")
    return df.sort_values(["date", "channel"]).reset_index(drop=True)

# performance metrics
def summarize(rows):
    """CTR, CVR, CPA and conversions-per-dollar for any slice of data."""
    spend = rows["spend"].sum()
    impressions = rows["impressions"].sum()
    clicks = rows["clicks"].sum()
    conversions = rows["conversions"].sum()
    return {
        "spend": spend, "impressions": impressions,
        "clicks": clicks, "conversions": conversions,
        "ctr": clicks / impressions if impressions else 0.0,
        "cvr": conversions / clicks if clicks else 0.0,
        "cpa": spend / conversions if conversions else float("inf"),
        "conv_per_dollar": conversions / spend if spend else 0.0,
    }


def fmt_cpa(cpa):
    return "n/a" if cpa == float("inf") else f"${cpa:.2f}"


# guardrail check (runs on every decision, agent or baseline)
def check_guardrails(prev, new):
    tol = 1e-6
    if abs(sum(new.values()) - TOTAL_DAILY_BUDGET) > tol:
        raise RuntimeError("Guardrail broken: budgets must sum to the daily total.")
    for c in CHANNELS:
        if new[c] < MIN_BUDGET - tol:
            raise RuntimeError(f"Guardrail broken: {c} is below the minimum budget.")
        if new[c] > MAX_BUDGET + tol:
            raise RuntimeError(f"Guardrail broken: {c} is above the maximum budget.")
        if abs(new[c] - prev[c]) > MAX_CHANGE * prev[c] + tol:
            raise RuntimeError(f"Guardrail broken: {c} moved more than {MAX_CHANGE:.0%}.")


# THE AGENT: decide tomorrow's budgets from recent results
def agent_policy(day_idx, prev, history):
    """Return (new_budgets, reasons) where reasons maps channel -> log string."""
    if day_idx == 0:
        return dict(prev), {c: "Day 1: no history yet, so start with the equal split."
                            for c in CHANNELS}

    def hold(why):
        return dict(prev), {c: f"HOLD: {why}" for c in CHANNELS}

    # Learn from the last few days of results (the "lightweight learning loop").
    n_days = min(LOOKBACK_DAYS, day_idx)
    window = history[history["day"] >= day_idx - LOOKBACK_DAYS]
    stats = {c: summarize(window[window["channel"] == c]) for c in CHANNELS}

    # Rank channels by conversions per dollar (higher = better).
    ranked = sorted(CHANNELS, key=lambda c: stats[c]["conv_per_dollar"], reverse=True)
    top = ranked[0]
    # Donor = weakest channel that still has room above the minimum budget.
    donor = next((c for c in reversed(ranked[1:]) if prev[c] - MIN_BUDGET > 0.01), None)

    if stats[top]["conv_per_dollar"] == 0:
        return hold("no conversions in the lookback window, so there is nothing to learn from.")
    if donor is None:
        return hold(f"every other channel is already at the ${MIN_BUDGET:.0f} minimum.")

    gap = 1 - stats[donor]["conv_per_dollar"] / stats[top]["conv_per_dollar"]
    cpa_t, cpa_d = fmt_cpa(stats[top]["cpa"]), fmt_cpa(stats[donor]["cpa"])
    if gap < MIN_GAP:
        return hold(f"{top} leads {donor} by only {gap:.0%} (CPA {cpa_t} vs {cpa_d}), "
                    f"under the {MIN_GAP:.0%} threshold, so it may be noise.")

    # How much can move? The smallest of all the limits wins.
    limits = [
        (f"the {SHIFT_PCT:.0%} daily step", SHIFT_PCT * prev[top]),
        (f"{donor}'s -{MAX_CHANGE:.0%} daily limit", MAX_CHANGE * prev[donor]),
        (f"{donor}'s ${MIN_BUDGET:.0f} minimum", prev[donor] - MIN_BUDGET),
        (f"{top}'s ${MAX_BUDGET:.0f} maximum", MAX_BUDGET - prev[top]),
    ]
    limit_text, moved = min(limits, key=lambda item: item[1])
    if moved <= 0.01:
        return hold(f"{top} is the best performer (CPA {cpa_t}) but cannot grow: "
                    f"limited by {limit_text}.")

    new = dict(prev)
    new[top] += moved
    new[donor] -= moved

    reasons = {c: "No change: neither the top performer nor today's donor." for c in CHANNELS}
    reasons[top] = (f"{top} +{moved / prev[top]:.1%} (${prev[top]:.2f} -> ${new[top]:.2f}): "
                    f"best conversions per dollar over the last {n_days} day(s) "
                    f"(CPA {cpa_t}, CVR {stats[top]['cvr']:.1%}) vs {donor} CPA {cpa_d}. "
                    f"Size set by {limit_text}.")
    reasons[donor] = (f"{donor} -{moved / prev[donor]:.1%} (${prev[donor]:.2f} -> ${new[donor]:.2f}): "
                      f"funds {top}; weakest channel with room to give (CPA {cpa_d}). "
                      f"Stays above ${MIN_BUDGET:.0f} so it keeps generating data.")
    return new, reasons


def equal_split_policy(day_idx, prev, history):
    """Baseline: the same budget for every channel, every day."""
    equal = TOTAL_DAILY_BUDGET / len(CHANNELS)
    return {c: equal for c in CHANNELS}, {c: "Baseline: fixed equal split." for c in CHANNELS}



# SIMULATION: replay each day's data under a given budget
def simulate_day(day_rows, budgets):
    """What would this day's results be at these budgets?
    Results scale with (budget / historical spend) ** RETURNS_EXPONENT."""
    results = []
    for _, row in day_rows.iterrows():
        channel = row["channel"]
        scale = (budgets[channel] / row["spend"]) ** RETURNS_EXPONENT
        results.append({
            "channel": channel,
            "spend": budgets[channel],
            "impressions": row["impressions"] * scale,
            "clicks": row["clicks"] * scale,
            "conversions": row["conversions"] * scale,
        })
    return results


def run_policy(df, policy):
    """Run a budgeting policy day by day. Returns (results history, decision log)."""
    dates = sorted(df["date"].drop_duplicates())
    prev = {c: TOTAL_DAILY_BUDGET / len(CHANNELS) for c in CHANNELS}
    history_rows, log_rows = [], []

    for day_idx, date in enumerate(dates):
        history = pd.DataFrame(history_rows)            # only past days: no peeking
        budgets, reasons = policy(day_idx, prev, history)
        check_guardrails(prev, budgets)

        for result in simulate_day(df[df["date"] == date], budgets):
            history_rows.append({"day": day_idx, "date": date, **result})
        for c in CHANNELS:
            log_rows.append({
                "date": date, "channel": c,
                "old_budget": prev[c], "new_budget": budgets[c],
                "change_pct": (budgets[c] - prev[c]) / prev[c],
                "reason": reasons[c],
            })
        prev = budgets

    return pd.DataFrame(history_rows), pd.DataFrame(log_rows)

# OUTPUT: decision log and evaluation
def print_decision_log(log):
    print("\n=== AGENT DECISION LOG ===")
    for date, day in log.groupby("date"):
        budgets = ", ".join(f"{r.channel} ${r.new_budget:.0f}" for r in day.itertuples())
        print(f"{date:%Y-%m-%d} | {budgets}")
        changed = day[day["change_pct"].abs() > 1e-9]
        shown = changed if not changed.empty else day.head(1)
        for r in shown.itertuples():
            print(f"    - {r.reason}")


def print_evaluation(base_hist, agent_hist, agent_log):
    b, a = summarize(base_hist), summarize(agent_hist)
    print("\n=== EVALUATION: equal split vs agent ===")
    print(f"{'Metric':<14}{'Equal split':>14}{'Agent':>14}")
    for label, key, fmt in [("Spend ($)", "spend", ",.0f"), ("Clicks", "clicks", ",.0f"),
                            ("Conversions", "conversions", ",.1f"), ("CTR", "ctr", ".2%"),
                            ("CVR", "cvr", ".2%"), ("CPA ($)", "cpa", ".2f")]:
        print(f"{label:<14}{format(b[key], fmt):>14}{format(a[key], fmt):>14}")

    lift = (a["conversions"] - b["conversions"]) / b["conversions"]
    cpa_change = (a["cpa"] - b["cpa"]) / b["cpa"]
    print(f"\nAgent vs baseline: {lift:+.1%} conversions, {cpa_change:+.1%} CPA "
          f"(same total spend).")

    print("\nAverage daily budget by channel (agent):")
    for channel, avg in agent_log.groupby("channel")["new_budget"].mean().items():
        print(f"  {channel:<8} ${avg:.2f}   (baseline ${TOTAL_DAILY_BUDGET / len(CHANNELS):.2f})")


# MAIN
def main():
    parser = argparse.ArgumentParser(description="Ad Optimization Agent")
    parser.add_argument("--csv", default=str(CSV_PATH), help="path to the data CSV")
    parser.add_argument("--regenerate", action="store_true", help="overwrite the CSV with new mock data")
    parser.add_argument("--days", type=int, default=21, help="days of mock data (14-30)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for mock data")
    parser.add_argument("--show-prompt", action="store_true", help="print the agent's system prompt")
    args = parser.parse_args()

    if args.show_prompt:
        print(SYSTEM_PROMPT + "\n")

    csv_path = Path(args.csv)
    if args.regenerate or not csv_path.exists():
        generate_mock_data(args.days, args.seed).to_csv(csv_path, index=False)
        print(f"Created mock data: {csv_path}")

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows: {df['date'].nunique()} days x {df['channel'].nunique()} channels")

    base_hist, _ = run_policy(df, equal_split_policy)
    agent_hist, agent_log = run_policy(df, agent_policy)

    agent_log.to_csv(LOG_PATH, index=False)
    print_decision_log(agent_log)
    print_evaluation(base_hist, agent_hist, agent_log)
    print(f"\nFull decision log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()