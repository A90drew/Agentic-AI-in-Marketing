# Ad Optimization Agent

A minimal agent that splits a fixed daily ad budget across Search, Social and
Display to maximize total conversions. Each day it learns from recent results,
shifts budget toward the most efficient channel within guardrails, and logs a
reason for every decision. It is compared against an equal-split baseline.

**Group members:** Andrew Rivas, Mohamed Ibrahim, Santos Sandoval, Yaqub Dayib, Noel Bustos, Matthew Throne

## How to run

```bash
pip install pandas numpy
python ad_agent.py
```

The first run creates `ad_data.csv` (21 days of mock data) and writes
`decision_log.csv`. Useful options:

- `python ad_agent.py --regenerate --days 30 --seed 7` makes new mock data
- `python ad_agent.py --csv your_data.csv` uses your own file
- `python ad_agent.py --show-prompt` prints the agent's system prompt

No API key is needed.

## Files

| File | What it is |
|---|---|
| `ad_agent.py` | Data generation, agent, simulation, evaluation |
| `ad_data.csv` | Mock data: `date, channel, spend, impressions, clicks, conversions` |
| `decision_log.csv` | One row per channel per day: old/new budget, % change, reason |

## How the agent works

1. **Day 1:** equal split, since there is no history.
2. **Each later day:** compute CTR, CVR and CPA per channel over the last 5 days.
3. **Rank** channels by conversions per dollar (1 / CPA), which combines CTR and CVR.
4. **Shift** 15% of the top channel's budget to it from the weakest channel that
   still has room to give. The other channel is left alone.
5. **Log** a reason string for every channel, every day.

### Guardrails

| Rule | Value |
|---|---|
| Max change per channel per day | +/-20% |
| Minimum budget per channel (never 0%) | 10% of total ($30) |
| Maximum budget per channel | 60% of total ($180) |
| Ignore tiny gaps (noise filter) | less than 10% efficiency gap means hold |
| Total always equals the daily budget | checked every day |

The minimum floor keeps every channel getting data, so the agent can notice if a
weak channel improves (explore/exploit).

## Assumptions

- The CSV is the "world." Each row shows what a channel did on a day at its
  historical spend. To test a different budget, the script replays that row,
  scaling impressions, clicks and conversions by (new budget / old spend)^0.9.
  The 0.9 exponent models diminishing returns. Set `RETURNS_EXPONENT = 1.0` for linear.
- The agent only sees results up to the previous day.
- Data is aggregated by channel. No personal or user-level data is used.
- Mock data is generated on purpose so that Social improves mid-period.

## Results snapshot

21 days, 3 channels, $300/day total ($6,300 total spend in both runs).

| Metric | Equal split | Agent |
|---|---|---|
| Spend | $6,300 | $6,300 |
| Clicks | 9,871 | 10,625 |
| Conversions | 385.0 | 432.4 |
| CTR | 1.08% | 1.49% |
| CVR | 3.90% | 4.07% |
| CPA | $16.36 | $14.57 |

The agent produced **+12.3% conversions and about 11% lower CPA** at the same spend.
(Conversions are fractional because the replay scales historical results by budget.)

**What happened**
- Oct 2-7: Search had the best conversions per dollar, so it gained 15% a day until it
  hit the $180 cap on Oct 7. Display was cut to $37.
- Oct 8-16: HOLD. Search was the best channel but capped, so the guardrail blocked
  further growth.
- Oct 17-21: Social's conversion rate had improved and it overtook Search in the
  5-day window. The agent moved budget from Search to Social (+15% a day) while
  Display sat at its $30 floor.
- Average daily budget: Search $160.77, Social $93.59, Display $45.64 (baseline $100 each).

**Trade-offs**
- The 5-day lookback smooths noise but reacts late: Social improved around Oct 7-12,
  and the first shift toward it came on Oct 17. A shorter window reacts faster but
  chases noise.
- The caps and floors limit upside (Search was held at $180 while it was best) but
  keep every channel generating data.
- When the top channel is capped, the agent holds instead of promoting the next-best channel.
- This is simulated data with a planted trend, so real results would be noisier.

## Limitations and next steps

- Mock data with a planted trend favors an adaptive agent; real data would be noisier.
- No carryover effects (e.g., a Display view leading to a later Search conversion).
- Next steps: Thompson sampling instead of fixed thresholds, per-channel
  diminishing-returns curves, a real ad-platform API.
