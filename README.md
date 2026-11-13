# polymarket-nvidia-ceo-forecast-01 entry

A forecaster for Polymarket's "Next CEO of NVIDIA?" event. `main.py`
(Python 3.13, standard library only) reads one case from the input
directory — every market of the event with its Yes price at each
minute up to a checkpoint — and writes, for each market and each of
the 240 minutes after the checkpoint, five quantiles (5%, 25%, 50%,
75%, 95%) of that minute's price to the output directory.

## Model

Per market, estimated only from the prices in the case itself:

- **Median = persistence.** The last observed price. These markets
  move in steps and level shifts are close to unpredictable, so the
  last price is the honest centre.
- **Spread = a 50/50 blend** of two views of the market's own recent
  history (last 1440 minutes):
  - empirical quantiles of h-minute price changes, computed directly
    at anchor horizons 1, 5, 15, 30, 60, 120, 240 minutes and
    interpolated in sqrt(h) — this captures the heavy tails, the
    large mass of zero moves, and the mean reversion of one-minute
    spikes;
  - a Gaussian random walk whose per-minute sigma comes from the
    mean absolute one-minute change (robust to rare huge prints),
    floored at 0.0012 and capped at 0.004.
- **Volatility regime:** the last 6 hours of absolute movement are
  compared with the whole history; the spread is scaled by that
  ratio, clamped to [0.7, 1.6].
- A market with too little history of its own borrows the median
  sigma of the case's established markets.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. The program is deterministic and runs in well under a
second per case.

Backtested on the kit's practice inputs (later cases used as ground
truth for earlier checkpoints), this scores a mean pinball loss of
about 0.366 against the house baseline's 0.399 on the same folds.

## License

MIT — see `LICENSE`.
