# stim/gate.py

## Plan

- [x] Read `interface/roboroach.py`, report `configure()` and `StimulationSettings`
- [x] Resolve the two spec conflicts: `configure()` returns `None`, and it is
      async while construction is not
- [x] `StimGate` with required roach, `t_refrac_s`, settings, run dir
- [x] Refractory check, Bluetooth write, and `n` increment under one lock
- [x] JSONL line per request, accepted or rejected
- [x] `sources.json` merged, not overwritten
- [x] Test suite with an injected stub, no hardware
- [x] `ruff format . ; ruff check --fix . ; pytest -q`

## Review

`StimGate` is the only object that calls `RoboRoach.turn`. It holds the
refractory period and the trial counter `n`, so the stimulation budget is
enforced once instead of once per caller.

Two things the spec did not name but that the code needs:

- `configure()` returns `None`, so the board settings come from a
  `read_settings()` readback. The readback is what will actually fire, so the
  refractory assertion is checked against it as well as against the request.
- A failed Bluetooth write may already have been delivered. It holds the
  refractory window shut, is logged as `write_failed`, does not increment `n`,
  and re-raises. `interface/AGENTS.md` requires visible failure and no retry.

`interface/` is excluded from ruff in `pyproject.toml`. Running `ruff format .`
without that would rewrite `roboroach.py`, including `turn()`.

Open: `T_refrac` still has to be set from the habituation budget, and the
shared consumer that the voice module emits into does not exist in the
repository yet.
