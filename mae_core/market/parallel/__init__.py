"""mae_core.market.parallel — Independent background analysis processes.

These processes run in their own OS processes, completely separate from
the daemon's event loop. They read from shared archives and write to their
own output files. No EventBus, no bootstrap, no interference.

Modules:
  continuous_postmortem  — Continuously analyzes graded outcomes to find
                           what domain combos win/lose and why. Writes to
                           data/midge/postmortem_continuous.json.
  continuous_replay      — Continuously replays archived signals through the
                           convergence engine with current Thompson weights.
  launch                 — Multiprocessing launcher for the replay process.
"""
