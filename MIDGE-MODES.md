# MIDGE Operating Modes

Pick a mode based on what else you're doing on Wardenclyffe.

## Mode 1: SLEEP (0% CPU)
**When:** You need the machine for heavy work (video editing, other projects)
```bash
powershell -Command "Stop-Process -Name python -Force -ErrorAction SilentlyContinue"
```
MIDGE is completely off. No processes. No CPU. No memory.

## Mode 2: WHISPER (~5% CPU, ~2GB RAM)
**When:** You're multitasking and need the machine responsive
```bash
python main.py --daemon --agents 3 --steps 100 --pace 5.0
```
Just the daemon, slow pace (5 seconds between steps), 3 agents. She senses and watches but doesn't dig. No research team, no sprint, no parallel workers.

## Mode 3: WALK (~20% CPU, ~8GB RAM)
**When:** Light work, browsing, documents
```bash
python -m mae_core.market.ecosystem.supervisor --only core --only analysis
```
Daemon + analysis processes (granger, postmortem, cross-market, grader). No research team, no sprint, no mining. She senses, learns from outcomes, and discovers causal relationships.

## Mode 4: RUN (~50% CPU, ~15GB RAM)
**When:** You're away for a while but might come back
```bash
python -m mae_core.market.ecosystem.supervisor
```
Full ecosystem: daemon + all 10 registered processes. Research team with 5 workers. No crypto sprint. Good balance of learning and resource usage.

## Mode 5: SPRINT (~90% CPU, ~30GB RAM)
**When:** You're sleeping or away for hours. Let MIDGE learn as fast as possible.
```bash
python -m mae_core.market.ecosystem.supervisor &
python -m mae_core.market.parallel.crypto_education_sprint &
```
Everything + the 8-worker crypto education sprint. Maximum learning. Maximum resource usage. Run overnight or when you leave for work.

## Quick Stop (any mode → sleep)
```
powershell -Command "Stop-Process -Name python -Force"
```

## Quick Check
```
python -m mae_core.market.ecosystem.supervisor --status
```
