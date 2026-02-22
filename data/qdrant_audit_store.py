"""Temporary script to store audit findings to Qdrant. Delete after use."""
import json
import os
import tempfile
import subprocess
import sys

chunks_data = [
    {
        "title": "Mae signal path audit synthesis",
        "content": "Mae 10-agent audit (2026-02-11, ~50 sub-agents) assessed full signal path. Overall health 5.2/10. Key findings: (1) ACT is a 2-line stub returning 0.0, no environmental effect (1/10). (2) Learning does not learn - _learn_from_batch() updates no weights. (3) Mae is reactive, not predictive - no PREDICT step before _observe(). (4) Many systems wired but not connected - CuriosityDrive, ValidatedImagination, PatternDistiller, CollectiveDream outputs never consumed. (5) Fractal self-similarity violated - each scale uses different architecture. Step scores: SENSE 6/10, ADVISE 6/10, DECIDE 5/10, ACT 1/10, LEARN 4.7/10, CONSOLIDATE 6/10, RECALL 5/10, HEAL 5/10, SELF-AWARENESS 4/10. 20 bugs found. 44 upgrade recommendations across 3 tiers.",
        "tags": ["mae", "audit", "synthesis", "2026-02-11"]
    },
    {
        "title": "Mae audit bugs found",
        "content": "CRITICAL BUGS: (1) _act() stub returns 0.0 (base_agent.py:92). (2) _learn_from_batch() updates no weights (episodic_memory.py:147). (3) MemoryConsolidator calls nonexistent get/set_learning_rate (memory_consolidator.py:101). (4) Semantic recall returns SemanticQuery, code treats as list (mycelial_agent.py:326). (5) _route_with_advisory passes available_actions=None (mycelial_agent.py:384). HIGH BUGS: (6) set_reflex_bias() missing on DecisionRouter. (7) Adrenaline re-check identical to first. (8) PatternBus mutates shared signals in-place. (9) Z-score self-inclusion bias. (10) PatternDistiller injected never called. (11) Agent parent_id init None.",
        "tags": ["mae", "audit", "bugs", "2026-02-11"]
    },
    {
        "title": "Mae audit priority roadmap",
        "content": "Tier 1 Critical (9 items): Implement action environment + _act() override (HIGH effort, TRANSFORMATIVE impact). Add PREDICT step before _observe() (MEDIUM, satisfies FEP). Fix _learn_from_batch() to update weights (MEDIUM). Fix MemoryConsolidator interface (LOW). Fix endocrine-router wiring (LOW). Fix semantic recall type bug (LOW). Pass available_actions to advisory routing (LOW). Fix signal mutation bug (LOW). Fix z-score self-inclusion (LOW). Tier 2 High Impact (15 items): Wire CuriosityDrive to reward, ValidatedImagination to WorldModel, PatternDistiller into consolidator, AwarenessPulse to AutoHealer, Advisory to Endocrine, Endocrine to SignalPriority. Implement GWT ignition, TRN attentional gating, habituation. Wire self-awareness into lifecycle. Tier 3 Architectural (20 items): Make sensing/decision fractal, COMPARE step, goal management, Go/NoGo, reconsolidation, meta-healing triad, DEVELOP step.",
        "tags": ["mae", "audit", "roadmap", "2026-02-11"]
    },
    {
        "title": "Mae revised autopoietic loop",
        "content": "Current 7-step loop: Detect Advise Decide Act Learn Consolidate Recall. Proposed 13-step loop: (1) PREDICT - WorldModel generates expected next state BEFORE sensing. (2) ATTEND - Top-down attention gates signal processing. (3) SENSE - Observe environment. (4) COMPARE - Compute prediction error, the FEP surprise signal. (5) ADVISE - PatternBus to Cortex to Advisory, triggers hormonal response. (6) SELECT - Goal-directed selection with GWT competition. (7) INHIBIT/ACT - Go/no-go gate, triadic plan/execute/verify. (8) LEARN - Store experience, prediction error as reward, update weights. (9) REGULATE - Endocrine response, active parasympathetic recovery. (10) COMMUNICATE - GNN, stigmergy, pattern sharing. (11) CONSOLIDATE - Memory consolidation plus distillation. (12) RECALL - Ancestral pattern recall. (13) DEVELOP - Slow timescale: apoptosis, re-differentiation, reorganization.",
        "tags": ["mae", "audit", "autopoietic-loop", "2026-02-11"]
    },
    {
        "title": "Mae audit cross-cutting themes",
        "content": "Five systemic patterns across all audit reports: (1) REACTIVE NOT PREDICTIVE: FEP compliance failure. WorldModel never called before _observe(). No prediction error signal. Single largest gap. (2) WIRED BUT NOT CONNECTED: CuriosityDrive, ValidatedImagination, PatternDistiller, CollectiveDream, EndocrineSystem, AwarenessPulse, peer recall - all bootstrapped but outputs never consumed. (3) NOT FRACTAL: Most violated principle. Each scale uses different architecture. Sensing, deciding, healing, consolidation all different at each level. (4) NO GWT COMPETITION: Every signal reaches cortex. Every digest produces advisory. No ignition threshold. (5) SELF-AWARENESS DOES NOT PARTICIPATE: holon_know methods exist on every holon but never called during step lifecycle.",
        "tags": ["mae", "audit", "themes", "2026-02-11"]
    },
    {
        "title": "Mae mathematical identity compliance gaps",
        "content": "Mathematical identity compliance: 9/55 criteria fully compliant (16%), 19 partial (35%), 27 non-compliant (49%). Most violated: (1) Fractal self-similarity in 8/10 capabilities. (2) Prediction/FEP in 6/10. (3) Triadic structure in 5/10. (4) GWT competition in 4/10. (5) Self-produced boundaries in 4/10. Worst capabilities: ACT (0/6 compliant), KNOW UP (0/3), KNOW PEERS (1/4). Best: SENSE (3/10 full). Overall: WEAK compliance.",
        "tags": ["mae", "audit", "mathematical-identity", "2026-02-11"]
    },
    {
        "title": "Mae biological accuracy gaps",
        "content": "70 biological mechanisms assessed. 43% have no Mae analog (ABSENT). Only 17% rated GOOD/HIGH. Top 5 missing: (1) Predictive coding - brain predicts before sensing. (2) Dopamine RPE - most important learning signal absent from _learn(). (3) Thalamic Reticular Nucleus - suppresses 80% of signals. (4) Efference copy - predicts action consequences. (5) Habituation - suppresses repeated stimuli. Other major gaps: lateral inhibition, Go/NoGo, autophagy, reconsolidation, pattern completion, spreading activation, allostasis, nociception, theory of mind, default mode network.",
        "tags": ["mae", "audit", "biology", "2026-02-11"]
    },
    {
        "title": "Mae disconnected systems",
        "content": "Systems wired in main.py but outputs never consumed: (1) CuriosityDrive intrinsic reward never added to agent reward. (2) ValidatedImagination cognition.imagination_validated has zero subscribers. (3) PatternDistiller injected at main.py:950 but never called. (4) CollectiveDream cognition.collective_dream_complete has zero subscribers. (5) EndocrineSystem calls dr.set_reflex_bias() which does not exist. (6) AwarenessPulse holon.anomaly_detected has zero subscribers. (7) MemoryBridge.recall_peer_experiences() never called from decision path. (8) EventBus channels with no subscribers: healing.started, healing.complete, healing.failed, haven.intervention, holon.awareness_pulse, somatic.modification_rolled_back, defense.threat_detected, cognition.model_trained.",
        "tags": ["mae", "audit", "disconnected", "2026-02-11"]
    }
]

llm_output = {
    "meta": {
        "topic": "Mae Signal Path Audit 2026-02-11",
        "perspective": "Full 10-agent audit synthesis",
        "categories": "mae,audit,architecture,bugs,biology,compliance"
    },
    "summary": "Comprehensive 10-agent audit of Mae signal path. 50 sub-agents. Overall health 5.2/10. 20 bugs found. 44 upgrade recommendations.",
    "chunks": chunks_data
}

tmpfile = os.path.join(tempfile.gettempdir(), "mae_audit_qdrant.json")
with open(tmpfile, "w", encoding="utf-8") as f:
    json.dump(llm_output, f, indent=2)

print(f"Wrote {len(chunks_data)} chunks to {tmpfile}")

# Now run the store script
script = r"C:\Users\baenb\.claude\scripts\qdrant-store-llm.py"
result = subprocess.run(
    [sys.executable, script, "--hybrid", "--session", "mae-audit-2026-02-11", "-i", tmpfile],
    capture_output=True, text=True, timeout=120
)
print("STDOUT:", result.stdout[:2000])
if result.stderr:
    print("STDERR:", result.stderr[:2000])
print("Return code:", result.returncode)
