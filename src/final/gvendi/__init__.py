"""H-Vendi-guided synthetic client expansion for the ReLoRA curriculum.

This package implements gradient-entropy-guided selection of the clients added
at a rectangular curriculum boundary (first target: 7x7 -> 8x8). It lives
alongside ``curriculum`` and imports its helpers rather than modifying them.

Modules
-------
config       GVendiConfig (extends ReLoRAConfig) + validation
grad_vendi   projection, entropy / H-Vendi, k-means shortlist, selection rules
probing      adapter-gradient probes, proxy metrics, learnability test
data         manifest-driven synthetic client construction
candidates   candidate pool generation + quality gate
manifest     atomic persistence, validation, resume detection
train_8x8    stage training loop with new-client sampling boost + old/new metrics
run_gvendi_8x8  end-to-end orchestrator
"""
