"""
Phase 2 inbox — simulated SU email inbox trigger.

Folder layout:
    inbox/pending/   — SU drops email bundles here (one sub-dir per shipment)
    inbox/done/      — successfully processed bundles are archived here
    inbox/failed/    — bundles that errored during processing land here
    inbox/results/   — one JSON result file per processed shipment
    inbox/samples/   — pre-built sample bundles for testing
"""
