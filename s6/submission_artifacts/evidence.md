# Evidence Bundle

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | manifests/shard-*.json (sha256 + tokenizer_hash), re-verified against shards/*.bin |
| Evaluation firewall | PASS | run.log [PASS] eval_shard_blocked; ledgers/consumption_ledger.jsonl |
| Packing correctness | PASS | ledgers/consumption_ledger.jsonl segments, reassembled via packing.rebuild_batch |
| Mixture compliance | PASS | manifests/mixture_schedule.json vs. accepted rows in ledgers/consumption_ledger.jsonl |
| OPUS audit trail | PASS | ledgers/opus_audit.jsonl (+ ledgers/fork/opus_audit.jsonl) |
| Crash recovery | PASS | ledgers/resume_report.json; run.log [PASS] resume_next_batch_matched / checkpoint_saved |
| Replay | PASS | ledgers/replay_report.json; run.log [PASS] replay_hash_matched |
| Learning trace | PASS | ledgers/learning_ledger.jsonl joined to ledgers/consumption_ledger.jsonl on batch_id |
| Throughput | PASS | performance.json; recomputed from ledgers/consumption_ledger.jsonl |

**Overall: PASS**

## Detail
- **Tokenizer integrity**: 4 shard manifests hash-verified against the live tokenizer
- **Evaluation firewall**: 0 eval-lane rows in consumption ledger; block event logged=True
- **Packing correctness**: all 51 ledger batches (main + fork) structurally rebuilt from segments and mask-checked
- **Mixture compliance**: protected floor never breached in any stage
- **OPUS audit trail**: 51 candidate decisions logged (one per consumption-ledger row); distribution={'ACCEPT': 28, 'PROTECTED_FLOOR_OVERRIDE': 6, 'DEFER': 1, 'REJECT': 16}
- **Crash recovery**: 4 post-crash batches independently compared against the pre-crash expected batch ids/hashes, all_match=True
- **Replay**: 4 ledger records replayed, all_match=True
- **Learning trace**: 31 learning-ledger rows all trace back to a consumption-ledger batch id
- **Throughput**: performance.json packing_utilization=0.9575, independently recomputed from the ledger=0.9575
