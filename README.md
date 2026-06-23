# DDSH-VoxelNeXt

**Demand-Driven Sparse Hybrid VoxelNeXt for Cooperative 3D Object Detection**

This repository provides **DDSH-VoxelNeXt**, a fully sparse cooperative 3D object detection framework built on top of OpenCOOD and VoxelNeXt.

DDSH-VoxelNeXt keeps the whole cooperative intermediate-fusion pipeline sparse. It communicates and fuses **sparse BEV tokens** instead of dense BEV feature maps.

```text
points
-> voxelization
-> MeanVFE
-> VoxelNeXt sparse 3D backbone
-> encoded_spconv_tensor
-> SparseHeightCompression
-> sparse BEV tokens
-> demand/supply sparse communication
-> sparse token fusion
-> VoxelNeXt sparse detection head
-> final predictions
```

## Key Constraint

The DDSH main path does **not** use dense BEV intermediate features:

```text
No encoded_spconv_tensor.dense()
No [B, C, H, W] dense BEV feature
No BaseBEVBackbone
No dense HeightCompression
No spatial_features_2d
```

## Highlights

- Fully sparse VoxelNeXt-based cooperative detection.
- Sparse BEV token communication between CAVs.
- Ego demand tokens for demand-driven collaboration.
- Helper supply token selection after helper-to-ego pose alignment.
- Sparse coordinate fusion and optional local sparse attention.
- Six YAML-controlled stages for ablation.
- Training and validation statistics for communication bytes, token counts, memory, and latency.
- Paper visualization support for sparse tokens, demand/supply, alignment, matching, and communication.

## Method

For each CAV:

```text
points -> voxelization -> VoxelNeXt sparse backbone
-> sparse 3D voxel features -> sparse BEV tokens
```

Ego:

```text
ego sparse BEV tokens -> demand generator -> demand tokens
```

Helper:

```text
helper sparse BEV tokens
-> pose align to ego frame
-> supply selector
-> selected supply tokens
```

Fusion:

```text
ego tokens + aligned helper supply tokens
-> sparse token fusion
-> optional local sparse attention
-> fused sparse tokens
```

Detection:

```text
fused sparse tokens -> SparseConvTensor -> sparse detection head
```

## Main Files

```text
opencood/models/ddsh_voxelnext.py
opencood/models/sub_modules/ddsh/
opencood/tools/debug_ddsh_forward.py
opencood/tools/profile_ddsh_statistics.py
opencood/tools/ddsh_stats_utils.py
opencood/visualization/ddsh_paper_vis.py
scripts/run_ddsh_all_stages.sh
README_DDSH_VoxelNeXt.md
```

## YAML Stages

| Stage | Name | Description |
|---:|---|---|
| 0 | `sparse_single` | Ego-only sparse baseline |
| 1 | `sparse_all_token` | All helper sparse tokens |
| 2 | `sparse_topk` | Helper quality top-K sparse tokens |
| 3 | `demand_supply` | Ego demand + helper supply selection |
| 4 | `sparse_attention` | Local sparse attention fusion |
| 5 | `hybrid_late` | Sparse intermediate fusion plus optional late compensation |

V2V4Real configs:

```text
opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage0_sparse_single.yaml
opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage1_sparse_all_token.yaml
opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage2_sparse_topk.yaml
opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage3_demand_supply.yaml
opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage4_sparse_attention.yaml
opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage5_hybrid_late.yaml
```

OPV2V configs:

```text
opencood/hypes_yaml/ddsh_voxelnext_stage0_sparse_single.yaml
opencood/hypes_yaml/ddsh_voxelnext_stage1_sparse_all_token.yaml
opencood/hypes_yaml/ddsh_voxelnext_stage2_sparse_topk.yaml
opencood/hypes_yaml/ddsh_voxelnext_stage3_demand_supply.yaml
opencood/hypes_yaml/ddsh_voxelnext_stage4_sparse_attention.yaml
opencood/hypes_yaml/ddsh_voxelnext_stage5_hybrid_late.yaml
```

## Quick Start

### Environment

Use the OpenCOOD/VoxelNeXt environment with PyTorch, CUDA, spconv, and cumm.

```bash
conda activate opencood
```

### Debug One Batch

```bash
python3 opencood/tools/debug_ddsh_forward.py \
  --hypes_yaml opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage3_demand_supply.yaml
```

### Train All Stages

Recommended V2V4Real order:

```text
Stage0 -> Stage3 -> Stage1 -> Stage2 -> Stage4 -> Stage5
```

Run all stages:

```bash
bash scripts/run_ddsh_all_stages.sh
```

Run in background:

```bash
nohup bash scripts/run_ddsh_all_stages.sh > ddsh_all_stages.out 2>&1 &
tail -f ddsh_all_stages.out
```

Fast smoke test:

```bash
bash scripts/run_ddsh_all_stages.sh --profile-max-batches 100
```

### Train One Stage

```bash
python3 opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage3_demand_supply.yaml
```

## Evaluation

```bash
python3 opencood/tools/inference.py \
  --model_dir opencood/logs/<RUN_DIR> \
  --fusion_method intermediate \
  --global_sort_detections
```

Outputs:

```text
opencood/logs/<RUN_DIR>/eval.yaml
opencood/logs/<RUN_DIR>/eval_global_sort.yaml
```

## Communication Statistics

```bash
python3 opencood/tools/profile_ddsh_statistics.py \
  --model_dir opencood/logs/<RUN_DIR> \
  --split val \
  --batch_size 1 \
  --num_workers 0 \
  --max_batches -1
```

Outputs:

```text
opencood/logs/<RUN_DIR>/ddsh_statistics/ddsh_statistics_val_detail.csv
opencood/logs/<RUN_DIR>/ddsh_statistics/ddsh_statistics_val_summary.csv
opencood/logs/<RUN_DIR>/ddsh_statistics/ddsh_statistics_val_summary.json
```

Metrics include:

```text
demand_bytes
supply_bytes
total_bytes
num_demand_tokens
num_supply_tokens
num_fused_tokens
fusion_match_token_count
elapsed_ms
peak_memory_mb
```

## More Documentation

See the detailed runbook:

```text
README_DDSH_VoxelNeXt.md
```

Paper notes and tables:

```text
paper/ddsh_voxelnext_zh.md
paper/ddsh_voxelnext_experiment_tables.md
paper/ddsh_voxelnext_ablation_plan.md
paper/source_code_mapping.md
paper/todo_missing_results.md
```

## Results

No pretrained checkpoints or benchmark numbers are included in this README. Please run the provided stage YAML files and fill in your own result tables.

## Citation

```bibtex
@misc{ddsh_voxelnext,
  title  = {DDSH-VoxelNeXt: Demand-Driven Sparse Hybrid VoxelNeXt for Cooperative 3D Detection},
  author = {TODO},
  year   = {2026},
  note   = {Fully sparse cooperative 3D detection based on OpenCOOD and VoxelNeXt}
}
```

## License

This repository builds on OpenCOOD and VoxelNeXt-related components. Please follow the licenses of OpenCOOD and all third-party dependencies.
