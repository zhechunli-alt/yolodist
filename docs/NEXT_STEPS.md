# Next Steps

## Current priority

Build a reliable server-side training workflow for the model part of the thesis.

## Short-term plan

1. Prepare `datasets/processed/mvtec_detection/` with `tools/prepare_mvtec_detection.py`.
2. Put pretrained weights into `weights/`.
3. Run baseline training and confirm the end-to-end path works.
4. Train a student baseline in `experiments/modified_model/`.
5. Run pseudo-label distillation in `experiments/distillation/`.

## Medium-term plan

1. Replace the placeholder student with a true lightweight architecture.
2. Add evaluation, comparison tables, and ablation support.
3. Support a second industrial dataset such as NEU-DET.

## Deferred work

These are still aligned with the thesis, but not the first milestone:

- prompt-based synthetic data generation
- TensorRT deployment
- backend service
- frontend system
