# Experiment log

## v2: grouped Batch 02

The first useful baseline used YOLO11s at 640 px for 30 epochs. Batch 02 was split
by capture session so that near-identical views could not cross between training,
validation, and test sets.

## v4: public training supplement

The v2 checkpoint was fine-tuned after adding product-style training images. The
best validation mAP50 improved, but the validation set remained small. These public
images are not included in this repository because their reuse licences have not
all been verified.

## v5: own-image fine-tuning

The v4 checkpoint was fine-tuned again on the captured FindMine scenes. The first
epoch produced the highest validation mAP50 in this run. Later epochs reduced
training loss while validation performance fluctuated, which suggests that the
small validation set is not stable enough for fine model selection.

## Current work

- Add difficult wallet examples with clutter, rotation, small scale, and occlusion.
- Rebuild the dataset using scene-grouped 70/15/15 splits.
- Remove or document every public image before using it in a published dataset.
- Run a final test-set evaluation only after model and confidence threshold are fixed.
- Review false positives on upholstery, cushions, and rectangular background objects.

