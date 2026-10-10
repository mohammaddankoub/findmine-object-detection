# Experiment history

## Detector pilots

Early YOLO11s runs exposed missed wallets and false positives on sofas, cushions, and
other rectangular or textured background objects. Labels were corrected, difficult and
negative scenes were added, and the split was rebuilt by scene rather than by individual
image.

## Final detector

The final detector used a COCO-pretrained YOLO11s model at 640 x 640 resolution, batch
size 16, and up to 100 epochs with early-stopping patience 20. Ultralytics selected AdamW
automatically. Mosaic, HSV colour jitter, random scaling and translation, and horizontal
flips were enabled; MixUp and rotation were disabled.

The checkpoint and the final detection confidence of 0.88 were selected on validation
data before the test set was evaluated.

## Owner classifiers

ResNet18 and MobileNetV3-Small were initialized from ImageNet weights and trained on
224 x 224 crops for 35 epochs. The classification head was trained for five epochs while
the backbone was frozen; the full network was then fine-tuned. Class-weighted
cross-entropy addressed class imbalance.

ResNet18 produced the strongest validation macro-F1 for both wallets and travel mugs and
was therefore selected for end-to-end evaluation.

## Final pipeline

YOLO detections were matched to ground truth at IoU >= 0.5. The detected crops were sent
to the category-specific owner classifier, with predictions below 0.50 rejected as
`Unknown`.

Out of 50 ground-truth test objects, 39 were detected and 36 received the correct owner.
Eleven objects were missed, three matched objects had an owner error, and seven additional
detections were false positives. The resulting end-to-end F1 was 0.750.
