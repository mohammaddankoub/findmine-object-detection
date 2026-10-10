# Dataset notes

The final detection dataset contains 280 labelled images. Images, labels, owner
manifests, and model checkpoints are not stored in this public repository.

| Split | Images | Objects | Background-only images |
|---|---:|---:|---:|
| Train | 208 | 244 | 26 |
| Validation | 40 | 52 | 1 |
| Test | 32 | 50 | 1 |

Near-duplicate views were grouped by capture batch and scene identifier before splitting.
The 49 training-only support images were excluded from detector validation and test.

Owner-recognition datasets were created from object crops. Closely related views were
kept in the same split, and each category included a supervised `Unknown` class. These
crops are excluded because they reveal participant identities and registered belongings.

The external wallet images used during training were distributed under CC BY 4.0 by the
original dataset publisher. The external tumbler collection may have restricted access;
neither source is redistributed here.
