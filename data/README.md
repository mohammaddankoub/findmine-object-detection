# Dataset notes

The current internal dataset contains 231 images collected across multiple capture
sessions. Images and annotations are not stored in this repository.

| Split | Images | Background images | Wallet boxes | Travel mug boxes |
|---|---:|---:|---:|---:|
| Train | 194 | 3 | 125 | 146 |
| Validation | 18 | 1 | 8 | 10 |
| Test | 19 | 2 | 10 | 13 |

Frames from the same scene are visually very similar. Splitting individual images
at random would put near-duplicates in different sets and inflate validation scores.
The grouped split therefore assigns all views of a `(capture batch, scene number)`
pair to the same split.

Public product images explored during later experiments are excluded here. Their
source and reuse licence must be verified before they can be published or included
in a reproducible dataset release.

