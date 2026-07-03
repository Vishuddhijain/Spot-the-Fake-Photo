# Spot the Fake Photo

Classifies an image as a REAL photo or a screen/printout RECAPTURE.
Classical computer vision (no deep learning) — 6 hand-crafted features
feeding a small logistic regression.

## Quick start

```bash
pip install -r requirements.txt
python predict.py path/to/image.jpg
# -> prints a float in [0,1]:  0 = real, 1 = screen/recapture
```

## Files

| File           | Purpose                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------ |
| `features.py`  | 6 hand-crafted CV features (moiré/FFT, high-freq energy, glare, bezel edges, banding, sharpness) |
| `train.py`     | Fits logistic regression on `real/` + `screen/` folders, 5-fold CV, saves `weights.json`         |
| `predict.py`   | One-line predictor. Uses `weights.json` if present, else a hand-tuned fallback                   |
| `weights.json` | Trained model (7 floats — mean/std/coef/intercept)                                               |
| `index.html`   | Live camera demo, runs entirely client-side                                                      |

## Results

- **Accuracy:** 76.2% (5-fold cross-validated on 80 photos: 30 real, 50 screen)
- **Latency:** ~46 ms/image (laptop CPU)
- **Cost:** $0/image on-device; ~$0.03–0.05 per 1,000 images on cloud CPU fallback

```bash
python train.py --real_dir real/ --screen_dir screen/ --out weights.json
```
