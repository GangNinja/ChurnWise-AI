# MLP training report - phase13-final-live-mlp

- **Preprocessing run:** `phase13-final-live-prep`
- **Dataset:** `openml:46915`
- **Dataset SHA-256:** `87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6`
- **Split fingerprint:** `4b0c45db677fa9619de99b6a8b786a3da2c4da93824c46db65412a85829594dd`
- **Preprocessor hash:** `3c31f13c266c21084050bb021e60a41d4145a2055b983eb03e2267fab9f52174`
- **Created:** 2026-10-23T00:46:45Z
- **Code version:** 0.10.0

## Architecture

- Input features: 27
- Hidden layers: [32, 16] (ReLU)
- Dropout: [0.2, 0.1]
- L2: 0.0001
- Output: sigmoid probability of churn
- Parameters: 1441

## Training configuration

- Optimizer: `adam`
- Learning rate: 0.001
- Loss: `binary_crossentropy`
- Batch size: 64
- Epoch limit: 24
- Early stopping: patience 5 on `val_loss`
- ReduceLROnPlateau: factor 0.5, patience 3, min lr 1e-05
- Class weights: `{'0': 0.5822981366459627, '1': 3.5377358490566038}`
- Training rows: 3000 (positive rate 0.14133333333333334)
- Validation rows: 1000 (positive rate 0.142)
- Test rows used: 0

## Training outcome

- Epochs completed: 24
- Best epoch: 24
- Best validation loss: 0.4042159914970398
- Early stopped: False
- Duration: 7.55 s
- Warnings: 2

## Learning curves

Learning curves (generated artifact: `./.phase13_final\artifacts\models\phase13-final-live-mlp\mlp\learning_curves.png`)

## Per-epoch history

| Epoch | loss | val_loss | accuracy | val_accuracy | learning_rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.82882 | 0.656384 | 0.654667 | 0.63 | 0.001 |
| 2 | 0.655826 | 0.616194 | 0.630333 | 0.671 | 0.001 |
| 3 | 0.606615 | 0.593299 | 0.66 | 0.673 | 0.001 |
| 4 | 0.597712 | 0.576546 | 0.677667 | 0.694 | 0.001 |
| 5 | 0.55533 | 0.551011 | 0.715333 | 0.71 | 0.001 |
| 6 | 0.540092 | 0.521196 | 0.731333 | 0.729 | 0.001 |
| 7 | 0.540368 | 0.521265 | 0.727667 | 0.732 | 0.001 |
| 8 | 0.513953 | 0.511254 | 0.752 | 0.742 | 0.001 |
| 9 | 0.503622 | 0.488114 | 0.764 | 0.769 | 0.001 |
| 10 | 0.485325 | 0.467257 | 0.781333 | 0.786 | 0.001 |
| 11 | 0.482621 | 0.467714 | 0.778 | 0.786 | 0.001 |
| 12 | 0.465821 | 0.463173 | 0.796 | 0.79 | 0.001 |
| 13 | 0.459139 | 0.448616 | 0.8 | 0.811 | 0.001 |
| 14 | 0.452013 | 0.444268 | 0.809667 | 0.814 | 0.001 |
| 15 | 0.447528 | 0.442957 | 0.818333 | 0.819 | 0.001 |
| 16 | 0.434546 | 0.434533 | 0.820333 | 0.825 | 0.001 |
| 17 | 0.436795 | 0.437574 | 0.830667 | 0.827 | 0.001 |
| 18 | 0.416892 | 0.428074 | 0.842 | 0.833 | 0.001 |
| 19 | 0.419434 | 0.412008 | 0.840667 | 0.847 | 0.001 |
| 20 | 0.422383 | 0.412171 | 0.847333 | 0.846 | 0.001 |
| 21 | 0.410586 | 0.408726 | 0.846333 | 0.845 | 0.001 |
| 22 | 0.408979 | 0.411637 | 0.851667 | 0.843 | 0.001 |
| 23 | 0.406837 | 0.408196 | 0.848 | 0.85 | 0.001 |
| 24 | 0.399329 | 0.404216 | 0.861667 | 0.853 | 0.001 |

## Warnings

- maximum epoch limit (24) reached; early stopping did not trigger
- TensorFlow op determinism is disabled; repeated runs may vary at floating-point precision

## Artifacts

- model: `./.phase13_final\artifacts\models\phase13-final-live-mlp\mlp\model.keras`
- metadata: `./.phase13_final\artifacts\models\phase13-final-live-mlp\mlp\metadata.json`
- history: `./.phase13_final\artifacts\models\phase13-final-live-mlp\mlp\history.json`
- curves: `./.phase13_final\artifacts\models\phase13-final-live-mlp\mlp\learning_curves.png`
- report_json: `.phase13_final\reports\models\phase13-final-live-mlp.json`
- report_md: `.phase13_final\reports\models\phase13-final-live-mlp.md`
- checkpoint: `./.phase13_final\artifacts\models\phase13-final-live-mlp\mlp\best.keras`
