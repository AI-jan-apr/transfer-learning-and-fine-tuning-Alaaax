# Food-11 Image Classification using EfficientNet (Transfer Learning)

## Project Overview
This project applies **Transfer Learning** using **EfficientNetB0** to classify food images from the **Food-11 dataset**.
Two main approaches were tested:

- Feature Extraction
- Fine-Tuning

The goal is to compare these techniques and evaluate how fine‑tuning pretrained deep learning models affects performance.

---

## Dataset

Dataset: **Food‑11 Image Dataset**

Classes:

- Bread
- Dairy Product
- Dessert
- Egg
- Fried Food
- Meat
- Noodles / Pasta
- Rice
- Seafood
- Soup
- Vegetable / Fruit

Dataset split:

- Training
- Validation
- Test

---

## Preprocessing

Steps applied:

- Images resized to **224 × 224**
- Labels converted from **text → integer**
- Data loaded using **TensorFlow tf.data pipeline**
- Batch size = **32**

Important issue discovered:

At first, an additional image scaling step (`Rescaling(1./255)`) was applied.  
However, **EfficientNet already includes internal preprocessing**, so this caused a mismatch and reduced performance.

After removing the extra scaling step, model accuracy improved significantly.

---

## Experiment 1 — Feature Extraction

EfficientNetB0 was used as a **feature extractor**.

All pretrained layers were frozen and only the classification head was trained.

Technique:
```
base_model.trainable = False
```

### Model Architecture

EfficientNetB0 (ImageNet pretrained)
-> GlobalAveragePooling
-> Dropout
-> Dense (11 classes – Softmax)

### Results

Test Accuracy: **0.9116**  
Test Loss: **0.2753**

Observation:

- Training accuracy reached around **0.91**
- Validation accuracy stabilized around **0.88**
- Training and validation curves were close, indicating good generalization

---

## Experiment 2 — Fine‑Tuning

Technique used:

### Unfreeze only last n layers

```
for layer in base_model.layers[:-20]:
    layer.trainable = False
```

Reason for choosing this technique:

- Allows higher‑level features to adapt to the new dataset
- Preserves general visual features learned from ImageNet
- Reduces risk of destroying pretrained representations
- Common and stable transfer learning strategy

A smaller learning rate was used:

```
learning_rate = 1e-6
```

### Fine‑Tuning Results

Test Accuracy: **0.8832**  
Test Loss: **0.3704**

---

## Comparison

| Approach | Accuracy | Loss |
|---------|---------|------|
| Feature Extraction | **0.9116** | **0.2753** |
| Fine‑Tuning | **0.8832** | **0.3704** |

Observation:

Feature Extraction achieved better performance in this experiment.
This can happen when pretrained features from ImageNet are already highly suitable for the dataset.

Fine‑tuning slightly modified these representations and did not improve the final accuracy.

---


## Challenges

1. **Preprocessing mismatch**

An extra image scaling step (`1./255`) was initially applied before EfficientNet.
Since EfficientNet already includes preprocessing layers, this caused poor performance.

Removing this step significantly improved accuracy.

2. **Fine‑Tuning sensitivity**

Fine‑tuning pretrained models is sensitive to:

- learning rate
- number of unfrozen layers
- training duration

Even after adjustments, fine‑tuning did not outperform feature extraction in this case.


3. **Learning rate sensitivity during fine-tuning**

During the fine-tuning stage, a learning rate of 1e-5 was initially used. This caused the pretrained weights to update too aggressively, which negatively affected performance. Reducing the learning rate helped stabilize the training process.

---

## Conclusion

Transfer learning with EfficientNetB0 proved highly effective for the Food‑11 classification task.

Key findings:

- Feature Extraction achieved **91% accuracy**
- Fine‑Tuning achieved **88% accuracy**
- Pretrained EfficientNet features generalized well to food images
- Proper preprocessing is critical when using pretrained models

---

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- Google Colab
- Matplotlib
- Transfer Learning


## 🔗 Helpful Links

- 📚 EfficientNet models in Keras:  
  https://keras.io/api/applications/efficientnet/

- 🎓 Transfer Learning guide (Keras):  
  https://keras.io/guides/transfer_learning/

- 📦 MLflow for experiment tracking:  
  https://www.mlflow.org/docs/latest/index.html

- ☁️ DVC + DagsHub integration:  
  https://dagshub.com/docs/integrations/dvc/

- 🧑‍🍳 How to freeze/unfreeze layers in Keras:  
  https://keras.io/getting_started/faq/#how-can-i-freeze-layers-in-a-model

- 📈 Using callbacks in Keras (e.g. EarlyStopping, ReduceLROnPlateau):  
  https://keras.io/api/callbacks/

