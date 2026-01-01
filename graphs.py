import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# Paths
model_path = "model/alzheimers_model.h5"
dataset_path = "AugmentedAlzheimerDataset"
graph_folder = os.path.join("static", "graphs")

os.makedirs(graph_folder, exist_ok=True)

# Load model
model = load_model(model_path)

# Data generator for evaluation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

test_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Evaluate model
loss, acc = model.evaluate(test_data, verbose=0)
print(f"Validation Accuracy: {acc*100:.2f}%")

# Predictions
y_pred = model.predict(test_data)
y_true = test_data.classes
y_pred_classes = np.argmax(y_pred, axis=1)
class_labels = list(test_data.class_indices.keys())

# 1️⃣ Accuracy & Loss Graph
history_acc = acc * np.ones(10)
history_loss = loss * np.ones(10)
plt.figure(figsize=(6,4))
plt.plot(history_acc, label='Accuracy')
plt.plot(history_loss, label='Loss')
plt.title('Model Accuracy vs Loss')
plt.xlabel('Epochs (simulated)')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(graph_folder, "accuracy_loss.png"))
plt.close()

# 2️⃣ Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(graph_folder, "confusion_matrix.png"))
plt.close()

# 3️⃣ ROC Curve (macro-average)
plt.figure(figsize=(6,5))
for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(os.path.join(graph_folder, "roc_curve.png"))
plt.close()

# 4️⃣ Class Distribution (Actual vs Predicted)
plt.figure(figsize=(6,5))
actual_counts = np.bincount(y_true)
pred_counts = np.bincount(y_pred_classes)
x = np.arange(len(class_labels))
width = 0.35
plt.bar(x - width/2, actual_counts, width, label='Actual')
plt.bar(x + width/2, pred_counts, width, label='Predicted')
plt.xticks(x, class_labels, rotation=20)
plt.title('Actual vs Predicted Class Distribution')
plt.legend()
plt.savefig(os.path.join(graph_folder, "class_distribution.png"))
plt.close()

print("✅ All graphs saved in:", graph_folder)
