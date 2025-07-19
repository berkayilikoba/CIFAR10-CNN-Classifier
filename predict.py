import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Sınıf isimleri
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Test verisini yükle ve normalize et
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0

# Gerçek etiketleri integer olarak al
y_test_int = y_test.flatten()

# Eğitilmiş modeli yükle
model = load_model("best_cifar10_model.h5")

# Tahmin yap
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Accuracy hesapla
acc = accuracy_score(y_test_int, y_pred_classes)
print(f"Test Accuracy: {acc:.4f}\n")

# Classification report
print("Classification Report:")
print(classification_report(y_test_int, y_pred_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test_int, y_pred_classes)

# Confusion matrix görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
plt.tight_layout()
plt.show()
