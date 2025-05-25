import os
import pandas as pd
import random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def limit_data(data_dir, categories, n=None):
    """
    This function limits the number of samples per class (category) and returns a DataFrame.

    Args:
        data_dir (str): Base directory containing class subfolders.
        categories (list): List of class folder names (e.g., ['Bird', 'Drone']).
        n (int or None): Max number of samples per class (None = use all).

    Returns:
        pd.DataFrame: DataFrame with columns ['path', 'label']
    """
    data = []
    for class_name in categories:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[!] Skipping non-directory: {class_dir}")
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        for i, filename in enumerate(image_files):
            if n is not None and i >= n:
                break
            img_path = os.path.join(class_dir, filename)
            data.append((img_path, class_name))

    return pd.DataFrame(data, columns=["path", "label"])

def evaluate(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = outputs.cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Compute ROC AUC
    auc = roc_auc_score(all_labels, all_probs)
    print(f"AUC: {auc:.4f}")

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

    # Compute confusion matrix (using 0.5 threshold)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    print(f"all_labels type: {type(all_labels)}, length: {len(all_labels)}")
    print(f"preds type: {type(preds)}, length: {len(preds)}")
    y_true = np.array([label.item() if hasattr(label, 'item') else label for label in all_labels])
    y_pred = np.array([pred.item() if hasattr(pred, 'item') else pred for pred in preds])
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Drone', 'Bird'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def extract_features(dataloader, feature_extractor):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs)
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            features.append(feats)
            labels.append(lbls.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def ROC_curve(models, X_train, y_train, X_test, y_test):
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
    
    print(f"{name} AUC: {auc:.4f}")
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Linear Models')
    plt.legend()
    plt.show()

def confusion_matrix_linear(models,X_test,y_test):
   for model_name, clf in models.items():
       y_pred = clf.predict(X_test)
       cm = confusion_matrix(y_test, y_pred)
       disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Drone', 'Bird'])
       disp.plot(cmap=plt.cm.Blues)
       plt.title(f'Confusion Matrix: {model_name}')
       plt.show()