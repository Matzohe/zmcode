import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


def load_fashion_mnist(classes=[0,1,2,3,4], resize=(128,128)):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_indices = [i for i, label in enumerate(train_dataset.targets) if label in classes]
    test_indices = [i for i, label in enumerate(test_dataset.targets) if label in classes]
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=1000, shuffle=False)
    X_train, y_train = [], []
    for images, labels in train_loader:

        for img in images:
            img_np = img.numpy().squeeze() * 255
            img_resized = cv2.resize(img_np.astype(np.uint8), resize)
            X_train.append(img_resized)
        y_train.extend(labels.numpy())
    
    X_test, y_test = [], []
    for images, labels in test_loader:
        for img in images:
            img_np = img.numpy().squeeze() * 255
            img_resized = cv2.resize(img_np.astype(np.uint8), resize)
            X_test.append(img_resized)
        y_test.extend(labels.numpy())
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test


def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for idx, img in enumerate(images):
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
        else:
            descriptors_list.append(np.array([]))
        if idx % 1000 == 0:
            print(f"Processed {idx} / {len(images)} images for SIFT feature extraction")
    return descriptors_list


def build_vocabulary(descriptors_list, k=100):
    all_descriptors = [desc for desc in descriptors_list if desc.size != 0]
    all_descriptors = np.vstack(all_descriptors)
    print(f"Total descriptors: {all_descriptors.shape}")
    kmeans = KMeans(n_clusters=k, random_state=42, verbose=1, n_init=10)
    kmeans.fit(all_descriptors)
    return kmeans


def build_histograms(descriptors_list, kmeans, k):
    histograms = []
    for idx, descriptors in enumerate(descriptors_list):
        if descriptors.size != 0:
            words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(words, bins=np.arange(k+1))
        else:
            histogram = np.zeros(k)
        histograms.append(histogram)
        if idx % 1000 == 0:
            print(f"Built histogram for {idx} / {len(descriptors_list)} images")
    return np.array(histograms)


def train_svm(histograms, labels):
    scaler = StandardScaler().fit(histograms)
    histograms_scaled = scaler.transform(histograms)
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(histograms_scaled, labels)
    return svm, scaler


def evaluate_model(svm, scaler, histograms, labels):
    histograms_scaled = scaler.transform(histograms)
    predictions = svm.predict(histograms_scaled)
    acc = accuracy_score(labels, predictions)
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names=["T-Shirt", "Trouser", "Pullover", "Dress", "Coat"]))
    return acc, predictions


def main():
    print("Loading and preprocessing Fashion-MNIST data...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    print("Extracting SIFT features from training images...")
    train_descriptors_list = extract_sift_features(X_train)
    k = 100
    print("Clustering descriptors to build vocabulary...")
    kmeans = build_vocabulary(train_descriptors_list, k=k)
    print("Building histograms for training images...")
    train_histograms = build_histograms(train_descriptors_list, kmeans, k)
    print("Training SVM classifier...")
    svm, scaler = train_svm(train_histograms, y_train)
    print("Extracting SIFT features from testing images...")
    test_descriptors_list = extract_sift_features(X_test)
    print("Building histograms for testing images...")
    test_histograms = build_histograms(test_descriptors_list, kmeans, k)
    print("Evaluating the model on test data...")
    evaluate_model(svm, scaler, test_histograms, y_test)
    
    for i in range(10):
        img = X_test[i]
        true_label = y_test[i]
        descriptor = test_descriptors_list[i]
        words = kmeans.predict(descriptor) if descriptor.size != 0 else []
        prediction = svm.predict(scaler.transform([test_histograms[i]]))[0]
        
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {label_map(true_label)} | Pred: {label_map(prediction)}")
        plt.show()

def label_map(label):
    mapping = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat"
    }
    return mapping.get(label, "Unknown")

if __name__ == "__main__":
    main()
