import torch
import pandas as pd
import os
from utils import limit_data
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from model import BirdDroneNet, BirdDroneResNet50, ImageDataset
import torch.nn as nn
import torch.nn.functional as F
from utils import extract_features, evaluate
from torchvision import transforms, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils import ROC_curve, confusion_matrix_linear
from model import ImageFeatureDataset
from torch.utils.data import DataLoader
import torchvision.models 
from model import MLP
import warnings 

warnings.filterwarnings('ignore')

train_path = "Train"
test_path = "Test"
val_path = "Val"
categories = ['Bird', 'Drone']
IMAGE_SAMPLES = None
df_train=pd.DataFrame(columns=['path','label'])
df_val=pd.DataFrame(columns=['path','label'])
df_test=pd.DataFrame(columns=['path','label'])

df_train=limit_data(train_path,categories,IMAGE_SAMPLES)
df_val=limit_data(val_path,categories,IMAGE_SAMPLES)
df_test=limit_data(test_path,categories,IMAGE_SAMPLES)

df_train=df_train.sample(frac=1, ignore_index=True)
df_val=df_val.sample(frac=1, ignore_index=True)
df_test=df_test.sample(frac=1, ignore_index=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

train_dataset = ImageDataset(df_train, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = ImageDataset(df_test, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, nepochs):
    #List to store loss to visualize
    valid_loss_min = np.inf # track change in validation loss

    train_losses = []
    test_losses = []
    acc_eval = []
    lr = []
    #test_counter = [i*len(train_loader.dataset) for i in n_epochs]

    for epoch in range(nepochs):
        # keep track of training and validation loss
        train_loss = 0.
        valid_loss = 0.

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device = device)
            target = target.to(device = device).float()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            # calculate the batch loss
            loss = criterion(output, target)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()
            
            # update training loss
            train_loss += loss.item() * data.size(0)
        
        ######################    
        # validate the model #
        ######################
        model.eval()
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            with torch.no_grad():
                data = data.to(device = device)
                target = target.to(device = device).float()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)

                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
                output = output.view(-1, 1)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
        
        valid_loss /= len(test_loader)  # Average validation loss
        scheduler.step(valid_loss)
        lr.append(optimizer.param_groups[0]['lr'])

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(test_loader.dataset)
        acc_eval.append(correct/len(test_loader.dataset)*100)
        train_losses.append(train_loss)
        test_losses.append(valid_loss)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
    X = range(1,nepochs+1)
    plt.plot(X,train_losses,color='red',label='Train loss')
    plt.plot(X,test_losses,color='green',label='Test loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Test Losses")
    plt.legend()
    plt.show()
    plt.plot(X,lr,color='blue',label="Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Evolution of the learning rate")
    plt.legend()
    plt.show()

resnet_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def main():
    # Define the model, loss function, optimizer, and learning rate scheduler
    print("Using device:", device)
    print("Loading BirdDroneNet (CNN) model...")
    model = BirdDroneNet().to(device = device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    nepochs = 10
    train_model(model, criterion, optimizer, scheduler, nepochs)
    torch.save(model.state_dict(), 'model_CNN.pth')
    print("Model saved as model_CNN.pth")

    #Evaluating the CNN model
    print("-"*50)
    print("Evaluating the CNN model...")
    model.load_state_dict(torch.load('model_CNN.pth'))
    evaluate(model, test_loader)
    print("Model evaluation completed.")
    print("-"*50)
    print("Training and evaluation completed.")
    print("-"*50)

    print("Loading BirdDroneResNet50 model...")
    # Load the ResNet50 model
    resnet_model = BirdDroneResNet50().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    nepochs = 10
    # Train the ResNet50 model
    train_model(resnet_model, criterion, optimizer, scheduler, nepochs)

    #torch.save(resnet_model.state_dict(), 'model_ResNet.pth')
    print("Model saved as model_ResNet.pth")
    
    input_channels = 3072  # Adjust according to input dimensions
    sigma_w = 0.01
    model = MLP([input_channels, 6, 16, 120, 84, 1], sigma_w).to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # Train the model
    nepochs = 5
    train_model(model, criterion, optimizer, scheduler, nepochs)
    torch.save(model.state_dict(), 'model_MLP.pth')
    print("Evaluating the MLP model...")
    resnet_model.load_state_dict(torch.load('model_MLP.pth'))
    evaluate(model, test_loader)
    print("Model evaluation completed.")
    print("-"*50)
    print("Training and evaluation completed.")

    #Evaluating the ResNet50 model
    print("Evaluating the ResNet model...")
    resnet_model.load_state_dict(torch.load('model_ResNet.pth'))
    evaluate(model, test_loader)
    print("Model evaluation completed.")
    print("-"*50)
    print("Training and evaluation completed.")
    print("-"*50)
    print("Extracting features from the ResNet50 model...")

    resnet50 = torchvision.models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    train_feat_dataset = ImageFeatureDataset(df_train, resnet_transform)
    test_feat_dataset = ImageFeatureDataset(df_test, resnet_transform)
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=32, shuffle=False)
    test_feat_loader = DataLoader(test_feat_dataset, batch_size=32, shuffle=False)

    X_train, y_train = extract_features(train_feat_loader, feature_extractor)
    X_test, y_test = extract_features(test_feat_loader, feature_extractor)

    print("Train features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)

        # Use the features extracted previously: X_train, y_train
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, scoring='accuracy')

    print(f"RandomForest cross-validation accuracy scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "Random Forest Classifier": RandomForestClassifier(),
    "SVM Classifier": SVC(probability=True, kernel='rbf', random_state=42)
    }
    print("Plotting the Linear Models...")
    ROC_curve(models, X_train, y_train, X_test, y_test)
    print("Plotting completed.")
    print("-"*50)
    print("Confusion Matrix for the models...")
    confusion_matrix_linear(models, X_test, y_test)

if __name__ == "__main__":
    main()