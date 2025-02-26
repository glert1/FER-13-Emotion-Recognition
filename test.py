import torch
from torchvision import models
from dataloader import get_data_loaders
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_model(model_name, num_classes):
    if model_name == "inception":
        model = models.inception_v3(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model.to(device)

def evaluate_model(model, test_loader, class_names):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

if __name__ == "__main__":
    _, test_loader, classes = get_data_loaders("dataset/train", "dataset/test")

    model_name = "vgg16"  
    num_classes = len(classes)

    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(f"{model_name}_trained.pth"))
    print(f"Model loaded from {model_name}_trained.pth")

    evaluate_model(model, test_loader, classes)
