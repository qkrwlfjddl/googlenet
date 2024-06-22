import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from googoolenet_618 import GoogLeNet

def load_data(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    x_data, t_data = dataset
    return x_data, t_data

def preprocess_data(x_data):
    x_data = torch.tensor(x_data, dtype=torch.float32)
    return x_data

def load_model(model_path):
    model = GoogLeNet(aux_logits=False, num_classes=14)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def evaluate_accuracy(model, data_loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    x_test, t_test = load_data('DATA_sample3D.pkl')
    idx = np.arange(x_test.shape[0])
    np.random.shuffle(idx)
    x_test = x_test[idx]
    t_test = t_test[idx]

    x_test = preprocess_data(x_test)
    t_test = torch.tensor(t_test, dtype=torch.long)

    test_dataset = TensorDataset(x_test, t_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model
    model_path = "params_googlenet_19000_32 (1).pkl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path).to(device)
    print("Load Network Parameters!")

    # Evaluate accuracy
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')

    """ 매개변수 """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters: ", total_params)
