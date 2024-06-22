import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from googoolenet_618 import GoogLeNet

# 데이터 로드 함수
def load_data(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    x_data, t_data = dataset
    return x_data, t_data

def augment_data(x_data, strategy):
    aug_data = []
    if strategy == 1:
        transform = transforms.Compose([
            transforms.RandomRotation(8),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        ])
    elif strategy == 2:
        transform = transforms.Compose([
            transforms.RandomRotation(16),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        ])

    for img in x_data:
        img = torch.tensor(img, dtype=torch.float32)
        aug_img = transform(img)
        aug_data.append(aug_img.numpy())

    return np.array(aug_data)

def load_dataset(batch_size):
    x_train1, t_train1 = load_data('data3D.pkl')
    x_train2, t_train2 = load_data('ttt3D.pkl')
    x_train3, t_train3 = load_data('Train_3D_Real.pkl')
    x_train4, t_train4 = load_data('testset3D.pkl')

    x_train = np.concatenate([x_train1, x_train2, x_train3, x_train4])
    t_train = np.concatenate([t_train1, t_train2, t_train3, t_train4])
    print(f"Original data size: {x_train.shape[0]} samples")
    x_train, x_test, t_train, t_test = train_test_split(x_train, t_train, test_size=0.3, random_state=1, stratify=t_train)

    x_train_aug1 = augment_data(x_train, strategy=1)
    x_train_aug2 = augment_data(x_train, strategy=2)
    t_train_aug1 = t_train
    t_train_aug2 = t_train

    x_train = torch.tensor(x_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.long)
    x_train_aug1 = torch.tensor(x_train_aug1, dtype=torch.float32)
    t_train_aug1 = torch.tensor(t_train_aug1, dtype=torch.long)
    x_train_aug2 = torch.tensor(x_train_aug2, dtype=torch.float32)
    t_train_aug2 = torch.tensor(t_train_aug2, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train, t_train)
    augmented_dataset1 = TensorDataset(x_train_aug1, t_train_aug1)
    augmented_dataset2 = TensorDataset(x_train_aug2, t_train_aug2)
    test_dataset = TensorDataset(x_test, t_test)

    combined_train_dataset = ConcatDataset([train_dataset, augmented_dataset1, augmented_dataset2])

    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # 하이퍼파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', action='store', type=int, default=32)
    parser.add_argument('--learning_rate', action='store', type=float, default=0.001)
    parser.add_argument('--n_epochs', action='store', type=int, default=20)
    parser.add_argument('--plot', action='store', type=bool, default=True)
    args = parser.parse_args()

    np.random.seed(1)
    torch.manual_seed(1)

    # 데이터셋 로드
    train_loader, test_loader = load_dataset(args.batch_size)

    # 모델, 손실 함수, 옵티마이저 정의
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(aux_logits=False, num_classes=14).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 학습 루프
    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0
        correct, total = 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                print(f"[*] Epoch: {epoch} \tStep: {batch_idx}/{len(train_loader)}"
                      f"\tTrain accuracy: {correct / total:.4f} \tTrain Loss: {train_loss / total:.4f}")

        # 검증 루프
        model.eval()
        valid_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                _, preds = torch.max(output, 1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

                if batch_idx % 100 == 0:
                    print(f"[*] Step: {batch_idx}/{len(test_loader)}\tValid accuracy: {correct / total:.4f}"
                          f"\tValid Loss: {valid_loss / total:.4f}")

    # 학습 후 파라미터 개수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters before training: ", total_params)

    # 모델 파라미터 저장
    torch.save(model.state_dict(), 'params_googlenet_19000_32 (1).pkl')
