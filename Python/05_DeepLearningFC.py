import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class WeatherData(Dataset):
    def __init__(self, csv_file_path):
        self.df = pd.read_csv(csv_file_path)
        self.X_columns = ['precipitation', 'temp_max', 'temp_min', 'wind']
        self.y_column = 'weather'

        self.X_df = self.df[self.X_columns]
        y_series = self.df[self.y_column]
        self.num_y = len(y_series.unique())
        self.y = pd.get_dummies(y_series)

        self.scaler = StandardScaler()
        self.X_df = pd.DataFrame(self.scaler.fit_transform(self.X_df), columns=self.X_columns)

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx):
        X = self.X_df.iloc[idx]
        y = self.y.iloc[idx]
        return torch.tensor(X).float(), torch.tensor(y).float()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


if __name__ == '__main__':
    # download data from here
    # https://www.kaggle.com/ananthr1/weather-prediction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    csv_file_path = 'seattle-weather.csv'
    dataset = WeatherData(csv_file_path)

    n_samples = len(dataset)
    train_size = int(len(dataset) * 0.8)
    val_size = n_samples - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
    )

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 20

    for epoch in range(n_epochs):
        with tqdm(total=len(train_loader), unit="batch") as pbar:
            pbar.set_description(f"Epoch[{epoch}/{n_epochs}]")
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            _, predicted = torch.max(y_pred.data, 1)
            y = torch.argmax(y, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Accuracy: {correct/total}")
