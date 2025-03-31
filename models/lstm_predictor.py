import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle  # For saving the scaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
data_path = "/Users/mihir/Desktop/BE Project/data/synthetic_data.csv"  
df = pd.read_csv(data_path)
df.rename(columns=lambda x: x.strip(), inplace=True) 

# Ensure correct columns are present
required_columns = ['Order_Quantity', 'Demand', 'Inventory_Level', 'Lead_Time', 'Machine_Availability', 'Supplier_Reliability']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}. Available columns: {df.columns.tolist()}")

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[required_columns])

# Save the scaler for future inference
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Convert to sequences for LSTM
def create_sequences(data, seq_length=10):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length, 1])  # Predicting 'Demand'
    return np.array(sequences), np.array(targets)

seq_length = 10
X, y = create_sequences(scaled_data, seq_length)

# Ensure correct shape for PyTorch tensors
X_train, y_train = torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)  # Ensure y_train is 2D

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# LSTM Model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.unsqueeze(1) if len(lstm_out.shape) == 2 else lstm_out  # Ensure 3D shape
        return self.fc(lstm_out[:, -1, :])

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPredictor(input_size=len(required_columns)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(model, train_loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Train the model
train_model(model, train_loader)

# Save Model
torch.save(model.state_dict(), "lstm_predictor.pth")
print("Model and scaler saved successfully!")
