import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd 
from pyspark.sql.functions import col
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, epochs, learning_rate):
    model = model.to(device)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()  # TensorBoard writer

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in rqdm(train_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.reshape(-1,1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, criterion)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Logging to TensorBoard
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

    writer.close()
    return model

def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.reshape(-1,1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def data_loader(train, val, batch_size=2048):
    # Load your dataset here and preprocess
    train_df = train.select('outcome', 'features').toPandas()
    val_df = val.select('outcome', 'features').toPandas()
    
    X_train = np.array(train_df['features'].apply(lambda x: x.toArray()).tolist())
    y_train = np.array(train_df['outcome'].values)
    X_val = np.array(val_df['features'].apply(lambda x: x.toArray()).tolist())
    y_val = np.array(val_df['outcome'].values)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

if __name__ == "__main__":
    pass
    #train_loader, val_loader = data_loader()
    #model = MLPRegressor(input_size=50, hidden_sizes=[128, 64, 32], output_size=1)
    #trained_model = train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001)
    # Model saving, further validation, or testing can be done here
