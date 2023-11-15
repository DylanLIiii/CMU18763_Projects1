import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For positional encoding in Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPRegressor, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)
        return x

# Transformer based model can be defined similarly
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1).to(device)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        
        return x

class TransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerRegressor, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.fc_in = nn.Linear(input_size, d_model)
        self.fc_out = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, src):
        src = self.fc_in(src)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output.mean(dim=0))
        return output

# Usage example: model = TransformerRegressor(input_size=50, d_model=512, nhead=8, num_layers=6, output_size=1)

class MLPResidualRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPResidualRegressor, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())

        self.skip_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.skip_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        residual = x
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)
            if i < len(self.skip_layers):
                residual = self.skip_layers[i//2](residual)
                x += residual  # Skip connection
                residual = x
            x = self.layers[i+1](x)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    model = MLPRegressor(50, [128, 64, 32], 1)
    print(model.__class__.__name__)