import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_conv_layers=2, conv_filters=[32, 64], kernel_sizes=[3, 3], dropout_rate=0.5, fc_sizes=[64, 32], input_channels=1):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)  # Adjust stride to 1 to avoid too much reduction
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        in_channels = input_channels
        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Conv1d(in_channels=in_channels, out_channels=conv_filters[i], kernel_size=kernel_sizes[i], stride=1, padding=1)
            )
            in_channels = conv_filters[i]
        
        # Calculate the size of the flattened feature map
        fc_input_size = in_channels * 1  # Adjusted to match the final output size
        
        self.fc_layers = nn.ModuleList()
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_size))
            fc_input_size = fc_size
        
        self.output_layer = nn.Linear(fc_sizes[-1], 1)
        
    def forward(self, x):
        for conv in self.conv_layers:
            x = self.pool(self.relu(conv(x)))
            x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        
        for fc in self.fc_layers:
            x = self.relu(fc(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x
