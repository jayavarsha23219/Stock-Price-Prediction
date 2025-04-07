# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.


## Design Steps
### Step 1:
Import necessary libraries.

### Step 2:
Load and preprocess the data.

### Step 3:
Create input-output sequences.

### Step 4:
Convert data to PyTorch tensors.

### Step 5:
Define the RNN model.

### Step 6:
Train the model using the training data.

### Step 7:
Evaluate the model and plot predictions.


## Program
#### Name: JAYAVARSHA T
#### Register Number: 212223040075
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out
    
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/user-attachments/assets/a270cbb6-9c4a-446f-92d5-112f1aeab417)


### Predictions 

![image](https://github.com/user-attachments/assets/e2c9b087-678a-47b0-9962-d0e5483e6c7d)


## Result
Thus, a Recurrent Neural Network model for stock price prediction has successfully been devoloped.

