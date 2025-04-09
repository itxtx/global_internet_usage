import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def split_multivariate_sequences(data, n_steps, n_outputs=1):
    """
    Split a multivariate dataset into samples for multiple country prediction
    
    Args:
        data: DataFrame with countries as columns and years as index
        n_steps: Number of time steps to use as input
        n_outputs: Number of countries to predict (default: 1 for single target)
        
    Returns:
        X, y: Input sequences and target values
    """
    X, y = [], []
    
    # For each starting point
    for i in range(len(data) - n_steps):
        # Get input sequence
        end_ix = i + n_steps
        # Get all countries data for the input sequence
        seq_x = data.iloc[i:end_ix].values
        
        # Get output value(s)
        if n_outputs == 1:  # Single target country
            # Assuming the target country is the first column
            seq_y = data.iloc[end_ix, 0]  
        else:  # Multiple target countries
            # Get multiple countries' values for prediction
            seq_y = data.iloc[end_ix, :n_outputs].values
            
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(StackedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # First LSTM layer - returns sequences for the second LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Second LSTM layer - processes the sequences from the first layer
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Fully connected layers
        self.dense1 = nn.Linear(hidden_size, hidden_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.dense2 = nn.Linear(hidden_size//2, output_size)
        
        # Training history
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'lr': []
        }
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden states for first LSTM layer
        h0_1 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0_1 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate first LSTM layer
        # Output shape: (batch_size, sequence_length, hidden_size)
        lstm1_out, _ = self.lstm1(x, (h0_1, c0_1))
        
        # Initialize hidden states for second LSTM layer
        h0_2 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0_2 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate second LSTM layer
        # Output shape: (batch_size, sequence_length, hidden_size)
        lstm2_out, _ = self.lstm2(lstm1_out, (h0_2, c0_2))
        
        # Get output of last time step
        lstm_out = lstm2_out[:, -1, :]
        
        # Apply dense layers
        x = self.dense1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        
        return x
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=75, batch_size=32, 
            learning_rate=1e-4, patience=10, factor=0.5, min_lr=1e-6, device="cpu"):
        """
        Complete training method with validation, early stopping, and learning rate scheduling
        """
        # Debug the shapes
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        
        # Make sure the shapes match on the first dimension
        assert X_train.shape[0] == y_train.shape[0], f"Training data mismatch: X_train has {X_train.shape[0]} samples but y_train has {y_train.shape[0]}"
        assert X_val.shape[0] == y_val.shape[0], f"Validation data mismatch: X_val has {X_val.shape[0]} samples but y_val has {y_val.shape[0]}"
        
        # Reset training history
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'lr': []
        }
        
        # Prepare data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience//2, 
            min_lr=min_lr, verbose=True
        )
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        
        # Move model to device
        self.to(device)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(inputs)
                
                # Ensure targets have correct shape for loss calculation
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)
                
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)
                    
                    # Ensure targets have correct shape for loss calculation
                    if len(targets.shape) == 1:
                        targets = targets.unsqueeze(1)
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_rmse = np.sqrt(val_loss)
            
            # Store metrics in history
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_rmse)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.6f} - Val loss: {val_loss:.6f} - Val RMSE: {val_rmse:.6f}')
            
            # Adjust learning rate
            scheduler.step(val_loss)
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict().copy()
                early_stopping_counter = 0
                print(f"New best model saved with validation loss: {val_loss:.6f}")
            else:
                early_stopping_counter += 1
                
            # Check early stopping condition
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            
        # Save the best model
        torch.save(self.state_dict(), 'best_country_time_series_model.pt')
        print("Training complete. Best model saved.")
        
        return self
    
    def plot_training_history(self, figsize=(20, 12), log_scale=True):
        """
        Plot the training history metrics
        
        Args:
            figsize: Figure size as (width, height)
            log_scale: Whether to use log scale for loss plots
        
        Returns:
            matplotlib.figure.Figure: The figure containing the plots
        """
        if not self.history['epochs']:
            print("No training history available. Please train the model first.")
            return None
        
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Plot training and validation loss
        ax = axes[0]
        ax.plot(self.history['epochs'], self.history['train_loss'], 'b-', label='Training Loss', 
                linewidth=2, marker='o', markersize=4)
        ax.plot(self.history['epochs'], self.history['val_loss'], 'r-', label='Validation Loss', 
                linewidth=2, marker='x', markersize=6)
        
        # Find best validation loss point
        best_val_loss_idx = np.argmin(self.history['val_loss'])
        best_val_loss_epoch = self.history['epochs'][best_val_loss_idx]
        best_val_loss = self.history['val_loss'][best_val_loss_idx]
        
        # Highlight best model
        ax.scatter(best_val_loss_epoch, best_val_loss, s=150, c='green', marker='*', 
                  label=f'Best Model (Epoch {best_val_loss_epoch}, Loss {best_val_loss:.6f})', zorder=10)
        
        # Add gray vertical line at best model
        ax.axvline(x=best_val_loss_epoch, color='gray', linestyle='--', alpha=0.5)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Use log scale if requested
        if log_scale:
            ax.set_yscale('log')
        
        # 2. Plot RMSE
        ax = axes[1]
        ax.plot(self.history['epochs'], self.history['val_rmse'], 'g-', label='Validation RMSE', 
                linewidth=2, marker='s', markersize=6)
        
        # Find best RMSE point
        best_rmse_idx = np.argmin(self.history['val_rmse'])
        best_rmse_epoch = self.history['epochs'][best_rmse_idx]
        best_rmse = self.history['val_rmse'][best_rmse_idx]
        
        # Highlight best RMSE
        ax.scatter(best_rmse_epoch, best_rmse, s=150, c='purple', marker='*', 
                  label=f'Best RMSE (Epoch {best_rmse_epoch}, RMSE {best_rmse:.6f})', zorder=10)
        
        # Add gray vertical line at best RMSE
        ax.axvline(x=best_rmse_epoch, color='gray', linestyle='--', alpha=0.5)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Validation RMSE Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 3. Plot Learning Rate
        ax = axes[2]
        ax.plot(self.history['epochs'], self.history['lr'], 'c-', 
                linewidth=2, marker='d', markersize=6)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Use log scale for learning rate
        ax.set_yscale('log')
        
        # 4. Plot Train vs Val Loss Ratio (to detect overfitting)
        ax = axes[3]
        loss_ratio = [v/t for t, v in zip(self.history['train_loss'], self.history['val_loss'])]
        ax.plot(self.history['epochs'], loss_ratio, 'm-', 
                linewidth=2, marker='^', markersize=6)
        
        # Add horizontal line at ratio=1
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Val Loss / Train Loss Ratio', fontsize=12)
        ax.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add annotation for interpretation
        if max(loss_ratio) > 1.5:
            ax.text(0.5, 0.9, "Ratio > 1: Potential overfitting", 
                   transform=ax.transAxes, ha='center', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Add a title for the entire figure
        plt.suptitle('Country Time Series LSTM Model Training Metrics', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Print summary statistics
        print(f"Training Summary:")
        print(f"Total Epochs: {max(self.history['epochs'])}/75")
        print(f"Best Validation Loss: {best_val_loss:.6f} (Epoch {best_val_loss_epoch})")
        print(f"Best Validation RMSE: {best_rmse:.6f} (Epoch {best_rmse_epoch})")
        print(f"Final Training Loss: {self.history['train_loss'][-1]:.6f}")
        print(f"Final Validation Loss: {self.history['val_loss'][-1]:.6f}")
        print(f"Final Validation RMSE: {self.history['val_rmse'][-1]:.6f}")
        
        return fig
    
    def predict(self, X, batch_size=64, device="cpu"):
        """
        Make predictions on new data
        """
        self.eval()
        self.to(device)
        
        # Convert data to tensor
        tensor_x = torch.FloatTensor(X)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def save(self, path):
        """
        Save model to file
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'num_layers': self.num_layers
            },
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, device="cpu"):
        """
        Load model from file
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = cls(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            num_layers=config['num_layers']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load history if available
        if 'history' in checkpoint:
            model.history = checkpoint['history']
        
        model.to(device)
        return model

def prepare_country_data(time_series_df, target_country=None, n_steps=5, test_size=0.2, random_state=42):
    """
    Prepare country time series data for LSTM model
    
    Args:
        time_series_df: DataFrame with countries as columns and years as index
        target_country: Country to predict (if None, first country is used)
        n_steps: Number of time steps for input sequence
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, y_train, y_val, scaler: Prepared data and scaler
    """
    # Handle missing values in the data
    df_filled = time_series_df.fillna(method='ffill').fillna(method='bfill')
    
    # If no target country is specified, use the first column
    if target_country is None:
        target_country = df_filled.columns[0]
    
    # Ensure target country is in the dataframe
    if target_country not in df_filled.columns:
        raise ValueError(f"Target country '{target_country}' not found in dataframe")
    
    # Create a new dataframe with target country as first column
    cols = [target_country] + [col for col in df_filled.columns if col != target_country]
    df_reordered = df_filled[cols]
    
    # Scale the data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_reordered),
        index=df_reordered.index,
        columns=df_reordered.columns
    )
    
    # Split into sequences
    X, y = split_multivariate_sequences(df_scaled, n_steps=n_steps)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val, scaler

    

# Example usage:
"""
# Assuming time_series_df is your pandas DataFrame with countries as columns
# and years as indices

# 1. Prepare the data
X_train, X_val, y_train, y_val, scaler = prepare_country_data(
    time_series_df,
    target_country='Germany',  # Country you want to predict
    n_steps=5,                 # Number of time steps to use as input
    test_size=0.2              # 20% of data for validation
)

# 2. Define model parameters
input_size = X_train.shape[2]   # Number of countries (features)
hidden_size = 64                # Hidden layer size
output_size = 1                 # Single target country
num_layers = 2                  # Number of LSTM layers

# 3. Create and train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StackedLSTM(input_size, hidden_size, output_size, num_layers)

model.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    patience=15,
    device=device
)

# 4. Plot training progress
fig = model.plot_training_history()
plt.show()

# 5. Make predictions on validation data
val_predictions = model.predict(X_val, device=device)

# 6. Inverse transform predictions to original scale
val_pred_reshaped = np.zeros((val_predictions.shape[0], X_train.shape[2]))
val_pred_reshaped[:, 0] = val_predictions.flatten()  # Put predictions in first column
val_pred_original = scaler.inverse_transform(val_pred_reshaped)[:, 0]  # Get first column

# 7. Inverse transform actual values
y_val_reshaped = np.zeros((y_val.shape[0], X_train.shape[2]))
y_val_reshaped[:, 0] = y_val  # Put actual values in first column
y_val_original = scaler.inverse_transform(y_val_reshaped)[:, 0]  # Get first column

# 8. Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_val_original, 'b-', label='Actual')
plt.plot(val_pred_original, 'r--', label='Predicted')
plt.title(f'Internet Usage Prediction for Germany')
plt.xlabel('Time Step')
plt.ylabel('Internet Usage (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""