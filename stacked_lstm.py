import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def prepare_multivariate_data(time_series_df, target_country=None, n_steps=5, test_size=0.2, val_size=0.25, random_state=42):
    """
    Prepare multivariate time series data for LSTM model with improved handling
    
    Args:
        time_series_df: DataFrame with countries as columns and years as index
        target_country: Country to predict (if None, first country is used)
        n_steps: Number of time steps for input sequence
        test_size: Proportion of data to use for testing
        val_size: Proportion of remaining data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing all prepared data and metadata
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
    
    # Print shapes for debugging
    print(f"Original dataframe shape: {df_reordered.shape}")
    print(f"Target country: {target_country}")
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_reordered)
    df_scaled = pd.DataFrame(
        scaled_values,
        index=df_reordered.index,
        columns=df_reordered.columns
    )
    
    # Create sequences for multivariate prediction
    X, y = [], []
    
    # For each starting point
    for i in range(len(df_scaled) - n_steps):
        # Get input sequence (all countries, n_steps time points)
        end_ix = i + n_steps
        seq_x = df_scaled.iloc[i:end_ix].values  # Shape: (n_steps, n_countries)
        
        # Get target value (only target country, single time point)
        seq_y = df_scaled.iloc[end_ix, 0]  # First column is target country
            
        X.append(seq_x)
        y.append(seq_y)
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)  # Reshape to (samples, 1)
    
    # Print shapes for debugging
    print(f"X shape after sequence creation: {X.shape}")
    print(f"y shape after sequence creation: {y.shape}")
    
    # First split data into train+val and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    # Then split the temporary set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, shuffle=False, random_state=random_state
    )
    
    # Print final shapes for debugging
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Save the years for each dataset for later visualization
    all_years = df_reordered.index.tolist()
    n_total = len(all_years)
    n_test = len(y_test)
    n_val = len(y_val)
    n_train = len(y_train)
    
    test_years = all_years[-n_test:]
    val_years = all_years[-(n_test + n_val):-n_test]
    train_years = all_years[:-(n_test + n_val)]
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'target_country': target_country,
        'input_size': X_train.shape[2],  # Number of features (countries)
        'train_years': train_years,
        'val_years': val_years,
        'test_years': test_years,
        'all_countries': df_reordered.columns.tolist()
    }

class ImprovedMultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout=0.2):
        """
        Improved LSTM model for multivariate time series forecasting
        
        Args:
            input_size: Number of input features (countries)
            hidden_size: Number of hidden units
            output_size: Number of output features (usually 1 for single target country)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(ImprovedMultivariateLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # LSTM layers with proper dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers with batch normalization for better training
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
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
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch_size, seq_len, hidden_size)
        
        # Get the output from the last time step
        last_time_step = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply batch normalization
        normalized = self.bn(last_time_step)
        
        # Apply fully connected layers
        fc1_out = self.fc1(normalized)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout1(fc1_out)
        output = self.fc2(fc1_out)
        
        return output
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
            learning_rate=1e-3, patience=15, factor=0.5, min_lr=1e-6, device="cpu"):
        """
        Train the model with early stopping and learning rate scheduling
        """
        # Verify input shapes
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        
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
        
        # Move model to device
        self.to(device)
        
        # Initialize optimizer with weight decay (L2 regularization)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
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
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader.dataset)
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            
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
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'num_layers': self.num_layers
            },
            'history': self.history
        }, 'best_multivariate_lstm_model.pt')
        
        print("Training complete. Best model saved.")
        return self
    
    def evaluate(self, X_test, y_test, scaler=None, target_idx=0, device="cpu"):
        """
        Evaluate the model on test data with option to inverse transform predictions
        
        Args:
            X_test: Test input data
            y_test: Test target data
            scaler: The scaler used to normalize the data
            target_idx: The index of the target variable in the original dataset
            device: Device to use for computation
            
        Returns:
            Dictionary with evaluation results
        """
        self.eval()
        self.to(device)
        
        # Convert data to tensors
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Reshape if needed
        if len(actuals.shape) == 1:
            actuals = actuals.reshape(-1, 1)
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Calculate scaled metrics
        rmse_scaled = np.sqrt(mean_squared_error(actuals, predictions))
        mae_scaled = mean_absolute_error(actuals, predictions)
        
        # If scaler is provided, inverse transform predictions and actuals
        if scaler is not None:
            # Create dummy arrays with zeros for all features
            pred_dummy = np.zeros((predictions.shape[0], scaler.scale_.shape[0]))
            actual_dummy = np.zeros((actuals.shape[0], scaler.scale_.shape[0]))
            
            # Place predictions and actuals in the target column position
            pred_dummy[:, target_idx] = predictions.flatten()
            actual_dummy[:, target_idx] = actuals.flatten()
            
            # Inverse transform
            pred_inversed = scaler.inverse_transform(pred_dummy)[:, target_idx].reshape(-1, 1)
            actual_inversed = scaler.inverse_transform(actual_dummy)[:, target_idx].reshape(-1, 1)
            
            # Calculate unscaled metrics
            rmse = np.sqrt(mean_squared_error(actual_inversed, pred_inversed))
            mae = mean_absolute_error(actual_inversed, pred_inversed)
            
            # Calculate MAPE, handling zero values
            epsilon = 1e-10  # Small value to avoid division by zero
            mape = np.mean(np.abs((actual_inversed - pred_inversed) / (np.abs(actual_inversed) + epsilon))) * 100
            
            print(f"Test Results (Original Scale):")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"MAPE: {mape:.6f}%")
            
            return {
                'predictions_scaled': predictions,
                'actuals_scaled': actuals,
                'predictions': pred_inversed,
                'actuals': actual_inversed,
                'rmse_scaled': rmse_scaled,
                'mae_scaled': mae_scaled,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
        else:
            print(f"Test Results (Scaled):")
            print(f"RMSE: {rmse_scaled:.6f}")
            print(f"MAE: {mae_scaled:.6f}")
            
            return {
                'predictions_scaled': predictions,
                'actuals_scaled': actuals,
                'rmse_scaled': rmse_scaled,
                'mae_scaled': mae_scaled
            }
    
    def plot_training_history(self, figsize=(20, 12), log_scale=True):
        """
        Plot the training history metrics
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
        plt.suptitle('Multivariate LSTM Model Training Metrics', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        return fig
    
    def plot_predictions(self, predictions, actuals, years=None, title="Model Predictions vs Actual Values", 
                         country_name=None, figsize=(14, 7)):
        """
        Plot model predictions against actual values
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            years: List of years for x-axis (if available)
            title: Plot title
            country_name: Name of the country being predicted
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Flatten arrays if needed
        predictions = predictions.flatten()
        actuals = actuals.flatten()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create x-axis values
        if years is not None:
            x_values = years
            x_label = "Year"
        else:
            x_values = np.arange(len(predictions))
            x_label = "Time Step"
        
        # Plot actual vs predicted
        ax.plot(x_values, actuals, 'b-', label='Actual', linewidth=2, marker='o')
        ax.plot(x_values, predictions, 'r--', label='Predicted', linewidth=2, marker='x')
        
        # Shade the area between
        ax.fill_between(x_values, actuals, predictions, color='lightgray', alpha=0.3)
        
        # Add country name to title if provided
        if country_name:
            title = f"{title} - {country_name}"
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        # Add metrics to plot
        metrics_text = f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Format x-axis to show years properly if provided
        if years is not None:
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        return fig