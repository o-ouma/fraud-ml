import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.ensemble import IsolationForest
import numpy as np


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x):
        # For inference, treat input as feature matrix without graph structure
        # Create a dummy edge_index that connects each node to itself
        batch_size = x.size(0)
        edge_index = torch.tensor([[i, i] for i in range(batch_size)],
                                  dtype=torch.long).t().contiguous()

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = x.mean(dim=1, keepdim=True)  # Average features for final prediction
        return x

    def get_embeddings(self, x):
        # Get intermediate embeddings from conv1 layer
        batch_size = x.size(0)
        edge_index = torch.tensor([[i, i] for i in range(batch_size)],
                                  dtype=torch.long).t().contiguous()

        with torch.no_grad():
            embeddings = self.conv1(x, edge_index)
        return embeddings


class FraudGNNModel:
    def __init__(self, model_path, threshold=-0.01):
        self.model_path = model_path
        self.model = None
        self.iso_forest = IsolationForest(contamination=0.1, random_state=42)
        self.threshold = threshold
        self.load_model()

        # Pre-fit isolation forest with some normal transaction patterns
        normal_patterns = torch.tensor([
            [1000.0, 2000.0, 1000.0, 1000.0, 2000.0],  # Normal transaction
            [500.0, 1000.0, 500.0, 500.0, 1000.0],     # Normal transaction
            [2000.0, 4000.0, 2000.0, 2000.0, 4000.0],  # Normal transaction
        ], dtype=torch.float32)

        # Get embeddings for normal patterns
        with torch.no_grad():
            normal_embeddings = self.model.get_embeddings(normal_patterns)
            normal_embeddings_np = normal_embeddings.numpy()

        # Pre-fit isolation forest with embeddings of normal patterns
        self.iso_forest.fit(normal_embeddings_np)

    def load_model(self):
        # Initialize model with the same architecture as training
        self.model = GNNModel(input_dim=5, hidden_dim=64, output_dim=32)
        # Load the state dictionary
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, features):
        with torch.no_grad():
            inputs = self._preprocess_features(features)

            # Get base prediction
            outputs = self.model(inputs)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            # Get embeddings for anomaly detection
            embeddings = self.model.get_embeddings(inputs)
            embeddings_np = embeddings.numpy()

            # Fit and predict anomalies if not already fitted
            if not hasattr(self.iso_forest, 'offset_'):
                self.iso_forest.fit(embeddings_np)

            # Get anomaly scores
            anomaly_scores = self.iso_forest.decision_function(embeddings_np)
            is_anomaly = (anomaly_scores < self.threshold).astype(int)

            return {
                "prediction": predictions.item(),
                "probability": probabilities.item(),
                "anomaly_score": float(anomaly_scores[0]),
                "is_potentially_fraudulent": int(is_anomaly[0])
            }

    def _preprocess_features(self, features):
        try:
            # Extract features in the same order as training
            required_features = [
                'amount',
                'oldbalanceOrg',
                'newbalanceOrig',
                'oldbalanceDest',
                'newbalanceDest'
            ]

            feature_values = []
            for feature in required_features:
                if feature not in features:
                    raise ValueError(f"Feature '{feature}' not found in input data")
                value = float(features[feature])
                feature_values.append(value)

            # Convert to tensor with correct shape and type
            processed_features = torch.tensor(
                feature_values,
                dtype=torch.float32).reshape(1, -1)  # Add batch dimension

            return processed_features

        except Exception as e:
            raise ValueError(f"Error preprocessing input features: {str(e)}")