"""
Random Forest model training and evaluation module.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


class RandomForestModel:
    """
    Random Forest classifier for crypto trading signals.
    """
    
    def __init__(self, **model_params):
        """
        Initialize Random Forest model.
        
        Args:
            **model_params: Parameters for RandomForestClassifier
        """
        self.model_params = model_params
        self.model = None
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            RandomForestClassifier: Trained model
        """
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_train, y_train)
        print("✓ Model trained successfully!")
        return self.model
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X)
        return {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions)
        }
    
    def get_detailed_metrics(self, X, y, data_type=""):
        """
        Get detailed evaluation metrics.
        
        Args:
            X: Features
            y: True labels
            data_type (str): Type of data being evaluated (e.g., "training", "test")
            
        Returns:
            dict: Detailed metrics including confusion matrix and classification report
        """
        if data_type:
            print(f"Evaluating on {data_type} data...")
        
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(y, predictions),
            'classification_report': classification_report(y, predictions, output_dict=True, zero_division=0),
            'predictions': predictions,
            'num_signals': sum(predictions)
        }
        
        if data_type:
            print(f"✓ {data_type.capitalize()} Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def get_model(self):
        """
        Get the trained model.
        
        Returns:
            RandomForestClassifier: Trained model
        """
        return self.model
