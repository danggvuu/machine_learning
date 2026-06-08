import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ScratchLDA(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis (LDA) classifier implemented from scratch.
    
    This implementation inherits from sklearn's BaseEstimator and ClassifierMixin
    to be fully compatible with scikit-learn Pipelines, GridSearchCV, and cross-validation.
    
    Mathematical details:
    - Within-class scatter matrix (S_W): Measures the spread of samples within each class.
      S_W = sum_{c} sum_{x in c} (x - m_c)(x - m_c)^T
    - Between-class scatter matrix (S_B): Measures the distance between class means and overall mean.
      S_B = sum_{c} N_c * (m_c - m)(m_c - m)^T
    - Projection vectors are computed as the eigenvectors of S_W^{-1} * S_B.
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.linear_discriminants = None
        self.classes_ = None
        self.centroids_ = None
        self.projected_centroids_ = None

    def fit(self, X, y):
        """
        Fit the LDA model according to the given training data.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns:
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_features = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Calculate overall mean
        mean_overall = np.mean(X, axis=0)
        
        # Initialize within-class and between-class scatter matrices
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        self.centroids_ = []

        for c in self.classes_:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self.centroids_.append(mean_c)
            
            # S_W += (X_c - mean_c)^T * (X_c - mean_c)
            diff = X_c - mean_c
            S_W += diff.T.dot(diff)
            
            # S_B += n_c * (mean_c - mean_overall) * (mean_c - mean_overall)^T
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * mean_diff.dot(mean_diff.T)
            
        # Compute eigenvectors of S_W^-1 * S_B
        # Using pseudo-inverse (pinv) to handle singular/collinear features
        A = np.linalg.pinv(S_W).dot(S_B)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Eigenvectors are returned as columns, transpose to work with rows
        eigenvectors = eigenvectors.T
        
        # Keep only the real part (eigenvalues and eigenvectors can be complex due to numerical precision)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Sort eigenvalues and eigenvectors in descending order
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[idxs]
        
        # Determine number of components
        if self.n_components is None:
            # Theoretical maximum components for LDA is min(n_features, n_classes - 1)
            self.n_components = min(n_features, n_classes - 1)
            
        # Take the top eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]
        
        # Project class centroids to the new LDA space
        self.projected_centroids_ = []
        for mean_c in self.centroids_:
            self.projected_centroids_.append(np.dot(mean_c, self.linear_discriminants.T))
            
        return self

    def transform(self, X):
        """
        Project data to the LDA space.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input data to project.
            
        Returns:
        X_projected : array-like of shape (n_samples, n_components)
            Projected data.
        """
        X = np.asarray(X)
        if self.linear_discriminants is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before calling 'transform'.")
        return np.dot(X, self.linear_discriminants.T)
        
    def predict(self, X):
        """
        Predict class labels for samples in X based on closest centroid distance in projected space.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns:
        y_pred : array-like of shape (n_samples,)
            Predicted class label for each sample.
        """
        X = np.asarray(X)
        X_projected = self.transform(X)
        
        y_pred = []
        for x in X_projected:
            # Calculate Euclidean distance to each projected centroid
            distances = [np.linalg.norm(x - centroid) for centroid in self.projected_centroids_]
            # Predict the class with the minimum distance
            y_pred.append(self.classes_[np.argmin(distances)])
            
        return np.array(y_pred)

    def predict_proba(self, X):
        """
        Estimate class probabilities based on softmax of negative Euclidean distances
        to the class centroids in the projected LDA space.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns:
        probas : array-like of shape (n_samples, n_classes)
            Estimated class probabilities.
        """
        X = np.asarray(X)
        X_projected = self.transform(X)
        
        probas = []
        for x in X_projected:
            distances = np.array([np.linalg.norm(x - centroid) for centroid in self.projected_centroids_])
            # Softmax of negative distances: higher distance -> lower probability
            # Subtract max to ensure numerical stability (standard softmax trick)
            neg_dist = -distances
            neg_dist_stable = neg_dist - np.max(neg_dist)
            exp_dist = np.exp(neg_dist_stable)
            probas.append(exp_dist / np.sum(exp_dist))
            
        return np.array(probas)
