import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def train_and_predict_svm(
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    c_values: list[float] = [0.1, 1.0, 10.0, 50.0, 100.0],
    gamma_values: list[str | float] = ["scale", "auto", 0.1, 0.01, 0.001],
    target_labels: list[int] = [1, 2],
    other_label: int = 0,
    prob_threshold: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Trains an RBF SVM using GridSearch over the gallery embeddings,
    and returns the predicted classes for the query embeddings.
    """
    X_train = gallery_embeddings.cpu().numpy()
    y_train = gallery_labels.cpu().numpy()
    X_test = query_embeddings.cpu().numpy()

    # Use StratifiedKFold, with min splits based on smallest class count
    min_class_count = int(np.min(np.bincount(y_train)))
    n_splits = min(5, min_class_count) if min_class_count > 1 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_grid = {
        'C': c_values,
        'gamma': gamma_values,
    }

    base_svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(
        base_svm,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    
    best_svm = grid_search.best_estimator_
    cv_accuracy = grid_search.best_score_
    
    # Predict probabilities for queries
    probs = best_svm.predict_proba(X_test)
    classes = best_svm.classes_
    
    final_predictions = []
    
    # Custom thresholding logic based on probability
    for prob_row in probs:
        max_target_prob = -1.0
        max_target_label = other_label
        
        for i, class_label in enumerate(classes):
            if class_label in target_labels:
                if prob_row[i] > max_target_prob:
                    max_target_prob = prob_row[i]
                    max_target_label = class_label
                    
        # If the highest probability for a target class exceeds threshold, pick it. 
        # Otherwise, fall back to "other"
        if max_target_prob >= prob_threshold:
            final_predictions.append(max_target_label)
        else:
            final_predictions.append(other_label)
            
    final_predictions = torch.tensor(final_predictions, dtype=torch.long)
    
    metrics = {
        "best_C": grid_search.best_params_["C"],
        "best_gamma": grid_search.best_params_["gamma"],
        "cv_accuracy": cv_accuracy
    }
    
    return final_predictions, metrics
