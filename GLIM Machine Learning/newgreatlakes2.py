# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score



data = pd.read_csv("C:/Users/toham/Downloads/Case comps/GLIM roun 2/Casestudy_dataset.csv")

# Step 2: Basic Data Inspection
print("Dataset Information:")
print(data.info())
print("\nFirst 5 Rows of the Dataset:")
print(data.head())

# Task 1: Dataset Exploration and
#Plot Histograms (Sets of 4 features for better visualization)
features = data.columns[1:-1]  # Exclude 'index' and 'Result'
print("\nVisualizing feature distributions...")
for i in range(0, len(features), 4):
    plt.figure(figsize=(15, 10))
    for j in range(4):
        if i + j < len(features):
            plt.subplot(2, 2, j + 1)
            data[features[i + j]].hist(bins=20, edgecolor='black')
            plt.title(features[i + j])
    plt.tight_layout()
    plt.show()

# correlation heatmap
plt.figure(figsize=(15, 12))
correlation_matrix = data.drop(columns=['index', 'Result']).corr()  # Safer approach
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap (Original Data)")
plt.show()



#Determine the number of samples and unique elements in each feature
print("\nNumber of Samples in the Dataset:", len(data))
print("\nUnique Elements in Each Feature:")
print(data.nunique())

#Check for null values
print("\nChecking for Null Values:")
print(data.isnull().sum())

# task 2.1 pre processing
# Remove irrelevant columns
data_cleaned = data.drop(columns=['index']).reset_index(drop=True)

# Reclassify the target variable
data_cleaned['Result'] = data_cleaned['Result'].apply(lambda x: 0 if x in [-1, 0] else 1)

# Check the target variable distribution
print("\nClass Distribution after Reclassification:")
print(data_cleaned['Result'].value_counts())

# Identify and remove highly correlated features
correlation_matrix_cleaned = data_cleaned.drop(columns=['Result']).corr()
high_correlation = correlation_matrix_cleaned[(correlation_matrix_cleaned > 0.85) & (correlation_matrix_cleaned < 1)].stack().reset_index()
high_correlation.columns = ['Feature1', 'Feature2', 'Correlation']
print("\nHighly Correlated Features (Threshold > 0.85):")
print(high_correlation)

# Remove one of the highly correlated features (e.g., 'popUpWidnow')
data_cleaned = data_cleaned.drop(columns=['popUpWidnow'])

#  Data Splitting
X = data_cleaned.drop(columns=['Result'])
y = data_cleaned['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TASK 2.2:  Model Training and Evaluation

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Model Evaluation
models = {
    "Random Forest": rf_model,
    "Decision Tree": dt_model
}

roc_curves = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, roc_auc)

#Plot all ROC Curves together for comparison
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curves for All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# K-Fold Cross-Validation
def cross_validate_model(model, X, y, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

print("\nPerforming K-Fold Cross-Validation...")
for name, model in models.items():
    mean_accuracy, std_dev = cross_validate_model(model, X, y)
    print(f"{name} - Mean Accuracy: {mean_accuracy:.4f}, Std Dev: {std_dev:.4f}")

#################################################################################


# Function to perform K-Fold Cross-Validation and display detailed results
def perform_kfold_validation(model, X, y, folds=5):
    print(f"\nPerforming {folds}-Fold Cross-Validation for {type(model).__name__}...\n")

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # Variables to track results
    fold = 1
    accuracies = []

    # Perform cross-validation
    for train_index, val_index in skf.split(X, y):
        # Split data into training and validation sets
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Train the model on training data
        model.fit(X_train, y_train)

        # Validate the model on validation data
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

        # Print results for the fold
        print(f"Fold {fold}: Accuracy = {accuracy:.4f}")
        fold += 1

    # Print overall results
    mean_accuracy = sum(accuracies) / len(accuracies)
    std_dev = (sum((x - mean_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5
    print(f"\n{type(model).__name__} - Mean Accuracy: {mean_accuracy:.4f}, Standard Deviation: {std_dev:.4f}\n")


# Apply K-Fold Cross-Validation to each model
for name, model in models.items():
    perform_kfold_validation(model, X, y, folds=5)

