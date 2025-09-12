# %% [markdown]
# # CA2 - Supervised machine learning classification pipeline - applied to medical data

# %% [markdown]
# ## Part I: Data loading and data exploration

# %%
# Import necessary libraries/modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.classifier import Adaline, LogisticRegression, Perceptron


# %%
# Load the dataset and check for missing data
def load_data():
    """
    Load the fetal health dataset from CSV file and perform basic data validation.

    Loads the dataset from 'assets/fetal_health.csv', checks for missing values,
    and removes any rows containing missing data.

    Returns:
        pd.DataFrame or None: The loaded and cleaned dataset, or None if loading fails

    Prints:
        Status messages about data loading and missing value handling
    """
    try:
        df = pd.read_csv("assets/fetal_health.csv", index_col=0)
        if df is not None and not df.empty:
            print("data loaded")
        else:
            print("data did not load")
            return None
    except Exception:
        print("data did not load")
        return None

    print(df)

    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values. Removing rows with missing data.")
        df.dropna(inplace=True)
    else:
        print("No missing values found.")

    return df


# Call the load_data function and assign result to df
df = load_data()

# %%
# Distribution of baseline value
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, ["baseline value", "accelerations", "fetal_health"]):
    if col == "fetal_health":
        sns.countplot(x=df["fetal_health"], ax=ax)  # Bruker countplot for kategoriske data
    else:
        sns.histplot(df[col], bins=20, kde=True, ax=ax)  # Bruker histplot for numeriske data

    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
plt.tight_layout()  # sørger for at figurene ikke overlapper.
plt.show()


"""
Since "baseline value" and "accelerations" are continuous, we used histograms from the Seaborn package to visualize their distributions. This allows us to quickly assess whether the data follows a normal distribution or is skewed (left or right). A statistical summary (such as mean, median, and percentiles) could also be used, but for an initial overview, histograms provided us with an intuitive representation.

For "fetal_health," which is a categorical variable, we used a count plot to observe the class distribution. This helps in identifying class imbalances, which could be important for further analysis.
"""


# %%
# Scatterplot to assess linear separability
sns.scatterplot(data=df, x="baseline value", y="accelerations", hue="fetal_health")


"""
Scaling the data is beneficial because models like Adaline and Logistic Regression use gradient descent, which converges more efficiently when features are on a similar scale. In this dataset, "baseline value" is measured in beats per minute (bpm), while "accelerations" has a different numerical range. Standardizing the features ensures that all variables contribute equally to the model and prevents any single feature from dominating due to its scale.

However, scaling fetal_health is not necessary because it is a categorical variable representing class labels. Scaling applies only to continuous numerical features, not to categorical labels.

Based on the scartterplot above, the points are overlapping significantly and therefore does not show features of bein linearly separable. Since the data points are not clearly divided by a stright line or plane, we cannot therefore not expect an accuracuy close to 100% from the lienar classifier. Therefore a more complex model such as a nonlinear model may be needed.

"""

# %% [markdown]
# ## Part II: Train/Test Split

# %%
# Creating DataFrames for different fetal health categories
df_0 = df[df["fetal_health"] == 0]
df_1 = df[df["fetal_health"] == 1]

# Split into training and test set by randomly sampling entries from the data frames
df_0_train = df_0.sample(frac=0.75, random_state=42)
df_1_train = df_1.sample(frac=0.75, random_state=42)
df_0_test = df_0.drop(df_0_train.index)
df_1_test = df_1.drop(df_1_train.index)

# Merge the datasets split by classes back together with concat
df_train = pd.concat([df_0_train, df_1_train])
df_test = pd.concat([df_0_test, df_1_test])

# Create dataframes of df_train and df_test without "fetal_health"
X_train = df_train.drop(columns=["fetal_health"], errors="ignore")
X_test = df_test.drop(columns=["fetal_health"], errors="ignore")
y_train = df_train["fetal_health"]
y_test = df_test["fetal_health"]

# Check shapes on datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# %%
"""

If we randomly divide the dataset directly into training and test sets, we may get a skewed distribution, where one class dominates the training set, while another dominates the test set.
To avoid imbalance in the classes, we first divided the data into each class (df_0 and df_1).

After we had separate DataFrames for each class, we could ensure that we take the same proportion of each class to the training and test sets.
We took 75% of each class to the training set (df_0_train and df_1_train).
The remaining 25% became the test set (df_0_test and df_1_test).

Once we have balanced training and test sets for each class, we can combine them (df_train and df_test).
This ensures that the training and test sets have the same class distribution as the original dataset.
This provides a more reliable evaluation of the model's performance.

Main points:
Prevents imbalance in training and test sets.
Ensures that both classes are represented correctly in both datasets.
Provides a more realistic model evaluation, because the model is not affected by a skewed class distribution.

"""


# %%
# Calculate class 0 percentage in different sets
def calculate_class_percentage(df, total):
    """
    Calculate the percentage of samples in a DataFrame relative to total samples.

    Args:
        df (pd.DataFrame): DataFrame containing the subset of samples
        total (int): Total number of samples in the complete dataset

    Returns:
        float: Percentage of samples (0-100), or 0 if total is 0
    """
    return (len(df) / total) * 100 if total > 0 else 0


initial_class_0_percentage = calculate_class_percentage(df_0, len(df))
train_class_0_percentage = calculate_class_percentage(df_0_train, len(df_train))
test_class_0_percentage = calculate_class_percentage(df_0_test, len(df_test))

print(f"Percentage of class 0 in initial dataset: {initial_class_0_percentage:.2f}%")
print(f"Percentage of class 0 in training set: {train_class_0_percentage:.2f}%")
print(f"Percentage of class 0 in test set: {test_class_0_percentage:.2f}%")

# %%
# Convert data to numpy arrays and shuffle the training data
# This is important because the training data is currently ordered by class.
# If we did not shuffle the data, the classifiers would only be trained on samples of class 0.
X_train_columns = X_train.columns
X_train_1 = X_train.to_numpy()
X_test_1 = X_test.to_numpy()
y_train_1 = y_train.to_numpy()
y_test_1 = y_test.to_numpy()
np.random.seed(42)
shuffle_index = np.random.permutation(len(X_train_1))
X_train_1, y_train_1 = (X_train_1[shuffle_index], y_train_1[shuffle_index])

# %% [markdown]
# ## Part III: Scaling the data


# %%
# Standardize the data using mean and standard deviation from training set
def Xscale(x, X_train):
    """
    Standardize features using Z-score normalization based on training data statistics.

    Applies standardization: (x - mean) / std_dev using the mean and standard deviation
    computed from the training data to avoid data leakage.

    Args:
        x (np.ndarray): Feature matrix to be scaled (can be training or test data)
        X_train (np.ndarray): Training feature matrix used to compute scaling parameters

    Returns:
        np.ndarray: Standardized feature matrix with mean=0 and std=1 based on training stats

    Note:
        Always use training data statistics for scaling both training and test sets
        to prevent data leakage and ensure consistent scaling.
    """
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    return (x - X_mean) / X_std


# Apply scaling - important: use training data statistics to avoid data leakage
X_train_scaled = Xscale(X_train_1, X_train_1)
X_test_scaled = Xscale(X_test_1, X_train_1)

# Check that the scaling was successful
print("Mean of scaled x features:", np.mean(X_train_scaled, axis=0))
print("\nStandard deviation of scaled x features:", np.std(X_train_scaled, axis=0))
df_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_columns)

# %%
# Violin plot to visualize scaled data distribution
# Convert to DataFrame
df_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_columns)

# Create a violin plot to visualize the distribution of scaled features
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_train_scaled, inner="quartile")
plt.xticks(rotation=90)
plt.title("Violin plot of scaled training data")
plt.show()

# %% [markdown]
# ## Part IV: Training and evaluation with different dataset sizes and training times

# %%
# Train models with different dataset sizes and epochs
dataset_sizes = list(range(50, 701, 50))
num_epochs = list(range(2, 98, 5))
classifiers = {"Perceptron": Perceptron, "Adaline": Adaline, "LogisticRegression": LogisticRegression}
results = np.zeros((len(classifiers), len(dataset_sizes), len(num_epochs)))

for c_idx, (clf_name, clf_class) in enumerate(classifiers.items()):
    print(f"Training {clf_name}...")
    for _i, size in enumerate(dataset_sizes):
        X_train_subset = X_train_scaled[:size]
        y_train_subset = y_train_1[:size].ravel()
        for _j, epochs in enumerate(num_epochs):
            if clf_class in [Adaline, LogisticRegression]:
                _model = clf_class(eta=0.0001, epochs=epochs, minibatches=1, random_seed=42)
            else:
                _model = clf_class(eta=0.0001, epochs=epochs, random_seed=42)
            _model.fit(X_train_subset, y_train_subset)
            _y_pred = _model.predict(X_test_scaled)
            accuracy = np.mean(_y_pred == y_test_1)
            results[c_idx, _i, _j] = accuracy

index_labels = ["Perceptron", "Adaline", "LogisticRegression"]
df_dict = {index_labels[_i]: pd.DataFrame(results[_i], index=dataset_sizes, columns=num_epochs) for _i in range(3)}
print("done")

# %%
# Performance visualization with heatmaps
_fig, _axes = plt.subplots(2, 2, figsize=(14, 12))
_axes = _axes.flatten()
for _i, (_model, df_2) in enumerate(df_dict.items()):
    sns.heatmap(df_2, annot=False, cmap="gist_heat", ax=_axes[_i])
    _axes[_i].set_title(f"{_model} Accuracy")
    _axes[_i].set_xlabel("Number of Epochs")
    _axes[_i].set_ylabel("Dataset Size")
_axes[3].set_visible(False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %% [markdown]
# ## Part V: Decision boundary visualization

# %%
# Train the classifier on training data
clf = LogisticRegression(eta=0.0001, epochs=300, minibatches=1, random_seed=42)
clf.fit(X_train_scaled, y_train_1)

# Create 8x8 subplot grid to show decision boundaries for all feature pairs
_fig, _axes = plt.subplots(8, 8, figsize=(24, 24))
for _i in range(0, 8):
    for _j in range(0, 8):
        feature_1 = _i
        feature_2 = _j
        _ax = _axes[_i, _j]
        _ax.set_xlabel(f"Feature {feature_1}", fontsize=8)
        _ax.set_ylabel(f"Feature {feature_2}", fontsize=8)
        _ax.tick_params(labelsize=6)

        # Use training data bounds for consistent scaling
        mins = X_train_scaled.min(axis=0)
        maxs = X_train_scaled.max(axis=0)
        x0 = np.linspace(mins[feature_1], maxs[feature_1], 100)
        x1 = np.linspace(mins[feature_2], maxs[feature_2], 100)
        X0, X1 = np.meshgrid(x0, x1)
        X_two_features = np.column_stack((X0.ravel(), X1.ravel()))

        # Create feature matrix with zeros and fill in the two selected features
        X_plot = np.zeros(shape=(X_two_features.shape[0], X_test_scaled.shape[1]))
        X_plot[:, feature_1] = X_two_features[:, 0]
        X_plot[:, feature_2] = X_two_features[:, 1]

        # Predict probabilities and create decision boundary
        _y_pred = clf.predict_proba(X_plot)
        Z = _y_pred.reshape(X0.shape)
        _ax.pcolor(X0, X1, Z, alpha=0.8)
        _ax.contour(X0, X1, Z, levels=[0.5], colors="k", linewidths=1)

        # Plot test data points
        _ax.scatter(
            X_test_scaled[y_test_1 == 0, feature_1],
            X_test_scaled[y_test_1 == 0, feature_2],
            color="b",
            marker="^",
            s=30,
            facecolors="none",
            alpha=0.7,
        )
        _ax.scatter(
            X_test_scaled[y_test_1 == 1, feature_1],
            X_test_scaled[y_test_1 == 1, feature_2],
            color="y",
            marker="o",
            s=30,
            facecolors="none",
            alpha=0.7,
        )
_fig.tight_layout(pad=0.5)
plt.show()


# %%

"""

2. A:
The plots show decision boundaries and classification of test data for a Logistic Regression model trained on different pairs of features from the dataset. Each subplot visualizes how the model distinguishes between the classes based on two specific features.

3. A:
The contour line (decision boundary) shows where the model is uncertain (50% probability for class 0 or 1).
The yellow circles refer to class 0 and the blue triangles refer to class 1. Class 0 and 1 refer to whether the y-value (fetal health) is 0 or 1. If a blue triangle lies far inside the yellow area → The model has misclassified it. If a yellow circle lies far inside the blue area → Also misclassification.

X and Y coordinates come from the values of two features from the test set. If the points are close to the black decision boundary, it means that the model was unsure of the classification.

Disadvantages of the subplots: The training data (X_train_scaled) is not displayed, even though the model is trained on it. If the training data had been plotted, we could have seen if the model is overfitting the training data. We cannot see how the model has actually learned from the training set.
Another thing; the decision boundaries may be more complex in full dimension, but look simpler in 2D. A data point that looks misclassified in one combination of features may actually be correctly classified if we include more features.

"""

"""
## Part VI: Additional discussion

### Part I:
1. What kind of plots did you use to visualize the raw data, and why did you choose these types of plots?


A: Fetal health is displayed in a "count-plot" since it is a categorical variable (either 0 or 1). The others were shown in a histogram, which shows the distribution of a numerical variable by grouping data into "bins". It helps to understand how the data is distributed (normally distributed, skewed, or has multiple peaks).

### Part II:
1. What happens if we don't shuffle the training data before training the classifiers like in Part IV?

A: The model can learn the order instead of patterns. Shuffling ensures that different classes are mixed so that the model learns a balanced pattern.

2. How could you do the same train/test split (Point 1.-4.) using scikit-learn?

A: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42, shuffle=True)

### Part IV:
1. How does increasing the dataset size affect the performance of the logistic regression model? Provide a summary of your findings.

A: Based on the heatmap in task 4, we can see that accuracy increases with larger datasets. This is probably because more data allows the model to learn better decision boundaries and generalize better to new test data. The improvement decreases after about 500 samples.
After this point, more data does not provide a significant increase in performance. Then the model has already learned the most important patterns in the data.

2. Describe the relationship between the number of epochs and model accuracy

A: For all dataset sizes, accuracy increases with the number of epochs. With smaller datasets (50–200 samples), we observe greater fluctuations in accuracy depending on the number of epochs. This suggests that the model tends to overfit small datasets, where it "learns" the training data too well, but does not generalize well to the test set. Large datasets provide more stable accuracy, which means that the model becomes less sensitive to the number of epochs.

3. Which classifier is much slower to train and why do you think that is?

A: The Perceptron classifier is slower because it updates weights after every misclassified sample, leading to frequent recomputations and updates. Additonally, it only converges if the data is linearly separable; otherwise, it keeps adjusting the weights indefinitely. In contrast, Adaline and Logistic Regression use Gradient Descent, where weights are updated gradually based on a cost function, leading to fewer and more controlled updates.


4. One classifier shows strong fluctuations in accuracy for different dataset sizes and number of epochs. Which one is it and why do you think this happens?

A: It updates the weights only when a misclassification occurs (binary update rule, either misclassified or not). With small datasets, the Perceptron often learns poor and random decision boundaries.

The Perceptron only provides binary predictions (0 or 1) without using probabilities. This can result in highly variable accuracy, depending on how the training data is organized and how many misclassifications occur early in training.

If the data is not perfectly separable, the model will never converge, leading to unstable accuracy. This results in high variation in performance because the model is sensitive to the order of training data. If the data is not separable, it will never find a stable decision boundary

"""
