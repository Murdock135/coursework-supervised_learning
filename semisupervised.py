import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import f1_score, accuracy_score
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from datetime import datetime


def load_data(mat_file):
    # Shape of data in the .mat files: (2, 75)
    Y1 = np.array(mat_file['Y1'])
    Y2 = np.array(mat_file['Y2'])

    # restructure training data
    y1_data = np.stack((Y1[0], Y1[1]), axis=1)
    y2_data = np.stack((Y2[0], Y2[1]), axis=1)

    # create labels
    y1_labels = np.ones(y1_data.shape[0], dtype=int)
    y2_labels = np.zeros(y2_data.shape[0], dtype=int)

    return np.vstack((y1_data, y2_data)), np.hstack((y1_labels, y2_labels))

def unlabel_data(data, ul_portion=0.8, seed=0):
    X, y = data
    X, y = X.copy(), y.copy()

    # Randomly obtain indices that are to be unlabeled
    seed = seed
    rng = np.random.default_rng(seed)
    n_unlabel = int(ul_portion * y.shape[0])
    unlabel_indices = rng.choice(y.shape[0], n_unlabel, replace=False)

    # Store true labels
    ground_truth = y[unlabel_indices].copy()

    # Set proxy label (0) for unlabeled data
    y[unlabel_indices] = -1
    
    # Get unlabeled input data
    X_unlabeled = X[unlabel_indices]

    return X_unlabeled, y, ground_truth

def add_noise(X_unlabeled, mu=0.0, var=1.0, amp_factor=1.0):
    rng = np.random.default_rng()

    mu = np.array([mu, mu])
    cov = amp_factor * np.array([[var, 0], [0, var]])
    noise2d = rng.multivariate_normal(mu, cov, size=len(X_unlabeled))

    return X_unlabeled + noise2d

def plot_semisupervised_results(results_df, noise_var_vals, save_path=None):    
    required_cols = ['threshold', 'noise_var', '#_of_labeled_samples', 'accuracy', 'n_iter']
    if not all(col in results_df.columns for col in required_cols):
        raise ValueError(f"Results DataFrame missing required columns: {required_cols}")

    # Clear any existing plots
    plt.close('all')
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Threshold vs Accuracy and Labeled Samples
    ax1_twin = ax1.twinx()  # Create twin axis for second y-axis
    
    for noise_var in noise_var_vals:
        subset = results_df[results_df['noise_var'] == noise_var]
        # Plot accuracy
        line1 = ax1.plot(subset['threshold'], subset['accuracy'], 
                marker='o', linestyle='-', label=f'Accuracy (Noise={noise_var})')
        # Plot number of labeled samples
        line2 = ax1_twin.plot(subset['threshold'], subset['#_of_labeled_samples'], 
                   marker='s', linestyle='--', label=f'#Labeled (Noise={noise_var})')
    
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Accuracy')
    ax1_twin.set_ylabel('Number of Labeled Samples')
    ul_prop = results_df['unlabel_proportion'].iloc[0]  # Get the unlabel proportion for this subset
    ax1.set_title(f'Threshold vs Accuracy and Labeled Samples\n(Unlabeled Proportion: {ul_prop:.1f})')
    
    # Combine legends from both y-axes and place below the plot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Plot 2: Threshold vs Iterations
    for noise_var in noise_var_vals:
        subset = results_df[results_df['noise_var'] == noise_var]
        ax2.plot(subset['threshold'], subset['n_iter'], 
                marker='o', label=f'Noise={noise_var}')
    
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Number of Iterations')
    ax2.set_title(f'Threshold vs Number of Iterations\n(Unlabeled Proportion: {ul_prop:.1f})')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Adjust layout with more space at bottom for legends
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legends
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_ulprop_vs_accuracy(results_df, noise_var_vals, save_path=None):
    plt.figure(figsize=(10, 6))
    
    for noise_var in noise_var_vals:
        # Get the best accuracy for each ul_proportion at this noise level
        subset = results_df[results_df['noise_var'] == noise_var]
        ul_props = subset['unlabel_proportion'].unique()
        best_accuracies = [
            subset[subset['unlabel_proportion'] == prop]['accuracy'].max()
            for prop in ul_props
        ]
        
        plt.plot(ul_props, best_accuracies, 
                marker='o', linestyle='-', 
                label=f'Noise Var={noise_var}')
    
    plt.xlabel('Initial Unlabeled Proportion')
    plt.ylabel('Best Accuracy')
    plt.title('Unlabeled Proportion vs Best Accuracy\nfor Different Noise Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Generate synthetic data using blobs instead of moons
    X_train, y_train = make_blobs(n_samples=800, 
                                 centers=2,  # 2 clusters
                                 n_features=2,  # 2D data
                                 cluster_std=2.0,  # Controls overlap
                                 random_state=42)
    X_test, y_test = make_blobs(n_samples=200, 
                               centers=2, 
                               n_features=2, 
                               cluster_std=2.0, 
                               random_state=43)
    
    # Plot the training data
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='viridis')
    plt.title("Training Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.subplot(122)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='viridis')
    plt.title("Test Data") 
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.tight_layout()
    plt.show()

    # Create classifier
    SVM = SVC(probability=True, gamma=0.001, random_state=42)

    # Parameters for grid search
    noise_mu = 0
    noise_var_vals = [1, 5, 10, 20]
    thresh_vals = np.linspace(0.4, 0.999, 10)
    unlabel_proportions = np.arange(0.1, 0.8, 0.2)  # Added different proportions

    # Create cartesian product of all parameter values
    param_combinations = list(product(thresh_vals, noise_var_vals, unlabel_proportions))

    # Run a selftraining algorithm
    results = []
    for threshold, noise_var, ul_proportion in tqdm(param_combinations):
        # Create noisy input and unlabel with varying proportion
        X_unlabeled, y, _ = unlabel_data((X_train, y_train), ul_portion=ul_proportion)
        X_unlabeled_noisy = add_noise(X_unlabeled, noise_mu, noise_var)

        # Get labelled data
        X_labeled = X_train[y != -1]
        y_labeled = y_train[y != -1]

        # Create selftrainer
        self_training_clf = SelfTrainingClassifier(base_estimator=SVM, threshold=threshold, verbose=True)

        # Combine labelled and unlabeled data
        X_combined = np.vstack((X_labeled, X_unlabeled_noisy))
        y_combined = y
        assert X_combined.shape[0] == y_combined.shape[0], "length of X_combined and y_combined should be equal"

        # Train
        self_training_clf.fit(X_combined, y_combined)
        
        # Validate
        y_preds = self_training_clf.predict(X_test)

        # Record scores
        accuracy = accuracy_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds)
        n_labeled = X_combined.shape[0] - np.unique(self_training_clf.labeled_iter_, return_counts=True)[1][0]

        results.append({
            'threshold': threshold,
            'noise_var': noise_var,
            'unlabel_proportion': ul_proportion,  # Added to results
            '#_of_labeled_samples': n_labeled,
            'accuracy': accuracy,
            'f1_score': f1,
            'n_iter': self_training_clf.n_iter_
        })

    # Convert results list to a pd.Dataframe
    results_df = pd.DataFrame(results)

    # Define base save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = os.path.join(
        r"C:\Users\Zayan\Documents\code\personal_repos\mizzou_stuff\supervised_learning\homework_3_results",
        f"run_{timestamp}"
    )
    
    # Create directory if it doesn't exist
    os.makedirs(base_save_path, exist_ok=True)

    # Plot results for each unlabel proportion
    for ul_prop in unlabel_proportions:
        results_subset = results_df[results_df['unlabel_proportion'] == ul_prop]
        
        # Create individual file path for each plot
        file_name = f'semisupervised_results_ulprop_{ul_prop:.1f}.png'
        plot_save_path = os.path.join(base_save_path, file_name)
        
        plot_semisupervised_results(results_subset, noise_var_vals, plot_save_path)
        plt.show()

    # After your existing plotting code, add:
    ulprop_plot_path = os.path.join(base_save_path, 'ulprop_vs_accuracy.png')
    plot_ulprop_vs_accuracy(results_df, noise_var_vals, ulprop_plot_path)
    plt.show()

    
    