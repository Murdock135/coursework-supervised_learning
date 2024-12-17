from scipy.io import loadmat
from scipy.optimize import minimize, LinearConstraint
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class SVM:
    def __init__(self, kernel=None, C=1.0):
        self.kernel = kernel
        self.C = C
        self.alpha = None # lagrange multipliers
        self.b = None
        self.support_vectors = None # support vectors
        self.support_labels = None # labels of support vectors
        self.support_alpha = None # lagrange multipliers of support vectors
        self.w = None

    def _gram_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))

        if self.kernel is None:
            return np.dot(X, X.T)
        else:
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = self.kernel(X[i], X[j])
            return K

    def solve(self, X, y):
        # Get number of samples
        n_samples = X.shape[0]

        # Compute the Gram matrix
        K = self._gram_matrix(X)

        # Define the objective function (dual problem)
        def objective(alpha):
            return 0.5 * np.sum(np.outer(alpha, alpha) * np.outer(y, y) * K) - np.sum(alpha)

        # Equality constraint: sum(alpha_i * y_i) = 0
        linear_constraint = LinearConstraint(y, [0], [0])

        # Bounding box constraint: 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(n_samples)]

        # Solve the quadratic problem
        initial_alpha = np.zeros(n_samples)
        result = minimize(objective, initial_alpha, bounds=bounds, constraints=[linear_constraint])

        # Check if the solver converged
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Extract the optimized alpha values
        self.alpha = result.x

        # Identify support vectors
        support_vector_indices = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_labels = y[support_vector_indices]
        self.support_alpha = self.alpha[support_vector_indices]

        # Compute the bias term b
        # TODO: vectorize
        b_vals = []
        K_support = K[support_vector_indices]
        for i in range(len(self.support_labels)):
            b = self.support_labels[i] - np.dot(self.alpha * y, K_support[i])
            b_vals.append(b)
        
        self.b = np.mean(b_vals)

        # If it's a linear kernel, compute the weight vector w
        if self.kernel is None:
            self.w = np.dot(self.alpha * y, X)

    def predict(self, X):
        if self.w is not None:
            # Linear case
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            # Non-linear kernel case
            K = np.array([[self.kernel(sv, x) for x in X] for sv in self.support_vectors])

            output_vals = np.dot((self.support_alpha * self.support_labels), K) + self.b

            predictions = np.sign(output_vals)
            return predictions

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=2, gamma=1, coef0=1):
    return (gamma * np.dot(x1, x2) + coef0) ** degree

def rbf_kernel(x1, x2, gamma=0.1):
    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-gamma * distance)

def plot_decision_boundary(svm, X, y, title=''):
    # Plot decision boundary for 2D data
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)

def load_data(mat_file):
    # Shape of data in the .mat files: (2, 75)
    Y1 = np.array(mat_file['Y1'])
    Y2 = np.array(mat_file['Y2'])

    # restructure training data
    y1_data = np.stack((Y1[0], Y1[1]), axis=1)
    y2_data = np.stack((Y2[0], Y2[1]), axis=1)

    # create labels
    y1_labels = np.ones(y1_data.shape[0], dtype=int)
    y2_labels = -np.ones(y2_data.shape[0], dtype=int)

    return np.vstack((y1_data, y2_data)), np.hstack((y1_labels, y2_labels))

if __name__ == "__main__":
    # load data
    train_file = loadmat("c:/Users/Zayan/Documents/code/personal_repos/mizzou_supervised_learning_8725/homework_2_data/training.mat")
    test_file = loadmat("c:/Users/Zayan/Documents/code/personal_repos/mizzou_supervised_learning_8725/homework_2_data/testing.mat")

    for key, val in train_file.items():
        print(f"key: {key}")
        print(f"value: {val}")

    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)

    print("Training data")
    print(X_train)

    plt.figure()
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='tab20b')
    plt.title("Training data")
    
    plt.figure()
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='tab20b')
    plt.title("Test data")

    # -------------------------------------------------------------------------------------
    # PART 1: My SVM
    # Linear SVM
    svm_linear_hard = SVM()
    svm_linear_hard.solve(X_train, y_train)

    # Make predictions on test data
    predictions_linear_hard = svm_linear_hard.predict(X_test)

    # Compute accuracy
    accuracy_linear = np.mean(predictions_linear_hard == y_test)
    print(f"Linear Kernel SVM Accuracy: {accuracy_linear * 100:.2f}%")

    # SVM with polynomial kernel
    degrees = [2, 3, 4]
    for d in degrees:
        svm_poly = SVM(kernel=lambda x1, x2: polynomial_kernel(x1, x2, degree=d), C=1.0)
        svm_poly.solve(X_train, y_train)

        # # Predict on the test set
        predictions_poly = svm_poly.predict(X_test)

        # # Compute accuracy
        accuracy_poly = np.mean(predictions_poly == y_test)
        print(f"Polynomial Kernel (degree {d}) SVM Accuracy: {accuracy_poly * 100:.2f}%")

        plot_decision_boundary(svm_poly, X_test, y_test, f'SVM using polynomial kernel of degree {d}')


    # SVM with RBF kernel
    gamma = 0.5
    svm_rbf = SVM(kernel=lambda x1, x2: rbf_kernel(x1, x2, gamma=0.5), C=1.0)
    svm_rbf.solve(X_train, y_train)

    # # Predict on the test set
    predictions_rbf = svm_rbf.predict(X_test)

    # # Compute accuracy
    accuracy_rbf = np.mean(predictions_rbf == y_test)
    print(f"RBF Kernel (gamma = {gamma}) SVM Accuracy: {accuracy_rbf * 100:.2f}%")

    # # Plot decision boundary
    plot_decision_boundary(svm_rbf, X_test, y_test, 'SVM using RBF kernel')
    plot_decision_boundary(svm_linear_hard, X_test, y_test, 'SVM using Linear kernel')

    # ---------------------------------------------------------------------------------------------------
    # PART 2: Scikit-learn's SVM
    print("-"*50)
    print("Sklearn SVM results:")
    # Linear kernel
    sklearn_svm_linear = svm.SVC(kernel='linear')
    sklearn_svm_linear.fit(X_train, y_train)

    # Predictions for Linear kernel
    predictions_sklearn_linear = sklearn_svm_linear.predict(X_test)
    accuracy_sklearn_linear = np.mean(predictions_sklearn_linear == y_test)
    print(f"Linear Kernel Accuracy: {accuracy_sklearn_linear * 100:.2f}%")

    # Polynomial kernel
    degrees = [2, 3, 4]
    for d in degrees:
        sklearn_svm_poly = svm.SVC(kernel='poly', degree=d, C=1.0)
        sklearn_svm_poly.fit(X_train, y_train)

        # Predictions for polynomial kernel
        predictions_sklearn_poly = sklearn_svm_poly.predict(X_test)
        accuracy_sklearn_poly = np.mean(predictions_sklearn_poly == y_test)
        print(f"Polynomial Kernel (degree {d}) Accuracy: {accuracy_sklearn_poly * 100:.2f}%")

        plot_decision_boundary(sklearn_svm_poly, X_test, y_test, f'Sklearn SVM using polynomial kernel of degree {d}')


    # RBF kernel
    gamma = 0.5
    sklearn_svm_rbf = svm.SVC(kernel='rbf', gamma=gamma, C=1.0)
    sklearn_svm_rbf.fit(X_train, y_train)

    # Predictions for RBF kernel
    predictions_sklearn_rbf = sklearn_svm_rbf.predict(X_test)
    accuracy_sklearn_rbf = np.mean(predictions_sklearn_rbf == y_test)
    print(f"RBF Kernel (gamma: {gamma}) Accuracy: {accuracy_sklearn_rbf * 100:.2f}%")

    # Plot decision boundaries for scikit-learn SVMs
    plot_decision_boundary(sklearn_svm_linear, X_test, y_test, 'Sklearn SVM using Linear kernel')
    plot_decision_boundary(sklearn_svm_rbf, X_test, y_test, 'Sklearn SVM using RBF kernel')

    plt.show()
