import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd

def circle(Cx, Cy, R):
    theta = np.arange(0, 360, 1)
    theta = np.radians(theta)

    x = Cx + R * np.cos(theta)
    y = Cy + R * np.sin(theta)
    return np.column_stack((x, y))

def noisy_circle(data, mean_vector, R, amp_factor=1):
    '''Adds gaussian noise (independent) to a 2d circle. Note that there is no cross-covariance.
    Variance is 0.05 * R'''
    rng = np.random.default_rng()

    mu = np.array(mean_vector)
    cov = amp_factor * np.array([[R, 0], [0, R]])
    noise2d = rng.multivariate_normal(mu, cov, size=len(data))

    return data + noise2d

def sse(truths, preds):
    error_x = truths[:, 0] - preds[:, 0]
    error_y = truths[:, 1] - preds[:, 1]

    sse_val = np.sum(error_x**2 + error_y**2)
    return round(sse_val, 3)

def Jacobian(truths, preds, Cx, Cy):
    error_x = truths[:, 0] - preds[:, 0]
    error_y = truths[:, 1] - preds[:, 1]

    Cx_grad = -2 * np.sum(error_x)
    Cy_grad = -2 * np.sum(error_y)
    
    thetas = np.arctan2((preds[:, 1] - Cy), (preds[:, 0] - Cx))
    R_grad = -2 * np.sum(np.cos(thetas) * error_x + np.sin(thetas) * error_y)

    return [Cx_grad, Cy_grad, R_grad]

def Hessian(truths, preds, Cx, Cy):
    thetas = np.arctan2((preds[:, 1] - Cy), (preds[:, 0] - Cx))

    H = np.zeros((3, 3))

    H[0,0] = H[1,1] = H[2,2] = 2 * len(truths)
    H[0,1] = H[1,0] = 0
    H[0,2] = H[2,0] = 2 * np.sum(np.cos(thetas))
    H[1,2] = H[2,1] = 2 * np.sum(np.sin(thetas))

    return H

def newtons_method_circle(observed_data, X0, max_iter=20, tol=1e-6):
    X = X0 # X = [Cx, Cy, R]
    preds = circle(X[0], X[1], X[2])
    SSEs = []

    for i in range(max_iter):
        gradient = Jacobian(observed_data, preds, X[0], X[1])
        H = Hessian(observed_data, preds, X[0], X[1])

        deltas = np.linalg.inv(H).dot(gradient)
        X -= deltas

        preds = circle(X[0], X[1], X[2])
        SSE = sse(observed_data, preds)

        SSEs.append(SSE)
        if np.linalg.norm(deltas) < tol:
            print(f"Stopped in {i} steps.")
            break

    return X.tolist(), SSEs

def steepest_descent(observed_data, X0, max_iter=20, tol=1e-6):
    X = X0 # X = [Cx, Cy, R]
    preds = circle(X[0], X[1], X[2])
    SSEs = []

    for i in range(max_iter):
        gradient = np.array(Jacobian(observed_data, preds, X[0], X[1]))
        H = Hessian(observed_data, preds, X[0], X[1])

        alpha = (gradient.T.dot(gradient)/ np.dot(gradient.T, H.dot(gradient)))
        deltas = alpha * gradient
        X -= deltas

        preds = circle(X[0], X[1], X[2])
        SSE = sse(observed_data, preds)

        SSEs.append(SSE)
        if np.linalg.norm(deltas) < tol:
            print(f"Stopped in {i} steps.")
            break

    return X.tolist(), SSEs

def main(true_vector, guess_vector, noise_mean=[0,0], noise_variance=None):
    Cx, Cy, R = true_vector
    Cx0, Cy0, R0 = guess_vector

    # generate observations
    data = circle(Cx, Cy, R)
    noisy_data = noisy_circle(data, [0, 0], (R if noise_variance is None else noise_variance))

    # perform optimization
    X_opt_Newton, SSEs_Newton = newtons_method_circle(noisy_data, [Cx0, Cy0, R0])
    X_opt_SD, SSEs_SD = steepest_descent(noisy_data, [Cx0, Cy0, R0])

    # Plot true circle and observations
    plt.figure(figsize=(8,8))
    plt.scatter(data[:, 0], data[:, 1], color='k', s = 15)
    plt.title("True data")

    plt.figure(figsize=(8,8))
    plt.scatter(noisy_data[:, 0], noisy_data[:, 1], color='k', s = 15)
    plt.title("Noisy data")

    # Display results
    # Part 1: Newton's method
    print(f"True values: Cx, Cy, R = {Cx, Cy, R}")
    print(f"Initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}")

    print("Newton's method results:")
    print(f"Final X: Cxf, Cyf, Rf= {X_opt_Newton[0], X_opt_Newton[1], X_opt_Newton[2]},\nFinal SSE: {SSEs_Newton[-1]}")

    plt.figure()
    plt.plot(SSEs)
    plt.xlabel("iteration")
    plt.ylabel("SSE")
    plt.title(f"Newton's Method\nSSE for initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}")

    # plot final circle if feasible
    final_circle = circle(X_opt_Newton[0], X_opt_Newton[1], X_opt_Newton[2])

    plt.figure(figsize=(8,8))
    plt.scatter(noisy_data[:, 0], noisy_data[:, 1], color='k', s = 15)
    plt.scatter(final_circle[:, 0], final_circle[:, 1])
    plt.title(f"Newton's method\nFinal circle for initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}")

    print("----------------END OF NEWTON'S METHOD---------------")
    # (c) Part 2: Steepest Descent
    print("Steepest Descent results:")
    print(f"Final X: Cxf, Cyf, Rf= {X_opt_SD[0], X_opt_SD[1], X_opt_SD[2]},\nFinal SSE: {SSEs_SD[-1]}")

    plt.figure()
    plt.plot(SSEs)
    plt.xlabel("iteration")
    plt.ylabel("SSE")
    plt.title(f"Steepest Descent\nSSE for initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}")

    # plot final circle if feasible
    final_circle = circle(X_opt_SD[0], X_opt_SD[1], X_opt_SD[2])

    plt.figure(figsize=(8,8))
    plt.scatter(noisy_data[:, 0], noisy_data[:, 1], color='k', s = 15)
    plt.scatter(final_circle[:, 0], final_circle[:, 1])
    plt.title(f"Steepest Descent\nFinal circle for initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}")

    plt.close()
    return X_opt_Newton, X_opt_SD, SSEs_Newton, SSEs_SD

def part_d():
    # Experiment with variance of noise
    print("-"*50)
    print("Experiment varying the variance of noise")

    # True parameters
    Cxs = [0, 50, 1000]
    Cys = [0, 100, 0]
    R = [2, 1000, 10000]

    truths = list(zip(Cxs, Cys, R))


    # Initial guesses
    Cx0s = [Cxs[0] + 0, Cxs[1] + 1000, Cxs[2] + 10000]
    Cy0s = [Cys[0] + 0, Cys[1] + 1000, Cys[2] + 10000]
    R0s = [R[0] + 0, R[1] + 1000, R[2] + 10000]

    guesses = list(zip(Cx0s, Cy0s, R0s))

    results = []
    for i in range(3):
        Cx, Cy, r = truths[i]
        Cx0, Cy0, r0 = guesses[i]
        mean=[Cx, Cy]

        X_opt_Newton, X_opt_SD, SSEs_Newton, SSEs_SD = main(
            true_vector=[Cx, Cy, r],
            guess_vector=[Cx0, Cy0, r0],
            noise_mean=mean
            )
        
        # Log results from Newton's method
        results.append({
            'Method': "Newton's Method",
            'X_true': (Cx, Cy, r),
            'X_initial': (Cx0, Cy0, r0),
            'X_final': X_opt_Newton,
            'Noise (mean, variance)': (tuple(mean), r),
            'Iterations': len(SSEs_Newton),
            'Final SSE': SSEs_Newton[-1]
        })

        # Log results from Steepest Descent
        results.append({
            'Method': "Steepest Descent",
            'X_true': (Cx, Cy, r),
            'X_initial': (Cx0, Cy0, r0),
            'X_final': X_opt_SD,
            'Noise (mean, variance)': (tuple(mean), r),
            'Iterations': len(SSEs_SD),
            'Final SSE': SSEs_SD[-1]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/variance_exp_results.csv", float_format='%.3f', index=False)
        
    # Experiment with shifted Noise
    print("-"*50)
    print("Experiment with shifted noise")

    means = [[0,0], [10,10], [100,100]]
    Cx, Cy, R = 100, 100, 100
    Cx0, Cy0, R0 = 1000, 1000, 1000
    variance = 5

    results = []
    for i in range(len(means)):
        X_opt_Newton, X_opt_SD, SSEs_Newton, SSEs_SD = main(
            true_vector=[Cx, Cy, R],
            guess_vector=[Cx0, Cy0, R0],
            noise_mean=means[i],
            noise_variance=variance
        )

        results.append({
            'Method': "Newton's Method",
            'X_true': (Cx, Cy, r),
            'X_initial': (Cx0, Cy0, r0),
            'X_final': X_opt_Newton,
            'Noise (mean, variance)': (tuple(means[i]), variance),
            'Iterations': len(SSEs_Newton),
            'Final SSE': SSEs_Newton[-1]
        })

        # Log results from Steepest Descent
        results.append({
            'Method': "Steepest Descent",
            'X_true': (Cx, Cy, r),
            'X_initial': (Cx0, Cy0, r0),
            'X_final': X_opt_SD,
            'Noise (mean, variance)': (tuple(means[i]), variance),
            'Iterations': len(SSEs_SD),
            'Final SSE': SSEs_SD[-1]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/mean_exp_results.csv", float_format='%.3f', index=False)



if __name__ == "__main__":

    # (a)
    Cx, Cy, R = 100, 345, 29
    data = circle(Cx, Cy, R)

    fig = plt.figure(figsize=(8,8))

    plt.scatter(data[:, 0], data[:, 1], color='k', s = 15)
    plt.title("True data")

    # (b)
    rng = np.random.default_rng()

    mu = np.array([0, 0])
    cov = 0.05 * np.array([[R, 0], [0, R]])
    noise2d = rng.multivariate_normal(mu, cov, size=len(data))

    noisy_data = data + noise2d

    fig = plt.figure(figsize=(8,8))

    plt.scatter(noisy_data[:, 0], noisy_data[:, 1], color='k', s = 15)
    plt.title("Noisy data")

    # (c) Part 1: Newton's method
    Cx0, Cy0, R0 = 10000000, -1000, 4000
    X_optimized, SSEs = newtons_method_circle(noisy_data, [Cx0, Cy0, R0])

    print("Newton's method results:")
    print(f"True values: Cx, Cy, R = {Cx, Cy, R}")
    print(f"Initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}\nFinal X: Cxf, Cyf, Rf= {X_optimized[0], X_optimized[1], X_optimized[2]},\nFinal SSE: {SSEs[-1]}")

    plt.figure()
    plt.plot(SSEs)
    plt.xlabel("iteration")
    plt.ylabel("SSE")
    plt.title(f"SSE for initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}")

    # plot final circle if feasible
    final_circle = circle(X_optimized[0], X_optimized[1], X_optimized[2])

    plt.figure(figsize=(8,8))
    plt.scatter(noisy_data[:, 0], noisy_data[:, 1], color='k', s = 15)
    plt.scatter(final_circle[:, 0], final_circle[:, 1])
    plt.title(f"Newton's method\nFinal circle for initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}")

    # (c) Part 2: Steepest Descent
    Cx0, Cy0, R0 = 10000000, -1000, 4000
    X_optimized, SSEs = steepest_descent(noisy_data, [Cx0, Cy0, R0])

    print("Steepest Descent results:")
    print(f"True values: Cx, Cy, R = {Cx, Cy, R}")
    print(f"Initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}\nFinal X: Cxf, Cyf, Rf= {X_optimized[0], X_optimized[1], X_optimized[2]},\nFinal SSE: {SSEs[-1]}")

    plt.figure()
    plt.plot(SSEs)
    plt.xlabel("iteration")
    plt.ylabel("SSE")
    plt.title(f"SSE for initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}")

    # plot final circle if feasible
    final_circle = circle(X_optimized[0], X_optimized[1], X_optimized[2])

    plt.figure(figsize=(8,8))
    plt.scatter(noisy_data[:, 0], noisy_data[:, 1], color='k', s = 15)
    plt.scatter(final_circle[:, 0], final_circle[:, 1])
    plt.title(f"Steepest Descent\nFinal circle for initial X: Cx0, Cy0, R0 = {Cx0, Cy0, R0}")

    plt.show()

    # (d)
    # part_d()
         