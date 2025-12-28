# LogisticRegressionFromScratch
I forecasted US recessions with my made from scratch logistic regression in python. I used gradient descent and newtons  method to achieve this.
Forecasting US Recessions: Implementing Regularized Logistic Regression from Scratch (Newton’s Method)
The reason I chose this project was to understand the basic algorithms in Machine learning to help me in the future to build more complex algorithms. I derived the Gradients and hessian Matrix to implement a custom optimizer that predicts recessions using the FRED real time data.
Optimization, testing both Gradient Descent and Newton optimization to figure out the best one and building off of each one to get the best result
Regularization, I initially coded the logistic regression without regularization but that limited the amount of indexes and use of lagged data together as there would be lots of multi-colinearity. Adding the regularization helped me add many more economic indicators improving my overall results.
Econometrics, Handling of non-stationary time series( ADF Testing, Log-Differenceing) and look ahead bias.

To ensure the model captures true economic signals rather than overfitting to noise (e.g., short-term market volatility), I implemented **L2-Regularized Logistic Regression** from first principles.

### The Hypothesis
The model outputs a probability using the Sigmoid activation function:
$$h_\theta(x) = \sigma(z) = \frac{1}{1 + e^{-\theta^T x}}$$

### The Objective Function (Regularized Log-Loss)
I minimized the binary cross-entropy loss augmented with an L2 (Ridge) penalty term. This prevents multicollinearity among economic indicators from exploding the feature weights.

$$J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

*Note: The regularization term starts at $j=1$, ensuring the bias term (intercept) is not penalized.*

### Optimization: Newton's Method
While standard Gradient Descent converges linearly, I implemented **Newton's Method** for quadratic convergence. This required deriving and computing the **Hessian Matrix** (the matrix of second-order derivatives) manually.

**The Gradient (First Derivative):**
$$\nabla J(\theta) = \frac{1}{m} X^T (h - y) + \frac{\lambda}{m} \theta_{reg}$$
*(Where $\theta_{reg}$ is the weight vector with the bias term set to 0).*

**The Hessian (Second Derivative):**
$$H = \frac{1}{m} X^T S X + \frac{\lambda}{m} I_{reg}$$
* $S$: A diagonal matrix where $S_{ii} = h^{(i)}(1 - h^{(i)})$
* $I_{reg}$: The Identity matrix with the top-left element ($0,0$) set to 0 to exclude the bias.

**The Update Rule:**
The weights are updated by inverting the Hessian, allowing the model to take a direct step towards the minimum:
$$\theta_{new} := \theta_{old} - H^{-1} \nabla J(\theta)$$

<img width="1363" height="658" alt="image" src="https://github.com/user-attachments/assets/f2555681-c88b-4815-8be9-d182a2f08056" />
In this graph we can see that my alogrithim is able to accuratley spike up before both the 2008 financial crisis and the pandemic. This shows our code is working and due to it's gradual growth and decreases shows proof of not being overfit.
