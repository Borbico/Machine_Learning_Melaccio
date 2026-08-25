# ❓ Compendio Integrale Quiz d'Esame — Machine Learning
**Corso del Prof. Alessio Micheli — Università di Pisa**
*(Raccolta completa di tutte le 128 domande dei quiz con risposte corrette e spiegazioni dettagliate)*

---
### 📌 Domanda 1 (ID: 1)
**Testo**: The gradient can be exploited to provide the direction toward local maxima of a function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The gradient shows the direction where the function grows, while the negative of the gradient show the direction where the function decreases.

---

### 📌 Domanda 2 (ID: 2)
**Testo**: The negative of gradient can be exploited to provide the direction toward local minima of a function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The gradient shows the direction where the function grows, while the negative of the gradient show the direction where the function decreases.

---

### 📌 Domanda 3 (ID: 3)
**Testo**: $$\|x\|_2 = \sum_i x_i^2$$

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Euclidean norm of a vector is given by $$\sqrt{\sum_i x_i^2} = \|x\|_2$$

---

### 📌 Domanda 4 (ID: 4)
**Testo**: $\|x\|_2^2 = \sum_i x_i^2$

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Euclidean norm of a vector is given by $$\sqrt{\sum_i x_i^2} = \|x\|_2$$

---

### 📌 Domanda 5 (ID: 5)
**Testo**: There exits the search inductive bias

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> There exist two types of inductive bias, the language bias and the search bias.

---

### 📌 Domanda 6 (ID: 6)
**Testo**: An unbiased learner corresponds to a look-up table model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The version space VS , with respect to hypothesis space H and training set TR, is the subset of H,TR hypotheses from H consistent with all training examples. The only examples that are unambiguously classified by an unbiased learner represented with the VS are the training examples themselves (we are implementing a lookup table). An unbiased learner is unable to generalize (on new instances). Each unobserved instance will be classified 1 (or positive) by precisely half the hypothesis in VS and 0 (or negative) by the other half.

---

### 📌 Domanda 7 (ID: 7)
**Testo**: For the supervised learning the loss is computed to evaluate the distance between the model response and its inputs

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The Loss measures the “distance” between and (the output and the target).

---

### 📌 Domanda 8 (ID: 8)
**Testo**: For the supervised learning the loss is computed to evaluate the distance between the target and the model inputs

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The Loss measures the “distance” between h(x) and d (the output and the target).

---

### 📌 Domanda 9 (ID: 9)
**Testo**: Removing the inductive bias from a learning system brings advantages for the learning system

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> A learner that makes no prior assumptions regarding the identity of the target function/concept has no rational basis for classifying any unseen instances. The bias (restriction, preference) is not only assumed for efficiency, but it is also needed for the generalization capability. However, it does not tell us (quantify) which one is the best solution for generalization yet.

---

### 📌 Domanda 10 (ID: 10)
**Testo**: Polynomial coefficients (magnitude) typically decreases after training with higher values of M

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> With a 0th order polynomial we have complexity M=0, and increasing the polynomial coefficient we are increasing the complexity.

---

### 📌 Domanda 11 (ID: 11)
**Testo**: Polynomial coefficients (magnitude) typically increases after training with higher values of M

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> With a 0th order polynomial we have complexity M=0, and increasing the polynomial coefficient we are increasing the complexity

---

### 📌 Domanda 12 (ID: 12)
**Testo**: The model selection is made looking to best result on the test set

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The Validation Set is used to select the best model (hyper-parameters tuning, model selection)

---

### 📌 Domanda 13 (ID: 13)
**Testo**: The model selection is made looking to best result on the validation set

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The Validation Set is used to select the best model (hyper-parameters tuning, model selection)

---

### 📌 Domanda 14 (ID: 14)
**Testo**: The model selection is made looking to best result on the training set

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The Validation Set is used to select the best model (hyper-parameters tuning, model selection)

---

### 📌 Domanda 15 (ID: 15)
**Testo**: The number of free parameters after a LBE is typically greater than in the original model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Typically, the number of parameters K can exceed the number of original features n (K>n, before it was n)

---

### 📌 Domanda 16 (ID: 16)
**Testo**: A linear classifier with LBE cannot solve a non-linear separable problem

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The model is linear in the parameters w_k (also in ϕ, not in x), even though the transformed features are non-linear. This approach can model more complicated relationships (than the linear).

---

### 📌 Domanda 17 (ID: 17)
**Testo**: A polynomial expansion is a form of LBE

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> A polynomial expansion is a form of linear basis expansion, where polynomials are used as bases to represent the function.

---

### 📌 Domanda 18 (ID: 18)
**Testo**: The perceptron is a LTU

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> A perceptron is a specific form of Linear Threshold Unit (LTU) where the activation function is a step function. It returns 1 if the weighted sum of inputs exceeds a certain threshold, otherwise, it returns 0. In simpler terms, a perceptron can be viewed as a simplification of an LTU with a binary activation function.

---

### 📌 Domanda 19 (ID: 19)
**Testo**: The perceptron learning algorithm works with a unit that compute $$sign(w^Tx)$$

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> For each training pa]ern (x,d) , where d can be +1 or –1, we compute the output activation $$out = sign(w^Tx)$$

---

### 📌 Domanda 20 (ID: 20)
**Testo**: The perceptron learning algorithm cannot solve the XOR problem

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> For the XOR (exclusive OR), it’s not possible using a linear separation because no linear separation surface exists.

---

### 📌 Domanda 21 (ID: 21)
**Testo**: A small networks with a hidden layer of perceptrons can solve the XOR problem

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> For the XOR we can’t use a linear separation, but we can use a NN. We can implement the network by introducing two additional variables and connecting them appropriately, such that the points are linearly separable in the new space.

---

### 📌 Domanda 22 (ID: 22)
**Testo**: The perceptron learning algorithm always converges in a finite number of steps

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The perceptron is guaranteed to converge (classifying correctly all the input pa]erns) in a finite number of steps if the problem is linearly separable.

---

### 📌 Domanda 23 (ID: 23)
**Testo**: The perceptron learning algorithm solves only tasks with all positive pa[erns

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The perceptron learning algorithm solves tasks with positive and negative pa]erns.

---

### 📌 Domanda 24 (ID: 24)
**Testo**: The sigmoidal logistic is not a differentiable function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The sigmoidal logistic is a differentiable function because there exists its derivative.

---

### 📌 Domanda 25 (ID: 25)
**Testo**: A linear model can be obtained by an artificial neural unit with an identity activation function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The identity activation function is a linear function. So, if you consider an artificial neural unit (neuron) with an identity activation function, it means that the output of the unit will be simply a linear combination of its inputs.

---

### 📌 Domanda 26 (ID: 26)
**Testo**: The perceptron has a differentiable activation function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The perceptron uses the threshold activation function that is non differentiable at every point because it lacks a defined derivative at points where there is a discrete jump. In such points, the derivative is practically zero.

---

### 📌 Domanda 27 (ID: 27)
**Testo**: A NN can be seen a LBE only when we consider LBE for classification

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> A NN can be seen as a LBE for both regression and classification

---

### 📌 Domanda 28 (ID: 28)
**Testo**: A NN can be seen as a LBE, and we showed it when the phi function of the LBE is the NN’s output neurons function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The phi function of the LBE is the NN’s hidden neurons function. So in $$h(x) = f\!\left(\sum_j w_j \phi_j(x, w)\right)$$, we have that $$\phi_j(x, w) = f_j\!\left(\sum_i w_{ji} x_i\right)$$ is the hidden units output.

---

### 📌 Domanda 29 (ID: 29)
**Testo**: A NN cannot address a regression task, but only classification tasks

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> A NN can address both a regression task and a classification tasks

---

### 📌 Domanda 30 (ID: 30)
**Testo**: The backpropagation is based on the minimization of the MSE

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The backpropagation is based on the minimization of different minimization functions, not only the MSE.

---

### 📌 Domanda 31 (ID: 31)
**Testo**: The backpropagation allows us to obtain the update rule for the units finding the delta for each of them

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

---

### 📌 Domanda 32 (ID: 32)
**Testo**: High values of w initialization accelerate the training

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> high initial values don’t accelerate the training but can lead to saturation issues. Saturation occurs when neurons in the network reach extreme values, causing the gradients during backpropagation to become very small

---

### 📌 Domanda 33 (ID: 33)
**Testo**: All zero values to initialize the weights can be a good strategy

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> If all weights are initialized to zero, all neurons in the network will produce the same output during forward propagation.

---

### 📌 Domanda 34 (ID: 34)
**Testo**: It is always convenient to have a high eta value to speed-up the training

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The eta value can be high or low and depends on different situations. Sometimes a high eta value can lead to overfijng.

---

### 📌 Domanda 35 (ID: 35)
**Testo**: The momentum has been introduces to provide a regularization of the model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The momentum in gradient optimization, particularly in the gradient descent algorithm with momentum, is not primarily introduced for the purpose of model regularization. Momentum is a technique used to accelerate convergence during the training of machine learning models. It introduces a "momentum" term that adds a fraction of the previous step to the current weight update. This helps maintain a momentum of acceleration in the direction of gradient descent, aiding in overcoming flat local minima or reducing oscillation during optimization.

---

### 📌 Domanda 36 (ID: 36)
**Testo**: Too few hidden units can lead to underfijng

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In general, too few hidden units can cause underfitting, and too many hidden units can cause overfijng.

---

### 📌 Domanda 37 (ID: 37)
**Testo**: With a well regularized approach, early stopping can even be not necessary or it can be used without showing its effect

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> If we have a good regularization, then we don’t need (or strictly need) the Early Stopping heuristic and we can stop at training convergence. Anyway, we can also use both. In that case, the Early Stopping will not enter in action if VL error does not increase.

---

### 📌 Domanda 38 (ID: 38)
**Testo**: In the Nesterov Momentum approach the gradient is computed aTer the movement due to the momentum

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> So, we first apply the momentum with $$w' = w + \alpha \,\Delta w_{old}$$, and then we evaluate the new old gradient at this interim point, i.e. with such w (in the previous methods we evaluate the gradient and then we applied the momentum)

---

### 📌 Domanda 39 (ID: 39)
**Testo**: In the cascade correlation algorithm the hidden units are trained to minimize their LMS

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In CCA, the model starts with an empty network and gradually adds hidden layers during training. Each new hidden layer is trained to approximate the residual error of the current network, rather than directly minimizing the error on the output variable, as in the case of Mean Squared Error (MSE/LMS).

---

### 📌 Domanda 40 (ID: 40)
**Testo**: The use of a K-fold CV can solve in itself the problem of a rigorous approach to the model selection and assessment

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Cross-Validation is a fundamental practice for evaluating model performance, but the rigorous selection and assessment of a model require a broader consideration of various aspects of the machine learning process.

---

### 📌 Domanda 41 (ID: 41)
**Testo**: It is not suggested to use the VL set to select among different models

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The validation set (VL) is used specifically to select the best model, among different models and/or hyper-parameters configurations. It’s for the model selection.

---

### 📌 Domanda 42 (ID: 42)
**Testo**: It is a good practice to stop selecting the number of epochs that provides very low training error

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Relying solely on the value of the training error may lead to a problem known as “overfitting”. It is common to use techniques such as the validation set and early stopping during training.

---

### 📌 Domanda 43 (ID: 43)
**Testo**: Both early stopping and regularization Tikhonov aims to control the effective complexity of the model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Early Stopping involves halting the training of the model once its performance on a validation dataset starts deteriorating, indicating a potential point where the model is beginning to overfit to the training data. Tikhonov Regularization involves adding a regularization term to the loss function during model training. Both of these techniques aim to prevent excessive model complexity, providing a trade-off between the model's ability to fit the training data and its ability to generalize to new data.

---

### 📌 Domanda 44 (ID: 44)
**Testo**: The momentum in the batch and on-line version has the same meaning and effect

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> For Batch we have to consider the previous batch, while for Online we have to consider the previous example.

---

### 📌 Domanda 45 (ID: 45)
**Testo**: The validation and training sets compose the “design set”

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The training set and validation set together are sometimes jointly called development/design set, because they are used to build the final model (to train different models and select the best final model).

---

### 📌 Domanda 46 (ID: 46)
**Testo**: The K-fold CV is an approach to split the dataset

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> With the K-fold Cross-Validation, we split the data such that we use all the data for validation purpose, instead of using a specific set for the training and another one for the validation.

---

### 📌 Domanda 47 (ID: 47)
**Testo**: Selecting the number of units of a NN by the grid search is enough, we can avoid other regularization approaches,True False

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Choosing the number of units (or neurons) in a neural network via grid search addresses only part of the complexity of the problem of training a neural network.

---

### 📌 Domanda 48 (ID: 48)
**Testo**: During the training the complexity of the NN increases

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> When we start the training, we are increasing the complexity, but it is important to find a balance between the complexity of the model and its ability to generalize to new data.

---

### 📌 Domanda 49 (ID: 49)
**Testo**: The model selection does not consider the TS error

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> For model selection we use only the validation set, and the test set results cannot be used for model selection. There is the gold rule to keep separation between goals and use separate sets.

---

### 📌 Domanda 50 (ID: 50)
**Testo**: The training error is the best indicator to choose among different models

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The best indicator to choose among different models is the validation error.

---

### 📌 Domanda 51 (ID: 51)
**Testo**: The test error is the best indicator to choose among different models

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The best indicator to choose among different models is the validation error.

---

### 📌 Domanda 52 (ID: 52)
**Testo**: The lambda and the patient are hyperparameters

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> They are hyperparameters because they are external parameters to the learning algorithm that must be set manually before starting training.

---

### 📌 Domanda 53 (ID: 53)
**Testo**: R-Prop is a variant of the backpropagation based algorithm in which the exact value of the gradient does not care

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> R-prop, that does not use the value of the gradient (it can vanish for deep layers) but the sign of gradient (to increase or decrease the weights)

---

### 📌 Domanda 54 (ID: 54)
**Testo**: It is suggested to use the VL set to estimate the general risk R of the SLT for assessment purposes

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> For assessment purposes, we use the TS set to estimate the general risk R

---

### 📌 Domanda 55 (ID: 55)
**Testo**: The validation set is used to estimate the $$R_{emp}$$ as used in the SLT

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> We use the TS set to estimate the Remp

---

### 📌 Domanda 56 (ID: 56)
**Testo**: Too many units can lead to overfitting unless proper regulations is adopted

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In general, too few hidden units can cause underfitting, and too many hidden units can cause overfitting. If a proper regulations is adopted, this can change the aspect.

---

### 📌 Domanda 57 (ID: 57)
**Testo**: H shatters X if and only if for every possible dichotomy of X, there exists a h in H that classifies perfectly all the data

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> H shatters X if and only if H can represent all the possible dichotomies on X (with 0 errors). There exists at least one hypothesis thanks to which we can discriminate perfectly all these points.

---

### 📌 Domanda 58 (ID: 58)
**Testo**: The VC dimension of a class of functions H is the maximum cardinality of a set (configuration) of points in X that can discriminated by a model in H for at least a labeling assignment

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> We have VC(H) = p if H shatters at least one set (configuration) of p points and if H cannot shatter any set (configuration) of p + 1 points.

---

### 📌 Domanda 59 (ID: 59)
**Testo**: The objective function of the primal form of the SVM hard margin minimizes the training errors

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The minimization of the equation $$\psi(w) = \frac{1}{2} w^{T} w$$ (objective function for hard margin) correspond to the minimization of ||w||.

---

### 📌 Domanda 60 (ID: 60)
**Testo**: In the SVM hard-margin the minimization of the norm of w leads to the increment of the $$R_{emp}$$ of the SLT

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Minimizing the norm of w is equivalent to minimizing the VC dimension and thus to minimizing the capacity term

---

### 📌 Domanda 61 (ID: 61)
**Testo**: The theorem of Vapnik shows for a SVM hard margin the relationships between the VC-dim and the margin

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Maximizing the margin we minimize the VC dimension

---

### 📌 Domanda 62 (ID: 62)
**Testo**: For a ML model with a n-dimensional input space, the VC-dim is n + 1

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In general, the VC dimension of a class of linear (separator/decision) hyperplanes (LTU) in a n-dimensional space is n+1. Note that could exist an upper-bound, because in some case the VC-dim can be less constraining for the model (because we want to reduce the VC dimension).

---

### 📌 Domanda 63 (ID: 63)
**Testo**: A support vector is a positive value

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> A support vector x(s) is a vector that satisfies the equation $$d_i \left( w^{T} x_i + b \right) \ge 1$$ exactly, such that we have $$d^{(s)} \left( w^{T} x^{(s)} + b \right) = 1$$ (1 is just a convention for the canonical representation of the hyperplane). Graphically, the support vectors are the closest data points to the hyperplane.

---

### 📌 Domanda 64 (ID: 64)
**Testo**: The SVM hard margin (under its assumption of use) does not need any hyperparameters

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> There aren’t hyperparameters because there are no external parameters that must be set manually before starting.

---

### 📌 Domanda 65 (ID: 65)
**Testo**: Some configuration of 3 points in the plane are not sha[ered by a linear model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> We can have non linearly separable problem, e.g when we have the points positioned one after the other.

---

### 📌 Domanda 66 (ID: 66)
**Testo**: An optimal hyperplane is the one with minimum margin

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The optimal hyperplane is the hyperplane which maximizes the margin.

---

### 📌 Domanda 67 (ID: 67)
**Testo**: The objective function of the SVM hard margin is related to the control of the VC-dim of the model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Maximizing the margin we minimize the VC dimension

---

### 📌 Domanda 68 (ID: 68)
**Testo**: Bagging for classification is likely to decrease the margin of the classifier

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> For classification, bagging it’s likely to increase (and not decrease) the margin of a classifier. That’s because the average of K separation curves is likely to be “in the middle” (in medio stat virtus).

---

### 📌 Domanda 69 (ID: 69)
**Testo**: Bagging can exploit high variance learners

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> We can use high variance models (with low bias) because they can perform well on average.

---

### 📌 Domanda 70 (ID: 70)
**Testo**: Boosting cannot be used with weak learners

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Boosting is a technique of ensembling, used to enhance the performance of weak learners incrementally.

---

### 📌 Domanda 71 (ID: 71)
**Testo**: Boosting differentiates each training progressively concentrating on well classified pa[erns

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Boosting differentiates each training progressively concentrating on misclassified (and not well classified) pa]erns

---

### 📌 Domanda 72 (ID: 72)
**Testo**: A deep neural network allows for multiple levels of abstraction of the internal representations

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The deep-learning methods exploit distributed representation with multiple levels of representation. We have an increase of level of abstraction through different layers (each layer learns increasingly abstract and complex representations of the input data).

---

### 📌 Domanda 73 (ID: 73)
**Testo**: A deep (i.e. not shallow) neural network is the only way to implement a representation learning

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Deep neural networks are effective, but they are not the only way to implement representation learning. There are other approaches and models that can be used for representation learning, such as shallower neural networks, recurrent neural networks, convolutional neural networks, and other unsupervised learning methods.

---

### 📌 Domanda 74 (ID: 74)
**Testo**: With a two layer logic network is possible to solve the parity problem with polynomial number of gates with respect to the input dimension

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> With a two layer logic network, we can solve the parity problem with an exponential (not polynomial) number of gates with respect to the input dimension. With N inputs, we have $$\frac{2^N}{2} + 1 = 2^{N-1} + 1$$ gates.

---

### 📌 Domanda 75 (ID: 75)
**Testo**: An autoencoder can be used to initialize a deep NN exploiting the decoder (pre)trained part

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> When the autoencoder is trained, then we have $$W_1$$ and $$W′$$ (encoder and decoder) and we can delete the decoder and use only the encoder part.

---

### 📌 Domanda 76 (ID: 76)
**Testo**: Stacking many layers with sigmoidal functions and using backprogation, gradient vanish can occur

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> “Gradient vanishing” occurs when the gradients of the activation functions become very small as one approaches the initial layers of the network during backpropagation. This results in weight updates of the initial layers becoming meaningless, and the network does not learn effectively from those layers. The sigmoid function has a limited range between 0 and 1, and its derivative reaches a maximum value of approximately 0.25. When calculating gradients during backpropagation, these gradients can decrease exponentially as they move backwards through the network. As a result, gradients become very small near the initial layers, contributing to the gradient vanishing problem.

---

### 📌 Domanda 77 (ID: 77)
**Testo**: Gradient clipping check the norm of the gradient to bound the weights update

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The repetition of multiplication through many layers can introduce cliffs in the cost function. With clipping, considering g as the gradient and v as the norm threshold, if ||g|| > v then we can impose a normalization of the gradient with g = vg/||g|| . So, we are moving in the gradient direction, but bounding the weight update.

---

### 📌 Domanda 78 (ID: 78)
**Testo**: Increasing the number of layers in a logic network is possible to solve the parity problem with polynomial number of gates with respect to the input dimension

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> For a solution with logN layers, we have a polynomial number of gates. We construct a complete tree with N − 1 (XOR) internal nodes (a complete binary tree with N leaves has N-1 internal nodes). Considering that the XOR has 3 AND/OR gates, so we have 3 gates for each node and 3(N − 1) total gates. So, with more layers we are reducing the number of gates.

---

### 📌 Domanda 79 (ID: 79)
**Testo**: Choose the appropriate claims (with, as usual, negative score for the not correct options):

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 2, Opzione 5, Opzione 7**

**Opzioni:**
  1. [✅] “No flattening” results show that for some functions, a shallow model would require an exponential number of units with respect to the dimension of the input
  2. [✅] “No flattening” results show that a deep model can be more efficient than a shallow one, with an exponential gain in terms of units
  3. [❌] “No flattening” results show that a deep model can be more efficient than a shallow one, with an exponential gain in terms of bits used to encode each weight of the network
  4. [❌] “No flattening” results show that a deep model is more expressive than a shallow one, with an exponential gain in terms of functions that can be represented
  5. [✅] “No flattening” results does not show that a deep model is more expressive than a shallow one, but that we can have exponential gain in terms of number of weights
  6. [❌] “No flattening” results show that for some functions, a shallow model would require an exponential number of units with respect to the number of layers
  7. [✅] “No flattening” results show that some compositional functions cannot be implemented by a shallow network holding the same number of units of a deep network
  8. [❌] “No flattening” results show that a deep models cannot be more efficient, in terms of units, than a shallow one

> [!NOTE]
> **Spiegazione / Derivazione**:
> The general idea of no-flattening is that when a function can be compactly represented by a deep architecture, it might need a very large architecture to be represented by an insufficiently deep one (we can always represent the same task, but the difference is in the efficiency). Deep architectures can compactly represent certain functions that might require a very large architecture in a shallow setting.

---

### 📌 Domanda 80 (ID: 80)
**Testo**: Choose from the following one or more alternatives of formulations for the bias/variance decomposition that are correct (with the notation and assumption used in the lecture of the course). All appropriate answers should be listed below (to gain the max score), but those indicated as appropriate that are not, however, affect the exercise score by negative values

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 2, Opzione 3**

**Opzioni:**
  1. [❌] $E_P[(y - h(x))^2] = (f(x) - \bar{h}(x))^2 + E_P[(h(x) - \bar{h}(x))^2] + \sigma^2$ with the mean computed with respect to all the possible data points x according to a distribution P.
  2. [✅] $E_P[(y - h(x))^2] = E_P[(h(x) - \bar{h}(x))^2] + (f(x) - \bar{h}(x))^2 + \sigma^2$ with the mean computed with respect to all the possible training sets sampled according to a distribution P.
  3. [✅] $E_P[(y - h(x))^2] = (f(x) - E_P[h(x)])^2 + E_P[(h(x) - \bar{h}(x))^2] + \sigma^2$ with the mean computed with respect to all the possible training sets sampled according to a distribution P.
  4. [❌] $E_P[(y - h(x))^2] = E_P[(h(x) - \bar{h}(x))^2] + (f(x) - h(x))^2 + \sigma^2$ with the mean computed with respect to all the possible training sets sampled according to a distribution P.
  5. [❌] $E_P[(y - h(x))^2] = (f(x) - \bar{h}(x))^2 - E_P[(h(x) - \bar{h}(x))^2] + \sigma^2$ with the mean computed with respect to all the possible training sets sampled according to a distribution P.
  6. [❌] $E_P[(y - h(x))^2] = (f(x) - h(x))^2 + E_P[(h(x) - \bar{h}(x))^2] + \sigma^2$ with the mean computed with respect to all the possible data points x according to a distribution P.

> [!NOTE]
> **Spiegazione / Derivazione**:
> We have that:

$$
E_P[(y-h(x))^2] = E_P[h(x)^2] + E_P[y^2] - 2E_P[y]E_P[h(x)]
$$

where:

$$
E_P[h(x)^2] = E_P[(h(x)-\bar{h}(x))^2] + \bar{h}(x)^2
$$

$$
E_P[y^2] = E_P[(y-f(x))^2] + f(x)^2
$$

$$
-2E_P[y]E_P[h(x)] = -2 f(x)\bar{h}(x)
$$

Therefore:

$$
E_P[(y-h(x))^2] = E_P[(h(x)-\bar{h}(x))^2] + \bar{h}(x)^2 - 2f(x)\bar{h}(x) + f(x)^2 + E_P[(y-f(x))^2]
$$

$$
= E_P[(h(x)-\bar{h}(x))^2] + (\bar{h}(x)-f(x))^2 + E_P[(y-f(x))^2]
$$

Moreover, the mean is computed with respect to all the possible training sets sampled according to a distribution P (and not with respect to all the possible data points x).

---

### 📌 Domanda 81 (ID: 81)
**Testo**: Choose one or more alternatives of formulations of the RBF kernel that are correct. All appropriate answers should be listed below (to gain the max score), but those indicated as appropriate that are not, however, affect the exercise by a negative score.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 5, Opzione 7, Opzione 8**

**Opzioni:**
  1. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2}\|v-t\|^2}$
  2. [❌] $K(t,v) = e^{-\frac{1}{2\sigma^2}(t-v)}$
  3. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2}\|t-v\|}$
  4. [❌] $K(t,v) = - e^{\frac{1}{2\sigma^2}\|t-v\|^2}$
  5. [✅] $K(t,v) = e^{-\frac{1}{2\sigma^2}\|v-t\|^2}$
  6. [❌] $K(t,v) = e^{-\frac{1}{2\sigma^2}\|t-v\|}$
  7. [✅] $K(t,v) = e^{-\frac{1}{2\sigma^2}\|t-v\|^2}$
  8. [✅] $K(v,t) = e^{-\frac{1}{2\sigma^2}\|v-t\|^2}$
  9. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2}\|t-v\|^2}$

> [!NOTE]
> **Spiegazione / Derivazione**:
> The Radial Basis Function kernel is defined as $K(x,x_i) = e^{-\frac{1}{2\sigma^2}\|x - x_i\|^2}$. The kernel is symmetric, so $K(t,v) = K(v,t)$. Therefore, any formulation with a negative exponent and the squared Euclidean norm is correct.

---

### 📌 Domanda 82 (ID: 82)
**Testo**: List the proper claims for the Soft margin SVM

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 2, Opzione 5, Opzione 6, Opzione 7**

**Opzioni:**
  1. [❌] Small values of C can lead to overfitting
  2. [✅] Slack variables change the constraints of the primal problem
  3. [❌] The slack variables have no effect on the width of the margin
  4. [❌] High values of C can lead to underfitting
  5. [✅] With slack variables is possible to gain a greater margin
  6. [✅] The values of alfa allow us to know the not-null slack variables
  7. [✅] C in a soft margin SVM can be tailored by the user with a cross validation approach
  8. [❌] Slack variables does not changes the constraints of the primal problem

> [!NOTE]
> **Spiegazione / Derivazione**:
> 1. For Soft Margin, small values of C can lead to underfitting.
2. For Soft Margin, we introduce the non-negative scalar variables $\xi_i$, such that we add a tolerance in the constraint, and we obtain $d_i(w^T x_i + b) \ge 1 - \xi_i\,(\forall i = 1,\ldots,N)$.
3. The slack variables have effect on the width of the margin because we are now admitting points inside the margin that allows us to have a larger margin and a noise tolerance.
4. For Soft Margin, high values of C can lead to overfitting.
5. With the slack variables we are now admitting points inside the margin that allows us to have a larger margin and a noise tolerance.
6. If $0 < \alpha_i < C$ then $\xi_i = 0$ (so we are on the edge of the margin), and if $\alpha_i = C$ then $\xi_i \ge 0$ (so we are inside the margin).
7. C is a regularization hyper-parameter to control the trade-off between the empirical risk minimization and the capacity term minimization.
8. With the slack variables $\xi_i$ we add a tolerance in the constraint, and we obtain $d_i(w^T x_i + b) \ge 1 - \xi_i\,(\forall i = 1,\ldots,N)$.

---

### 📌 Domanda 83 (ID: 83)
**Testo**: Compute the correct delta for a hidden unit u or z that is connected in output to units denoted by the index v, using the conventions assumed in the course.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 4, Opzione 6**

**Opzioni:**
  1. [❌] $\delta_u = (\sum_u \delta_v w_{vu}) f'_u(net_u)$
  2. [❌] $\delta_z = (\sum_z \delta_v w_{vz}) f'_z(net_z)$
  3. [❌] $\delta_u = (\sum_u \delta_u w_{vu}) f'_u(net_u)$
  4. [✅] $\delta_u = (\sum_v \delta_v w_{vu}) f'_u(net_u)$
  5. [❌] $\delta_z = (\sum_v \delta_v w_{zv}) f'_z(net_z)$
  6. [✅] $\delta_z = (\sum_v \delta_v w_{vz}) f'_z(net_z)$
  7. [❌] $\delta_z = (\sum_z \delta_z w_{vz}) f'_z(net_z)$
  8. [❌] $\delta_u = (\sum_v \delta_v w_{uv}) f'_u(net_u)$

> [!NOTE]
> **Spiegazione / Derivazione**:
> The equation for the hidden unit's error signal is $\delta_j = \left( \sum_{k=1}^{K} \delta_k w_{kj} \right) f'_j(net_j)$, where j denotes a hidden unit and k denotes an output unit.

---

### 📌 Domanda 84 (ID: 84)
**Testo**: Compute the correct delta for a hidden unit r or s that is connected in output to units denoted by the index v, using the conventions assumed in the course.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 3, Opzione 5**

**Opzioni:**
  1. [❌] $\delta_s = (\sum_s \delta_s w_{vs}) f'_s(net_s)$
  2. [❌] $\delta_s = (\sum_v \delta_s w_{uv}) f'_s(net_s)$
  3. [✅] $\delta_s = (\sum_v \delta_v w_{vs}) f'_s(net_s)$
  4. [❌] $\delta_r = (\sum_r \delta_v w_{vr}) f'_r(net_r)$
  5. [✅] $\delta_r = (\sum_v \delta_v w_{vr}) f'_r(net_r)$
  6. [❌] $\delta_s = (\sum_s \delta_v w_{vs}) f'_s(net_s)$
  7. [❌] $\delta_r = (\sum_r \delta_r w_{vr}) f'_r(net_r)$
  8. [❌] $\delta_r = (\sum_v \delta_v w_{rv}) f'_r(net_r)$

> [!NOTE]
> **Spiegazione / Derivazione**:
> The equation for the hidden unit's error signal is $\delta_j = \left( \sum_{k=1}^{K} \delta_k w_{kj} \right) f'_j(net_j)$, where j denotes a hidden unit and k denotes an output unit.

---

### 📌 Domanda 85 (ID: 85)
**Testo**: Given an unit with activation function f and δ_p = (d_p - f(net(x_p))), and denoting the weights with t or i, list the proper results among the following equations for a pattern p:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 3, Opzione 6**

**Opzioni:**
  1. [❌] $\frac{\partial E_p(w)}{\partial w_t} = +2\delta_p (x_{p,t}) f'(x_p) net(x_p)$
  2. [❌] $\frac{\partial E_p(w)}{\partial w_i} = -2\delta_p (x_{p,i}) f'(x_p)$
  3. [✅] $\frac{\partial E_p(w)}{\partial w_t} = -2 x_{p,t} \delta_p f'(net(x_p))$
  4. [❌] $\frac{\partial E_p(w)}{\partial w_i} = +2 \delta_p f'(net(x_p)) x_{p,i}$
  5. [❌] $\frac{\partial E_p(w)}{\partial w_t} = -2 x_{p,t} f'(net(x_p)) net(x_p)$
  6. [✅] $\frac{\partial E_p(w)}{\partial w_i} = -2 \delta_p f'(net(x_p)) x_{p,i}$

> [!NOTE]
> **Spiegazione / Derivazione**:
> Considering $E(w) = \sum_p (d_p - o(x_p))^2 = \sum_p (d_p - f(x_p^T w))^2$ and $o(x_p) = f(x_p^T w) = f(net(x_p))$, we obtain for one pattern p:

$\frac{\partial E_p(w)}{\partial w_j} = -2 x_{p,j} (d_p - f(net(x_p))) f'(net(x_p)) = -2 x_{p,j} \delta_p f'(net(x_p))$.

Therefore the correct expressions are the ones consistent with $-2 x_{p,j} \delta_p f'(net(x_p))$.

---

### 📌 Domanda 86 (ID: 86)
**Testo**: According to the picture and notation in the slide “Exercise to use a uniform notation for the inputs” of Lecture on NN-part1, what is the right expression for the net of a hidden unit? (note that the input argument of the net has been omitted)

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 6**

**Opzioni:**
  1. [❌] $net_j = \sum_j w_{ji} o_i$
  2. [❌] $net_i = \sum_i w_{ij} o_i$
  3. [❌] $net_k = \sum_j w_{kj} o_j$
  4. [❌] $net_j = \sum_j w_{ij} o_j$
  5. [❌] $net_j = \sum_i w_{ij} o_i$
  6. [✅] $net_j = \sum_i w_{ji} o_i$

> [!NOTE]
> **Spiegazione / Derivazione**:
> We used i for input units, j for hidden units, and k for output units. Therefore, the net of the hidden unit is $net_j(x) = \sum_i w_{ji} x_i$, i.e., the weighted sum of the outputs of the input units.

---

### 📌 Domanda 87 (ID: 87)
**Testo**: List the proper claims for the K-NN:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 3, Opzione 5, Opzione 7**

**Opzioni:**
  1. [✅] Neighborhoods are no longer "local" when the input dimension n increases
  2. [❌] The curse of dimensionality is due to the high retrieval cost for testing data in K-NN
  3. [✅] The curse of dimensionality is due to the changing in the data density when we change the input dimension
  4. [❌] A high value of K leads to very flexible models
  5. [✅] Lower values of K, for the K-NN, lead to more flexible models
  6. [❌] For the K-NN: if the dimension of input increases, we can compensate the curse of dimensionality reducing the value of K so to avoid negative effects on the generalization capability
  7. [✅] For the K-NN, fixing K: with high dimensional inputs, the range of features values to be considered becomes high, so you lose the similarity among examples in the N(x)

> [!NOTE]
> **Spiegazione / Derivazione**:
> In high input dimensions, near neighborhoods tend to be spatially large, and estimates are no longer local. The curse of dimensionality is related to the change in data density as dimensionality increases. Higher values of K lead to more rigid models, while lower values of K lead to more flexible models. In high-dimensional spaces, distances between examples become less meaningful, causing a loss of similarity in the neighborhood N(x).

---

### 📌 Domanda 88 (ID: 88)
**Testo**: The use of a Tikhonov loss introduces a new learning rule where we have a further addend for the gradient of MSE given by:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 6**

**Opzioni:**
  1. [❌] $+ 2 \lambda w$
  2. [❌] $- 2 \lambda w$
  3. [❌] $- 2 \lambda$
  4. [❌] $+ 2 \lambda$
  5. [❌] $+ 2 \lambda w$
  6. [✅] $- 2 \lambda w$

> [!NOTE]
> **Spiegazione / Derivazione**:
> With Tikhonov (L2) regularization, the loss becomes $E(w) + \lambda \|w\|^2$. The gradient of the regularization term is $2\lambda w$, therefore in gradient descent the update rule includes the term $-2\lambda w$. Hence the additional addend in the update rule is $-2\lambda w$.

---

### 📌 Domanda 89 (ID: 89)
**Testo**: A LTU for a pattern p can be expressed as (assuming x0 = 1):

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 4, Opzione 5, Opzione 8**

**Opzioni:**
  1. [✅] $h(x_p) = sign(x_p^T w)$
  2. [❌] $h(x_p) = sign(x_p^T w_p)$
  3. [❌] $h(x_p) = sign(\sum_{t=0}^n x_{p,t} w_u)$
  4. [✅] $h(x_p) = sign(\sum_{t=0}^n w_t x_{p,t})$
  5. [✅] $h(x_p) = sign(\sum_{u=0}^n x_{p,u} w_u)$
  6. [❌] $h(x_p) = sign(w_p^T x_p)$
  7. [❌] $h(x_p) = sign(\sum_{u=0}^n x_{p,u} w_t)$
  8. [✅] $h(x_p) = sign(w^T x_p)$

> [!NOTE]
> **Spiegazione / Derivazione**:
> For a Linear Threshold Unit (LTU), the output is given by the sign of the weighted sum. Assuming x0 = 1 (bias included in w), we have $h(x_p) = sign(w^T x_p) = sign(x_p^T w) = sign(\sum_{i=0}^n x_{p,i} w_i)$. All algebraically equivalent expressions are correct.

---

### 📌 Domanda 90 (ID: 90)
**Testo**: Considering the SLT, fixing l and δ (delta), a polynomial with increased/higher degree:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 4, Opzione 5, Opzione 6**

**Opzioni:**
  1. [❌] It implies higher R value
  2. [❌] It implies lower R value
  3. [❌] It implies that the VC-bound does not hold
  4. [✅] It has higher VC-confidence
  5. [✅] It has higher VC-dim
  6. [✅] It provides higher fitting capability, reducing $R_emp$
  7. [❌] It provides higher fitting capability, increasing $R_emp$

> [!NOTE]
> **Spiegazione / Derivazione**:
> Increasing the degree of a polynomial increases the model capacity and therefore its VC-dimension. With fixed l and δ, the VC-confidence term increases with VC-dim. Moreover, a higher-capacity model provides higher fitting capability, typically reducing the empirical risk R_emp (training error).

---

### 📌 Domanda 91 (ID: 91)
**Testo**: Considering the SLT, fixing l and δ (delta), a polynomial with decreased/lower degree:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 2, Opzione 3**

**Opzioni:**
  1. [❌] It implies higher R value
  2. [✅] It provides lower fitting capability, increasing R_emp
  3. [✅] It has lower VC-dim
  4. [❌] It implies lower R value
  5. [❌] It implies that the VC-bound does not hold

> [!NOTE]
> **Spiegazione / Derivazione**:
> Decreasing the polynomial degree reduces model capacity and therefore reduces the VC-dimension. A lower-capacity model has lower fitting capability and can lead to higher empirical risk R_emp (underfitting). However, it does not necessarily imply a direct increase or decrease of the true risk R, nor that the VC-bound does not hold.

---

### 📌 Domanda 92 (ID: 92)
**Testo**: List all the appropriate answers (but those indicated as appropriate that are not affect the exercise sum by a negative score).

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 4, Opzione 5, Opzione 8**

**Opzioni:**
  1. [✅] In a regression task each data is labeled with a real number
  2. [❌] In a classification task we search for subsets of "similar" data
  3. [❌] The hypothesis space contains all the input data
  4. [✅] In supervised learning for classification the labeled examples can be provided in the form <x,+1> e <x,-1>
  5. [✅] In the supervised learning the aim is to approximate an unknown target function
  6. [❌] In a unsupervised learning the labeled examples are provided in the form <x,d>
  7. [❌] In a multi-class problem f can have K continuous values in output
  8. [✅] If we have a Boolean target function we learn a classifier with two classes

> [!NOTE]
> **Spiegazione / Derivazione**:
> 1. Regression is a process of estimating a real-valued function from a finite set of noisy samples: known pairs $(x, f(x) + \text{random noise})$.
2. In classification, patterns (feature vectors) are members of a class and the goal is to assign each observed pattern to a specific class label based on a learned function.
3. The hypothesis space is a space of different functions.
4. If $f(x)$ is boolean (binary classification), the output class can be T/F, 0/1, -1/+1, negative/positive.
5. Given training examples $\langle x,d \rangle$ for an unknown function $f$, we want to find a good approximation to $f$, a hypothesis $h$ to predict on unseen data.
6. Unsupervised learning has no teacher: the training set is unlabeled data $\langle x \rangle$.
7. In a multi-class problem $(C_1, C_2, \dots, C_K)$ the function assigns the input to one of the multiple classes.
8. If $f(x)$ is boolean then we have a binary classification.

---

### 📌 Domanda 93 (ID: 93)
**Testo**: The negative of this vector provides the direction toward local minima of a function f(x1, x2) in a 2-dimensional space: (∂f/∂x1, ∂f/∂x2)

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The gradient ∇f gives the direction of maximum increase of the function. Therefore, the negative of the gradient gives the direction of steepest descent, i.e., toward a local minimum.

---

### 📌 Domanda 94 (ID: 94)
**Testo**: List all the appropriate answers (but those indicated as appropriate that are not affect the exercise sum by a negative score).

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 4**

**Opzioni:**
  1. [✅] Changing the nature (type) of the target we can see classification or regression as a function approximation task.
  2. [❌] In a unsupervised learning the labeled examples are provided in the form <x,d>.
  3. [❌] In a classification task we search for subsets of 'similar' data.
  4. [✅] If we have a Boolean target function we learn a classifier with two classes.

> [!NOTE]
> **Spiegazione / Derivazione**:
> Changing the nature (type) of the target we can see the classification or regression as a function approximation task, If we have a Boolean target function we learn a classifier with two classes

---

### 📌 Domanda 95 (ID: 95)
**Testo**: There is no inductive bias called "full space bias".

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> There is no formally defined inductive bias known as 'full space bias' in machine learning theory.

---

### 📌 Domanda 96 (ID: 96)
**Testo**: There exists the version space inductive bias.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The version space inductive bias refers to the assumption that the target hypothesis belongs to the version space, i.e., the set of hypotheses consistent with the training data.

---

### 📌 Domanda 97 (ID: 97)
**Testo**: For the supervised learning the loss is computed to evaluate the distance between the model response and the outputs mean.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In supervised learning, the loss function evaluates the discrepancy between the model output and the true target values, not the mean of the outputs.

---

### 📌 Domanda 98 (ID: 98)
**Testo**: The VC dimension of a class of functions H is the maximum cardinality of a set (configuration) of points in X that can be discriminated by a model in H for at least a labeling assignment.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The VC dimension is the maximum cardinality of a set of points that can be shattered by H, meaning that all possible labelings of those points can be realized by hypotheses in H. It is not sufficient that at least one labeling assignment is realizable.

---

### 📌 Domanda 99 (ID: 99)
**Testo**: The VC-dim can be used to define a nested hierarchy over hypothesis spaces of growing complexity.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The VC-dimension measures the capacity of a hypothesis space. By increasing VC-dim, we obtain hypothesis spaces of increasing complexity, which can be organized in a nested hierarchy.

---

### 📌 Domanda 100 (ID: 100)
**Testo**: The objective function of the primal form of the SVM hard margin minimizes the training errors.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In hard-margin SVM the objective function minimizes 1/2 ||w||^2, i.e., it maximizes the margin under the constraint that training errors are zero. It does not minimize training errors; it assumes perfect separability.

---

### 📌 Domanda 101 (ID: 101)
**Testo**: Choose one or more alternatives of formulations of the RBF kernel that are correct. All appropriate answers should be listed below (to gain the max score), but those indicated as appropriate that are not, however, affect the exercise by a negative score.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 7, Opzione 8**

**Opzioni:**
  1. [✅] $K(t,v) = e^{-\frac{1}{2\sigma^2} \|t-v\|^2}$
  2. [❌] $K(t,v) = e^{-\frac{1}{2\sigma^2} (t-v)}$
  3. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2} \|t-v\|^2}$
  4. [❌] $K(t,v) = e^{-\frac{1}{2\sigma^2} \|t-v\|}$
  5. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2} \|t-v\|}$
  6. [❌] $K(t,v) = -e^{-\frac{1}{2\sigma^2} \|t-v\|^2}$
  7. [✅] $K(v,t) = e^{-\frac{1}{2\sigma^2} \|v-t\|^2}$
  8. [✅] $K(t,v) = e^{-\frac{1}{2\sigma^2} \|v-t\|^2}$
  9. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2} \|v-t\|^2}$

> [!NOTE]
> **Spiegazione / Derivazione**:
> The RBF (Gaussian) kernel is defined as $K(x,x') = e^{-\frac{1}{2\sigma^2}\|x-x'\|^2}$. It is symmetric, so K(t,v) = K(v,t). Therefore, all formulations with negative exponent and squared Euclidean norm are correct.

---

### 📌 Domanda 102 (ID: 102)
**Testo**: The cascade correlation algorithm allows to automatically find the number of hidden units.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The Cascade-Correlation algorithm incrementally adds hidden units during training, automatically determining the number of hidden neurons needed to improve performance.

---

### 📌 Domanda 103 (ID: 103)
**Testo**: Both early stopping and regularization Tikhonov aims to control the effective complexity of the model.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Both early stopping and Tikhonov (L2) regularization control the effective complexity of the model. Early stopping limits model capacity by halting training before overfitting occurs, while Tikhonov regularization penalizes large weights, reducing effective model complexity.

---

### 📌 Domanda 104 (ID: 104)
**Testo**: The momentum in the batch and on-line version has the same meaning and effect.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Momentum has the same conceptual role (smoothing updates and accelerating along consistent directions), but its practical effect differs between batch and on-line/SGD settings because the gradient noise and update frequency are different.

---

### 📌 Domanda 105 (ID: 105)
**Testo**: The K-fold CV is an approach to split the dataset.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> K-fold cross-validation splits the dataset into K subsets (folds), using K−1 folds for training and one fold for validation, rotating the validation fold across runs.

---

### 📌 Domanda 106 (ID: 106)
**Testo**: The model selection does not consider the TR error.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Model selection is based on validation performance (e.g., cross-validation error), not on training error (TR error), which is optimistically biased.

---

### 📌 Domanda 107 (ID: 107)
**Testo**: A NN with 1 hidden layer (using logistic activation functions) has a universal approximation capability.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> According to the Universal Approximation Theorem, a feedforward neural network with a single hidden layer and a non-linear activation function (such as the logistic sigmoid) can approximate any continuous function on a compact domain, given sufficiently many hidden units.

---

### 📌 Domanda 108 (ID: 108)
**Testo**: In a NN by using multiple output units we can obtain a multi-regression or multi-classes classifier.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Using multiple output units in a neural network allows modeling multi-output regression problems or multi-class classification tasks (e.g., using one-hot encoding and softmax for classification).

---

### 📌 Domanda 109 (ID: 109)
**Testo**: ∂E/∂w = (∂E/∂o) · (∂w/∂o)

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> By the chain rule, ∂E/∂w = (∂E/∂o) · (∂o/∂w), not (∂w/∂o). The derivative must follow the correct dependency direction.

---

### 📌 Domanda 110 (ID: 110)
**Testo**: The backpropagation allows us to obtain the update rule for each weight in the network on the basis of an exact gradient computing.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Backpropagation computes the exact gradient of the loss function with respect to each network weight using the chain rule, enabling gradient-based update rules.

---

### 📌 Domanda 111 (ID: 111)
**Testo**: In the backpropagation the delta values are propagated from the input to the output layers units.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In backpropagation, the delta values are propagated from the output layer back to the hidden and input layers (i.e., from output to input), not the other way around.

---

### 📌 Domanda 112 (ID: 112)
**Testo**: According to the Backpropagation, given a TR set with l patterns, the $$ \Delta w $$ can be expressed as (list the proper results among the following equations):

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 4, Opzione 6**

**Opzioni:**
  1. [❌] $\Delta w_{rh} = \sum_{p=1}^{l} \delta_{p,r} \; o_{p,h}$
  2. [❌] $\Delta w_{ts} = - \sum_{p=1}^{l} \delta_{p,t} \; o_{p,s}$
  3. [❌] $\Delta w_{rh} = \sum_{p=1}^{l} \delta_{p,h} \; o_{p,r}$
  4. [✅] $\Delta w_{rh} = - \sum_{p=1}^{l} \delta_{p,r} \; o_{p,h}$
  5. [❌] $\Delta w_{ts} = - \sum_{p=1}^{l} \delta_{p,s} \; o_{p,t}$
  6. [✅] $\Delta w_{ts} = \sum_{p=1}^{l} \delta_{p,t} \; o_{p,s}$

> [!NOTE]
> **Spiegazione / Derivazione**:
> In backpropagation, the weight update over the full training set is proportional to the sum over patterns of the product between the delta of the receiving unit and the output of the sending unit. Depending on the sign convention adopted for Δw, the correct formulations correspond to options 4 and 6.

---

### 📌 Domanda 113 (ID: 113)
**Testo**: The w of a NN are hyper-parameters to be selected by the user.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The weights (w) of a neural network are learned parameters optimized during training. Hyper-parameters (e.g., learning rate, number of hidden units, regularization strength) are selected by the user.

---

### 📌 Domanda 114 (ID: 114)
**Testo**: Mini-batch is a technique introduced to optimize the complexity of the model.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Mini-batch is a training strategy used to approximate the gradient during optimization. It affects computational efficiency and convergence properties, not the intrinsic complexity (capacity) of the model.

---

### 📌 Domanda 115 (ID: 115)
**Testo**: It is always convenient to have a high eta value to speed-up the training.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> A high learning rate (η) can speed up training, but it may cause instability, divergence, or oscillations. The learning rate must be carefully tuned to balance convergence speed and stability.

---

### 📌 Domanda 116 (ID: 116)
**Testo**: $w_{qr}$ denote the weight of the unit r coming from the input with index q.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> According to the notation used in the course, the weight $w_{rq}$ (not $w_{qr}$) denotes the weight of unit r receiving input from unit q. The first index refers to the receiving unit.

---

### 📌 Domanda 117 (ID: 117)
**Testo**: A LTU for a pattern p can be expressed as (assuming x0 = 1):

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 2, Opzione 3**

**Opzioni:**
  1. [❌] $h(x_p) = sign(\sum_{t=0}^{n} x_{p,t} w_u)$
  2. [✅] $h(x_p) = sign(x_p^T w)$
  3. [✅] $h(x_p) = sign(\sum_{u=0}^{n} x_{p,u} w_u)$
  4. [❌] $h(x_p) = sign(w_p^T x_p)$

> [!NOTE]
> **Spiegazione / Derivazione**:
> An LTU computes the sign of the weighted sum of inputs. With $x_0 = 1$ and including the bias in w, the correct formulations are $h(x_p) = sign(x_p^T w)$ and equivalently $h(x_p) = sign(∑_{u=0}^{n} x_{p,u} w_u)$.

---

### 📌 Domanda 118 (ID: 118)
**Testo**: A linear classifier with LSE cannot solve a non-linear separable problem.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> A linear classifier in the original input space cannot solve a non-linearly separable problem. However, by using a suitable feature mapping (e.g., polynomial expansion), the problem can become linearly separable in the transformed space.

---

### 📌 Domanda 119 (ID: 119)
**Testo**: List the proper claims for the Tikhonov regularization.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 3, Opzione 4**

**Opzioni:**
  1. [❌] The regularization term does not depend on the size of w
  2. [❌] High lambda value allows maximum fitting of the training set
  3. [✅] The best lambda value allows us to optimize the model complexity
  4. [✅] Lambda = 0 corresponds to a minimum regularization effect

> [!NOTE]
> **Spiegazione / Derivazione**:
> Tikhonov regularization adds a term proportional to $||w||^2$ to control model complexity. The optimal lambda balances fitting and regularization. When $ \lambda = 0$ there is no regularization effect.

---

### 📌 Domanda 120 (ID: 120)
**Testo**: List the proper claims for the K-NN.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 2**

**Opzioni:**
  1. [✅] For the K-NN, fixing K with high dimensional inputs, the range of features values to be considered becomes high, so you lose the similarity among examples in the N(x)
  2. [✅] The curse of dimensionality is due to the changing in the data density when we change the input dimension
  3. [❌] A high value of K leads to a very flexible models
  4. [❌] If the dimension of input increases, we can compensate the curse of dimensionality reducing the value of K so to avoid negative effects on the generalization capability

> [!NOTE]
> **Spiegazione / Derivazione**:
> In high-dimensional spaces, distances lose meaning and neighborhoods are no longer truly local. Adjusting K can partially affect generalization behavior, but dimensionality still impacts similarity structure.

---

### 📌 Domanda 121 (ID: 121)
**Testo**: $||x||_2 = \sum_i x_i^2$

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The L2 norm is defined as $||x||_2 = sqrt(\sum_i x_i^2)$. The expression $\sum_i x_i^2$ corresponds to the squared L2 norm, i.e., $||x||_2^2$.

---

### 📌 Domanda 122 (ID: 122)
**Testo**: The gradient of f can be written as a vector whose components are the partial derivatives of f.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> The gradient of a scalar function $f(x_1, ..., x_n)$ is defined as the vector of its partial derivatives: $\Nabla f = (df/dx_1, ..., df/dx_n)$.

---

### 📌 Domanda 123 (ID: 123)
**Testo**: In a regression task each data is labeled with a real number.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In regression tasks, the target variable is continuous, meaning each example is labeled with a real-valued number.

---

### 📌 Domanda 124 (ID: 124)
**Testo**: Removing the inductive bias from a learning system is positive (i.e., it brings advantages for the learning system).

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Inductive bias is necessary for generalization. Without inductive bias, a learning system cannot prefer one hypothesis over another consistent with the training data, making learning and generalization impossible.

---

### 📌 Domanda 125 (ID: 125)
**Testo**: For the supervised learning the loss is computed to evaluate the distance between the model response and its inputs.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In supervised learning, the loss measures the discrepancy between the model output and the target (true label), not between the model output and the inputs.

---

### 📌 Domanda 126 (ID: 126)
**Testo**: The underfitting occurs because we have too few data.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> Underfitting occurs when the model is too simple to capture the underlying structure of the data (low model capacity). Too few data typically increases the risk of overfitting, not underfitting.

---

### 📌 Domanda 127 (ID: 127)
**Testo**: Given a unit with activation function f and $\delta_p = (d_p - f(net(x_p)))$, list the proper results among the following equations for a pattern p:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 3**

**Opzioni:**
  1. [❌] $\frac{\partial E_p(w)}{\partial w_t} = -2\, x_{p,t}\, f'\!\big(net(x_p)\big) \, net(x_p)$
  2. [❌] $\frac{\partial E_p(w)}{\partial w_i} = -2\, \delta_p\, (x_{p,i})\, f'(x_p)$
  3. [✅] $\frac{\partial E_p(w)}{\partial w_i} = -2\, \delta_p \, f'\!\big(net(x_p)\big)\, x_{p,i}$
  4. [❌] $\frac{\partial E_p(w)}{\partial w_t} = -2\, x_{p,t}\, \delta_p \, f\!\big(net(x_p)\big)$
  5. [❌] $\frac{\partial E_p(w)}{\partial w_t} = +2\, \delta_p \, (x_{p,t}) \, f'(x_p)\, net(x_p)$
  6. [❌] $\frac{\partial E_p(w)}{\partial w_i} = +2\, \delta_p \, f'\!\big(net(x_p)\big)\, x_{p,i}$

> [!NOTE]
> **Spiegazione / Derivazione**:
> Correct answers is 3

---

### 📌 Domanda 128 (ID: 128)
**Testo**: Given x input, d target, o the perceptron’s output, the update rule of the Perceptron Learning Algorithm: It is $w_new = w + η x d$ only if there is a classification error for x

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione / Derivazione**:
> In the Perceptron Learning Algorithm, the weight update $w_new = w + η x d$ is applied only when the current example is misclassified. If the example is correctly classified, no update is performed.

---
