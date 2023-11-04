# Quantum-Machine Learning Binary classification

## Abstract of Research
This research aims to introduce a potential application of Quantum Computing to a real-world dataset: binary classification of wine quality. The Quantum model used in this study is based on Bayesian theorem, and its circuit incorporates widely used quantum gates such as the X-Gate, CNOT Gate, RY Gate, and CRY Gate, as well as those unique to this research, the CCRY Gate and CCCRY Gate. 

Similar to classical machine learning models, the addition of attributes enhanced the model's accuracy on a test dataset. The quantum machine learning model developed for this research achieved slightly better performance than a classical machine learning model, the random forest algorithm.


## 1.	Introduction

### 1.1	Motivation 

Quantum computing has been gaining attentions from diverse fields. There has been significant amount of investment for Quantum Computing and significant technological advancement correspondingly. Quantum Computing, though it is still in early stage of development, is proven to overcome complexity problems that classical computing has faced for long time, such as Grover’s searching algorithm and Shor’s factorizing. Those algorithms are substantiated evidence that Quantum Computing will supersede in some areas, such as cryptography and optimization. As such, the author would like to explore areas where quantum computing can enhance computational performance over classical computation and establish applied quantum computations to innovate the real-world. 

### 1.2	Project Overview

Quantum computing has been attracting attention from diverse fields. Significant investment and technological advancements in Quantum Computing have been made. Although still in the early stages of development, Quantum Computing has proven to overcome complex problems that classical computing has faced for a long time, such as with Grover's search algorithm and Shor's factoring. These algorithms are substantial evidence that Quantum Computing will excel in certain areas, such as cryptography and optimization. As such, the author aims to explore areas where quantum computing can enhance computational performance over classical computation and to establish applied quantum computations that innovate in the real world.


### 1.3	Dataset

This research will utilize the “Wine Quality Data Set” listed on Kaggle [1], which comprises 1599 instances with 11 features and 1 label:

<div align="center">
<table width="100%">

 <tr>
    <th>1</th>
    <th>2</th>
    <th>3</th>
    <th>4</th>
    <th>5</th>
    <th>6</th>
 </tr>
 <tr>
    <td align = "center" width="(100/x)%">fixed acidity</td>
    <td align = "center" width="(100/x)%">volatile acidity</td>
    <td align = "center" width="(100/x)%">citric acid</td>
    <td align = "center" width="(100/x)%">residual sugar</td>
    <td align = "center" width="(100/x)%">chlorides</td>
    <td align = "center" width="(100/x)%">free sulfur dioxide</td>
  </tr>
 <tr>
    <th>7</th>
    <th>8</th>
    <th>9</th>
    <th>10</th>
    <th>11</th>
    <th>12</th>
 </tr>
 <tr>
    <td align = "center" width="(100/x)%">total sulfur dioxide</td>
    <td align = "center" width="(100/x)%">density</td>
    <td align = "center" width="(100/x)%">ph</td>
    <td align = "center" width="(100/x)%">sulphates</td>
    <td align = "center" width="(100/x)%">alcohol</td>
    <td align = "center" width="(100/x)%">quality</td>
  </tr>
</table>
</div>


## 2.	Experimentation Plan

We aim to build models to classify a dataset of wines into “good” and “bad” quality categories. This will involve using a classical machine learning model, specifically a random forest [2] with a maximum depth of 2, and four quantum models (QMLs). The classical model will serve as a benchmark to identify the four most significant factors for the binary classification task, using the sklearn.ensemble.RandomForestClassifier().feature_importances_ method.

The QMLs will be developed with varying numbers of features: two, three, and four. These features will be re-coded as either 1 (if ≥ mean) or 0 (otherwise). From these encodings, the probability of each value will be calculated and mapped onto corresponding qubits. For instance, Qubit0 will represent the likelihood of a wine having high or low alcohol content, while Qubit1 will represent high or low sulphate content. Further details are elaborated on later in the paper.

The quantum models will adopt a hybrid approach, combining classical and quantum computing. This approach includes three main components: pre-processing, training, and post-processing. For data ingestion/pre-processing and post-processing, both classical and quantum models will utilize classical computing techniques, such as sklearn.model_selection.train_test_split at a 20% ratio [3] and sklearn.metrics.confusion_matrix [4].


## 3.	Quantum circuit and Gates  [5][6]

A quantum circuit is a sequence that assigns likelihoods to respective qubits and manipulates superpositions via gates, effectively transforming quantum states. Numerous types of gates are commonly available; our models will utilize four standard gates and two derived variants.

<div align="center">
<table>
  <tr>
    <td>X-Gate</td>
    <td style="text-align: center; font-size: 0.8em; width: 100px; padding: 10px;">
      The X Gate is employed to invert the state of a targeted qubit. Initially, Qubit0 is set to the state |0>. When the X Gate is applied to Qubit0, it transitions to the state |1>, as illustrated in the subsequent diagram.
    </td>
    <td><img src="https://user-images.githubusercontent.com/62607343/138567451-c03aab9f-3a3f-450e-9236-c2669dee8682.png"></td>
  </tr>
  <tr>
   <td>CNOT-Gate</td>
    <td style="text-align: center; font-size: 0.8em; width: 100px; word-wrap: break-word; padding: 10px;">
    The CNOT Gate, denoted as CX in the Qiskit API [7], stands for Controlled NOT gate. It is used to flip the state of a target qubit (Qubit1 in this case) if the state of a control qubit (Qubit0 in this case) is |1>. As illustrated, Qubit1, which was initially in the state |0>, is flipped to the state |1> by the CNOT gate.
    </td>
   <td><img src="https://user-images.githubusercontent.com/62607343/138567459-c934cb90-e85a-491f-ae32-99fd8d6cda49.png"></td>
  </tr>
  <tr>
   <td>RY-Gate</td>
    <td style="text-align: center; font-size: 0.8em; width: 100px; word-wrap: break-word; padding: 10px;">
    The RY-Gate rotates the state of a qubit by an angle derived from a probability value using a helper function, prob_to_angle [8]. In the example provided, qubit0 undergoes a rotation corresponding to 50% probability by the RY Gate. As a result, the superposition of qubit0 shifts from |0> to an equal superposition of 0.5|0> + 0.5|1>.
    </td>
   <td><img src="https://user-images.githubusercontent.com/62607343/138567460-a8ee9381-d755-4fbe-b65a-ae2a7be49ab9.png"></td>    
  </tr>
  <tr>
   <td>CRY-Gate</td> 
    <td style="text-align: center; font-size: 0.8em; width: 100px; word-wrap: break-word; padding: 10px;">
    The CRY-Gate, indicated by the "C" for Controlled, rotates the state of a qubit by a specific angle when its controlling qubit is in the |1> state. Initially, Qubit1 is set to |1> after being flipped by the X gate. Subsequently, Qubit1 undergoes a 50% rotation by the RY Gate. Consequently, the superposition of Qubit1 becomes 0.5|0> + 0.5|1>.
    </td>
   <td><img src="https://user-images.githubusercontent.com/62607343/138567471-49054adf-fa1b-4dfe-8168-e44eccc22c47.png"></td>
  </tr>
</table>
</div>

The CRY-Gate, where "C" stands for Controlled, rotates the state of a qubit by a certain angle if its controlling qubit is in the |1> state. Initially, Qubit1 is set to |1> after being flipped by the X gate. Subsequently, Qubit1 is rotated by 50% using the RY Gate. As a result, the superposition state of Qubit1 becomes 0.5|0> + 0.5|1>. Derivative gates of the CRY gate, such as the CCRY gate and the CCCRY gate, are explored in depth in the subsequent section.

## 4.	Model
The flow chart provided below presents a structured blueprint of the quantum machine learning model designed for binary classification of wine quality.


<div align="center">
  <img src="https://user-images.githubusercontent.com/62607343/138567476-2862400c-e6cd-410f-b46d-9350742ad640.png" style="margin: auto;">
</div>


- The first step is data ingestion, conducted in the classical process involving data validation, cleaning, conversion, and splitting into train and test sets. Since later steps of the model require probabilities of the occurrence of certain variables and conditional probabilities of combinations of variables, such as alcohol content, sulphates content, volatile acidity, and sulfur dioxide content, these values are converted into binary at this stage. As mentioned earlier, a value of 1 is assigned when a measurement exceeds the dataset average, and 0 otherwise. The dataset is then split into a training dataset and a test dataset at an 80:20 ratio.

- Step 2 involves calculating the probabilities of each of the four variables that contribute the most. To build a Bayesian model with two variables, for example, alcohol and sulphates content, the model needs the marginal probability of each being above or below average and the joint probabilities of their combined states. For two variables, there are four joint probabilities.

- In Step 3, a "Norm" value is calculated, which represents an estimated conditional probability—specifically, the likelihood of 'good' wine quality given the higher or lower alcohol and sulphates content. This value is refined through a recursive training process in the "train_qbn_wine" function, primarily influenced by the "to_params" and "calculate_qual_params" functions, which are discussed in Step 5.

- Step 4 marks the beginning of the quantum computing process. It starts by assigning the likelihoods of alcohol and sulphates content to the first two qubits, the third qubit is for norm values, and the fourth qubit represents quality. After this operation, the superpositions of qubit0 (alcohol) and qubit1 (sulphates) become 0.57|0> + 0.43|1> and 0.63|0> + 0.37|1>, respectively.

<div align="center">
 <img src="https://user-images.githubusercontent.com/62607343/138567483-78363caa-db9f-4d33-ae1a-40613b9f2328.png">
</div>

It then entangles these qubits to apply norm parameters to the third qubit.

<div align="center">
 <img src="https://user-images.githubusercontent.com/62607343/138567487-1da1f0ea-c467-4459-857f-87c87401b806.png">
</div>


Lastly, it applies the quality parameter to the fourth qubit.

<div align="center">
 <img src="https://user-images.githubusercontent.com/62607343/138567495-972f11ba-d341-4b28-872a-d353b4d2f2cb.png">
</div>


To apply the norm parameter consistent with qubit0 and qubit1, we devised a supportive function named CCRY [9]. Its operational sequence is as follows:

1)	The parameter is applied to qubit2 only if both qubit0 and qubit1 are in the state 1. Initially, qubit2 is rotated by half of the desired probability when qubit1 is in state 1.
2)	Qubit1 is then flipped if qubit0 is in state 1.
3)	Subsequently, qubit2 is rotated back by half of the probability when qubit1 is in state 1. Since qubit1's state was flipped in the preceding step, this action restores qubit2 to its original state if qubit0 is in state 1 but qubit1 is in state 0. We aim to avoid applying any probability in this scenario because the intention is to apply it exclusively when both qubit0 and qubit1 are in state 1.
4)	Qubit1 is flipped again if qubit0 is in state 1.
5)	Finally, the remaining half of the probability is applied to qubit2 if qubit0 is in state 1.

<div align="center">
 <img height="250" src="https://user-images.githubusercontent.com/62607343/138567501-be2e75f4-ae6b-4c25-b592-4d371799d38d.png">
</div>

We also created the CCCRY function to control three qubits for three variables.

<div align="center">
<table>
  <tr>
    <td><img width="500" src="https://user-images.githubusercontent.com/62607343/138567508-36566c85-b711-470b-a2a5-a1e76d523d9a.png"></td>
    <td><img width="500" src="https://user-images.githubusercontent.com/62607343/138567512-edfe04f2-1e7d-419c-80a8-3f74b518117e.png"></td>
  </tr>
</table>
</div>


The previous explanation used the terms "apply probability" and "rotate" interchangeably because these gates accept only angles, not probabilities. 
Therefore, an additional function has been established to convert probabilities into angles [10].

 <img src="https://user-images.githubusercontent.com/62607343/138567516-a96b48ac-495e-432f-b277-73bb6dc65e82.png">

The complete model, incorporating two variables, is depicted in the diagram below.

<div align="center">
 <img height="300" src="https://user-images.githubusercontent.com/62607343/138567520-88092817-d38d-46be-a1b5-c31bece64d9d.png">
</div>

The model is trained during the fifth step. As mentioned in the third step, the 'train_qbn_wine' function executes the model training. This function's role is to generate recursions, producing 'results' (which are the quantum states at each recursive training iteration).

The 'results' are then input into the 'to_params' function. The 'to_params' function calculates 2^(n+1) parameters (Norm parameters). These parameters are computed for each category (such as higher/lower alcohol, higher/lower sulphates, and high/low quality) by dividing the sum of the previously determined parameters of the data favored by Norm (equal to 1) by the total sum of all data in that category. This new set of parameters is then applied to the model in the subsequent iteration.

Finally, the 'calculate_qual_params' function returns a set of two parameters. These are obtained by dividing the sum of the Norm parameters for good quality wine by the sum of the Norm parameters for all data. Conceptually, these parameters can be interpreted as an indication of the previous accuracy of the estimated Norm parameters.


## 5.	Result
We experimented with a total of five models: one classical machine learning model, the random forest model, and four quantum models. The following sections show the results and specifications for each model.

### 5.1	Random Forest model
The random forest model utilized the Scikit-learn API. With a depth of two layers, it achieved an accuracy of 72.5%. While this is a basic implementation with considerable room for improvement, we will use the 72.5% as a benchmark and proceed, as the primary purpose of this research is to explore the potential of quantum computing.

This implementation identified four major contributing variables in the dataset: alcohol, sulphates, volatile acidity, and sulfur dioxide contents. Future quantum computing models will be constructed based on these variables.

<div align="center">
 <img height="300" src="https://user-images.githubusercontent.com/62607343/138567538-8f771194-db6a-47ea-82c6-3f598cacb0ac.png">
</div>


### 5.2	Quantum Model with Two Variables 
The first quantum model incorporated two variables, Alcohol and Sulphates, to derive Norm parameters for predicting Quality. This model reached a peak accuracy of 71% after 20 iterations of training.

<div align="center">
 <img height="300" src="https://user-images.githubusercontent.com/62607343/138567547-9170c10c-7ec5-47e5-b535-ce1edba095fd.png">
</div>


### 5.3	Quantum Model with Three Variables 
We developed two quantum models incorporating three variables: Alcohol, Sulphates, and Volatile Acidity. One model calculates Norm parameters using all three variables, while the other uses two—Alcohol and Sulphates—and applies a conditional probability to the Quality qubit based on the Norm parameter and Volatile Acidity.

5.3-1 Bayesian Inference with Three Variables
This model, which calculates the Norm parameter using three variables, attained an accuracy of 52% after 40 iterations.

<div align="center">
 <img height="300" src="https://user-images.githubusercontent.com/62607343/138567553-a28aab69-4a45-4195-a8cd-0bdf7869bbe8.png">
</div>

5.3-2 Bayesian Inference with Two Variables and One Independent Variable
Here, the Norm parameter is derived from two variables, and the probability of Volatile Acidity is applied directly to the Quality qubit. This model achieved an accuracy of 74% after 30 iterations.

<div align="center">
 <img height="300" src="https://user-images.githubusercontent.com/62607343/138567566-8bed355d-0a33-43bb-9759-9e92c31a2a20.png">
</div>


### 5.4	Quantum Model with Four Variables
Considering the performance of the three-variable models, it seemed more promising to explore a model deriving Norm parameters from two variables. This model calculates the Norm parameter using Alcohol and Sulphates and applies probabilities of Volatile Acidity and Sulfur Dioxide directly to the Quality qubits as conditional probabilities. It achieved a maximum accuracy of 74% after 30 iterations.

<div align="center">
 <img height="300" src="https://user-images.githubusercontent.com/62607343/138567578-85daf410-e928-4973-80f8-53602e34a921.png">
</div>


## 6.	Conclusion and Future Directions
The quantum machine learning models, with both three and four variables, achieved 74% accuracy, slightly outperforming the classical random forest model. Among the four quantum models, the second set with three variables showed more volatile performance and even deteriorated after 20 and 30 iterations. The other three quantum models showed that increasing the number of training iterations had an impact on the results up to 10 iterations, but only minimal changes were observed afterward. The model with four variables showed a 3% improvement over the two-variable model. The most notable difference was in the initial accuracy after a single iteration: the two-variable model started at 53%, while the four-variable model began at 61%.

This research utilized classical computation to identify significant contributing variables. However, in future work, I aim to explore methods for identifying key variables using quantum computing, potentially employing techniques like Grover's Search Algorithm [11], which could theoretically find a desired objective in square root time complexity (sqrt(n)). It would also be intriguing to investigate the application of Shor's algorithm [12], as there is a belief that it can enhance the time complexity of optimization problems.

## 7.	References
[1] Naresh Bhat, Wine Quality Classification - Data set for Binary Classification [https://www.kaggle.com/nareshbhat/wine-quality-binary-classification?select=wine.csv]
[2] Scikit-learn, sklearn.ensemble.RandomForestClassifier [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html]
[3] Scikit-learn, sklearn, sklearn.model_selection.train_test_split [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html]
[4] Scikit-learn, sklearn, sklearn.metrics.confusion_matrix¶ [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html]
[5] Chris Bernhardt, QUANTUM COMPUTING FOR EVERYONE (P.118)
[6] Dr. Frank Zickert - Hands-On Quantum Machine Learning With Python Volume 1_ Get Started-PYQML (2021) (P. 147)
[7] Qiskit [https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html#cnot] (3. Multi-Qubit Gates 3.1 The CNOT-Gate)
[8] Dr. Frank Zickert - Hands-On Quantum Machine Learning With Python Volume 1_ Get Started-PYQML (2021) (P. 141)
[9] Dr. Frank Zickert - Hands-On Quantum Machine Learning With Python Volume 1_ Get Started-PYQML (2021) (P.276)
[10] Dr. Frank Zickert - Hands-On Quantum Machine Learning With Python Volume 1_ Get Started-PYQML (2021) (P.260)
[11][12] Chris Bernhardt, QUANTUM COMPUTING FOR EVERYONE (P.176)
