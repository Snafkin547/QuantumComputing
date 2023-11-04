# Quantum-ML

Quantum Machine Learning practice code Based on<a href="https://www.amazon.com/Hands-Quantum-Machine-Learning-Python/dp/B09786HQSB#:~:text=Hands%2DOn%20Quantum%20Machine%20Learning%20With%20Python%20strives%20to%20be,a%20practical%20and%20applied%20manner"> "Hands-on Quantum Machine Learning with Python (Dr. Frank Zickert)</a>  

### Ch1/2 Data Ingestion and Evaluation Method
Modeling a binary classification model for Titanic Survivor Database from <a href="https://www.kaggle.com/c/titanic/data">Kaggle</a>, who survived and who did not.
In this section, the code cleans the database, prepars training/test dataset, and formulate evaluation method for the predictive model.
Actual prediction with Quantum ML is not implemented in this section, but it examines three baseline models; one predicts using random integers 0 or 1; the second one predicts deaths consistently; and the third one predicts using weights as variable. Their results are visualized in a confusion matrix and graphs.

### Ch3 Variational Hybrid Quantum-Classical Classifier 
Built a variational Hybrid Quantum Classsical Classifer for the Titanic survivor database as we in the previous section.
This classifier consists of pre-processing and post-processing done in classical computing and prediction done in quantum computing.
Since the relevant database values were normalized in the previous section, we used the values as they were. 
To compute probability for the classification, we used coefficient/weight of each feature with the labels (survive/die), and multiply the the feature value and weight/correlation into probability of final classification(Survival/Death). Based on the correlation method, the model classified passengers at 70% of successful rate.

Over the course, we also learned the way to instantiate quantum state, measure and plot
![image](https://user-images.githubusercontent.com/62607343/131435759-3b0343ad-2215-4f0b-9d08-cb711871c81f.png)


### Ch4 Bayesian Approach
Reviewed Bayes Approach and Gaussian Naive Bayes using Titanic dataset. For Bayes Theorem, we calculated the probability of survival given femal passengers and second class tickets. For Gaussian Naive Bayes, we computed mean and std for 29 years old and those of survivors, and calculated the probability of survival given the age of 29. 

![image](https://user-images.githubusercontent.com/62607343/131521342-6230ff30-02f5-4cde-ab5d-232f30682946.png)


### Ch5 Gate and Qubit Rotation

Walked through various types of gate, such as Hadamard and X gates, which can be visualized in quantum circle, using qc.draw('mpl').

![image](https://user-images.githubusercontent.com/62607343/131569601-3a735b01-17ed-4490-ba10-ce34b9bd7df0.png)

Also, Qubit is rotatable by angle, using a self-made method 

![image](https://user-images.githubusercontent.com/62607343/131569807-3d9568d2-0e3b-4ea6-9e05-8ab37aaf50b4.png)

![image](https://user-images.githubusercontent.com/62607343/131569855-7f3c69c7-8b71-42b0-bdc1-0b74e11c7061.png)
![image](https://user-images.githubusercontent.com/62607343/131569891-9ef145b1-b766-43f5-b673-f4b56045ea78.png)

### Ch6 Multi-Qubit
Created circuits to compute three types of probability:
 Marginal Probability 
 Joint Probability
 Conditional Probability

We defined prob_to_angle function to convert probability into angle to rotate qubit.
Marginal Probability can be computed just by applying the function. 

Joint Probability by changing probability of Qubit0 to be 1 by event_a(=0.4) and probability of Qubit1 to be 1 by event_a(=0.8). This way, qubit 0011 represents joint probability 0.32.

Conditional probability was trickier due to mathematical reason. Basically, we 
 1) Initialize 12 Quantum Registers and 1 classical Register
 2) Set prior and modifier
 3) Apply prior to the qubit0
 4) X gate to qubit0
 5) Apply prior/(1-prior) to qubit1 when qubit0 is 1, in order to split and get intended prior probability - Qubit1 originally was qubit0 and vice versa
 6) X gate to qubit0 to reverse the operation3
 7) Apply modifier - 1, which is the actual increment when it is greater than 1, and register quantum sate to qubit11.
 8) Apply X gate to qubit0 and CNOT gate to 11 when qubit0=1
 9) Meaure qubit and reflect the result to classical register 


Tricky part of the code is Step 4 and 6, because, as the following image shows, there is a for loop. This loop actually registers quantum state multiplied by step * 0.1, which is aimed to smaller prior when it is greater than 0.5 and will be cancelled/divided later at the step 6, because step 6 refers to the quantum register calculated by pos=ceil((modifier-1)*10). The value of modifier does not matter because it just searches for according register.

![image](https://user-images.githubusercontent.com/62607343/131724666-09b7e3fd-186e-4315-bc45-91ca7ae220a1.png)

The actual circuit looks like this:
![image](https://user-images.githubusercontent.com/62607343/131725497-8214096b-3d49-41eb-a85d-e77b2d2cceff.png)

### Ch7 Quantum Naive Bayes

We applied Quantum naive bayse to the Titanic dataset, using Ticket Class and gender as two contributing factors to the probability of death/survival.
The entirety is a hybrid of Classical-Quantum-Classical model.

The code walks through the development phase of QC step by step, but ultimately we created a quantum circuit with 7 quantum registers and 1 classical register to hold results of measurement. Big picture follows:

 1) Initialize 7 Quantum Registers and 1 classical Register
 2) Make register 4, 5, and 6 second-auxiliary, auxiliary, and target qubits respectively
 3) Make the Target Qubit(=6), hold posterior probability of survival and death, 1 and 0 respectively 
   ![image](https://user-images.githubusercontent.com/62607343/132077747-4987d75a-cac3-4034-b852-3bd5de71c24a.png)

 When prior=0.38, modifiers=[1.6403508771929822, 1.9332048273550118]
   ![image](https://user-images.githubusercontent.com/62607343/132077755-ae68a9fc-c9e6-44ab-a3ef-5dff8a8c32b9.png)

### Ch8 Quantum Computing Theory

Theoretical/Mathematical topics around non-cloning theorem, Deutsch’s algorithm and Oracle to show that the quantum circuit is essentially a placeholder for a transformation gate, and the way to make qubits hold certain probabilities.

### Ch9 Quantum Bayesian Network

Using Titanic dataset, applied probability of being male/female and child/adult, from which we measured probability of survival/death.
Basically, we initiated 3 QRs and made them hold probabilities of child/adult, make/female, and survival or death. So, survival if the third qubit(=2) is 1, and death if it is =0. With measurement at the end of the circuit, it makes a classical register hold outcome of the measurement. By shooting 100 times with "qasm_simulator" as backend, it returns around 0.38 conditional survival probability.

![image](https://user-images.githubusercontent.com/62607343/132106907-fd6493c6-686f-4923-9ffc-7bd5e0473e75.png)
![image](https://user-images.githubusercontent.com/62607343/132106943-7215e402-cf88-4569-bb81-478498d95016.png)
![image](https://user-images.githubusercontent.com/62607343/132106951-f401bc27-f846-43c9-b783-3997f0eb02e5.png)

### Ch10 Bayesian Inference

Defined "Norm", contributing factor of survival/death in Titanic incident, train parameter(initiatializing with arbitrary values), added Norm into our Titanic dataframe, and assigned Norm conditional probability to QC, Qbit pos 2. All states with 1 in qbit pos 6 are summed to be prob of survival, and with 0 as otherwise. When p_surv > p_died, then it will predicts survival, and death when otherwise.

The overall information level of my quantum Bayesian network classifier achieved about 0.76.

![image](https://user-images.githubusercontent.com/62607343/132138747-922af691-c843-4d35-9600-5bfbff56ad74.png)

### Ch11 Quantum Phase

This section introduced a new notion: Phase, which rotates around Z-axis but does not affect probability.
Also, we leaned Z gate and Controlled Z gate, which essentially converts |+> to |->. as well as Phase Kickback.

Furthermore,we learned Boch's sphere, including the use of 
- plot_bloch_vector_spherical: Visualize sphere with an arrow pointing quantum state

![image](https://user-images.githubusercontent.com/62607343/132150742-69e3a5f8-4654-4eb5-a13a-532ca8c2d316.png)

- plot_bloch_multivector:Also visualizes spheres with arrows each pointing quantum state

![image](https://user-images.githubusercontent.com/62607343/132150716-4b91741a-4066-4eab-b639-9c4fe342f0ba.png)

 - plot_state_qsphere: Visualizes quantum Phase with color variations

![image](https://user-images.githubusercontent.com/62607343/132150700-7a455913-cfac-4a54-bf20-62dd90911d72.png)

Finaly, we built a circuit with Grover's iterate:
![image](https://user-images.githubusercontent.com/62607343/132150986-4c05c903-8b29-44fc-9607-444dcf7419c8.png)

![image](https://user-images.githubusercontent.com/62607343/132150945-4511cd37-dcf3-4e4f-a076-201098600b49.png)

### Ch12 Application of Phase 

Built circuits with Oracle to find each of |10>, |01>, |00>, and |11>.

To find |10>, the circuit applies Hadamard gates to both qubits, encapsulate Controlled-Z gate between X gates applied to Qubit0.
It then apply Hadamard gate, which flips amplifiers of |10> and |01>, and again apply Controlled-z gate in the same way.
We lastly apply Hadamard gate to isolate all other states but |10>.

![image](https://user-images.githubusercontent.com/62607343/132270184-dc49a38f-9243-485c-855b-737ac072836d.png)


