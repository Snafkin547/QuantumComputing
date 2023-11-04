# Quantum-ML

Quantum Machine Learning practice code based on<a href="https://www.amazon.com/Hands-Quantum-Machine-Learning-Python/dp/B09786HQSB#:~:text=Hands%2DOn%20Quantum%20Machine%20Learning%20With%20Python%20strives%20to%20be,a%20practical%20and%20applied%20manner"> "Hands-on Quantum Machine Learning" with Python bv Dr. Frank Zickert</a>  

<div align="center">
 <table>
   <tr>
     <td><img src="https://user-images.githubusercontent.com/62607343/132150700-7a455913-cfac-4a54-bf20-62dd90911d72.png"></td>
     <td align="center"><img width="50%" height="50%" src="https://user-images.githubusercontent.com/62607343/131725497-8214096b-3d49-41eb-a85d-e77b2d2cceff.png"/></td>
     <td><img src="https://user-images.githubusercontent.com/62607343/132150945-4511cd37-dcf3-4e4f-a076-201098600b49.png"></td>
   </tr>
 </table>
</div>

### Chapter 1 & 2: Data Ingestion and Evaluation Method
In these initial chapters, we take our first steps with the Titanic Survivor Database from <a href="https://www.kaggle.com/c/titanic/data">Kaggle</a>, focusing on the crucial task of predicting who survived the tragedy. The code provided here is designed to seamlessly guide you through the cleaning process of the dataset and setting up the training and test datasets that will be fundamental in evaluating our models.

At this stage, we're not diving into Quantum ML just yet. Instead, we lay the groundwork by testing out three simple baseline models to establish benchmarks for future comparisons:

Random Chance Predictor: It's like flipping a coin, assigning survival randomly with 0s and 1s.
The Grim Predictor: A morbid take that assumes the worst, predicting that no one survived.
The Weighted Guess: Taking into account different factors, this model attempts to predict survival with a bit more finesse using variable weights.
To visualize their effectiveness (or lack thereof), we'll look at confusion matrices and a series of graphs that paint a clear picture of each model's performance. It's a practical way to set the scene for the advanced Quantum ML methods we'll explore in later chapters.

### Chapter 3: Variational Hybrid Quantum-Classical Classifier 
In this chapter, we make a significant leap into the future of predictive modeling by constructing a Variational Hybrid Quantum-Classical Classifier specifically for the Titanic survivor database. Building upon the foundation laid in the previous chapters, this innovative approach synergizes the best of both worlds: classical computing handles the pre-processing and post-processing, while the core prediction is performed through quantum computing.

Thanks to the meticulous normalization of the dataset done earlier, we're able to employ the pre-processed values directly. The method we use to predict survival hinges on the coefficients, or weights, associated with each feature in our database. By correlating these weights with the labels (survive/die), we then calculate the probability of each possible outcome (Survival/Death).

Employing this correlation method has borne fruit, achieving a promising 70% success rate in classifying passengers.

Furthermore, this chapter isn't just about results. It's also a learning journey where we delve into quantum computing basics, including how to instantiate quantum states and the intricacies of measurement. We’ll also cover how to visually represent our findings through plotting. This is where quantum potential meets practical application, marking a bold step toward the future of machine learning.

<div align="center">
 <table>
   <tr>
    <td><img width="auto" height="300" alt="image" src="https://github.com/Snafkin547/QuantumComputing/assets/62607343/b02a51ef-7a96-4d2a-94df-c0b227f8067d"></td>
    <td><img width="auto" height="300" alt="image" src="https://github.com/Snafkin547/QuantumComputing/assets/62607343/113792f9-8e8b-4a90-9d3f-d237972f9661"></td>
   </tr>
 </table>
</div>

### Ch4 Bayesian Approach
In this analysis, we delved into the Bayesian approach using the Titanic dataset. Applying Bayes' theorem, we calculated the probability of survival given the passengers are female and possess second-class tickets. Furthermore, within the framework of Gaussian Naive Bayes, we determined the mean and standard deviation for passengers aged 29 and for those who survived. We then estimated the likelihood of survival for a 29-year-old passenger using these computed parameters.

<div align="center">
 <tr><img alt="image" src="https://user-images.githubusercontent.com/62607343/131521342-6230ff30-02f5-4cde-ab5d-232f30682946.png"></tr>
</div>

### Ch5 Gate and Qubit Rotation

In this section, we explore various quantum gates, such as the Hadamard (H) and Pauli-X (X) gates, and demonstrate their effects on qubits. The impact of these gates can be conveniently visualized on a quantum circuit using the qc.draw('mpl') function. Additionally, we introduce a custom method to rotate a qubit by a specified angle, showcasing the flexibility in manipulating qubit states.

Visualizations of Quantum Gates and State Rotations:


<div align="center">
 <table>
   <tr>
    <td><img src="https://user-images.githubusercontent.com/62607343/131569601-3a735b01-17ed-4490-ba10-ce34b9bd7df0.png" alt="Quantum Circuit" style="margin-right: 10px;"/></td>
    <td><img src="https://user-images.githubusercontent.com/62607343/131569807-3d9568d2-0e3b-4ea6-9e05-8ab37aaf50b4.png" alt="Quantum Circuit with Gates" style="margin-left: 10px;"/></td>
   </tr>
 </table>
</div>


<div align="center">
 <table>
   <tr>
    <td><img width="300" height="auto" src="https://github.com/Snafkin547/QuantumComputing/assets/62607343/03001431-67bd-4e67-9e2b-06abb9e9ec01" alt="Qubit State Rotation" style="margin-left: 10px;"/></td>
    <td><img width="300" height="auto" src="https://user-images.githubusercontent.com/62607343/131569891-9ef145b1-b766-43f5-b673-f4b56045ea78.png" alt="Qubit State Rotation" style="margin-left: 10px;"/></td>
   </tr>
 </table>
</div>


### Ch6 Multi-Qubit
Created circuits to compute three types of probability:

Marginal Probability
Joint Probability
Conditional Probability
We defined a prob_to_angle function to convert probability into an angle for qubit rotation. The Marginal Probability can be computed simply by applying this function.

For Joint Probability, we adjusted the probability of Qubit0 being in state 1 by event_a (set to 0.4) and the probability of Qubit1 being in state 1 by event_b (set to 0.8). In this setup, the qubit state 0011 represents a joint probability of 0.32.

Conditional Probability computation was more complex for mathematical reasons. Essentially, we:
(1) Initialized 12 Quantum Registers and 1 Classical Register.
(2) Set a prior and a modifier.
(3) Applied the prior to Qubit0.
(4) Applied an X gate to Qubit0.
(5) Adjusted Qubit1 with the prior/(1-prior) when Qubit0 was in state 1, to split and achieve the intended prior probability - here, Qubit1 was originally Qubit0 and vice versa.
(6) Applied an X gate to Qubit0 to reverse operation 3.
(7) Applied the modifier - 1, which is the actual increment when it is greater than 1, and recorded the quantum state to Qubit11.
(8) Applied an X gate to Qubit0 and a CNOT gate to Qubit11 when Qubit0 was in state 1.
(9) Measured the qubit and reflected the result in the Classical Register.

The intricate part of the code involves Steps 4 and 6, because, as illustrated in the accompanying image, there is a for loop. This loop registers the quantum state multiplied by step * 0.1, which is designed to reduce the prior when it is greater than 0.5 and will be offset/divided later at Step 6. This step refers to the quantum register calculated by pos=ceil((modifier-1)*10). The actual value of the modifier is inconsequential as it merely looks for the corresponding register.

<div align="center">
 <table>
   <tr>
    <td><img width="600" height="200" src="https://user-images.githubusercontent.com/62607343/131724666-09b7e3fd-186e-4315-bc45-91ca7ae220a1.png"  style="margin-right: 5%;"/></td>
    <td><img width="400" height="400" src="https://user-images.githubusercontent.com/62607343/131725497-8214096b-3d49-41eb-a85d-e77b2d2cceff.png" style="margin-left: 5%;"/></td>
   </tr>
 </table>
</div>


### Ch7 Quantum Naive Bayes

We applied Quantum naive bayse to the Titanic dataset, using Ticket Class and gender as two contributing factors to the probability of death/survival.
The entirety is a hybrid of Classical-Quantum-Classical model.

The code walks through the development phase of QC step by step, but ultimately we created a quantum circuit with 7 quantum registers and 1 classical register to hold results of measurement. Big picture follows:

 1) Initialize 7 Quantum Registers and 1 classical Register
 2) Make register 4, 5, and 6 second-auxiliary, auxiliary, and target qubits respectively
 3) Make the Target Qubit(=6), hold posterior probability of survival and death, 1 and 0 respectively


<div align="center">
 <table>
   <tr>
     <!-- Empty cell for alignment with the first image -->
     <td></td>
     <!-- Text above the second image -->
     <td style="text-align: center; font-size: 0.8em;">When prior=0.38, modifiers=[1.6403508771929822, 1.9332048273550118]</td>
   </tr>
   <tr>
     <td><img width="671" alt="image" src="https://github.com/Snafkin547/QuantumComputing/assets/62607343/46e0cdb1-fd52-47ba-a80e-71e795a59c6f"></td>
     <td><img width="600" alt="image" src="https://user-images.githubusercontent.com/62607343/132077755-ae68a9fc-c9e6-44ab-a3ef-5dff8a8c32b9.png"/></td>
   </tr>
 </table>
</div>



### Ch8 Quantum Computing Theory

Theoretical and mathematical discussions on the non-cloning theorem, Deutsch's algorithm, and the Oracle demonstrate that the quantum circuit fundamentally serves as a placeholder for a transformation gate. Additionally, they explore methods to manipulate qubits to represent specific probabilities


### Ch9 Quantum Bayesian Network

Using the Titanic dataset, we applied probabilities for being male/female and child/adult, from which we calculated the probability of survival or death. Essentially, we initialized three quantum registers (QRs) to represent the probabilities of being a child or adult, male or female, and for survival or death, respectively. Survival is indicated if the third qubit (indexed as 2) is 1, and death if it equals 0. At the end of the circuit, a measurement is taken, which is then stored in a classical register, reflecting the outcome. By running the simulation 100 times with the 'qasm_simulator' as the backend, we observed an approximate conditional survival probability of 0.38.


<div align="center">
 <table>
   <tr>
    <td align="center"><img alt="image" src="https://user-images.githubusercontent.com/62607343/132106907-fd6493c6-686f-4923-9ffc-7bd5e0473e75.png"></td>
    <td align="center"><img width="300" alt="image" src="https://user-images.githubusercontent.com/62607343/132106943-7215e402-cf88-4569-bb81-478498d95016.png"></td>
  </tr>
  <tr>
     <td colspan="2"><img alt="image" src="https://user-images.githubusercontent.com/62607343/132106951-f401bc27-f846-43c9-b783-3997f0eb02e5.png"></td>
  </tr>
 </table>
</div>

### Ch10 Bayesian Inference

We defined 'Norm' as a contributing factor to survival or death in the Titanic incident. We trained the parameter by initializing it with arbitrary values and then integrated 'Norm' into our Titanic dataset. Subsequently, we assigned the 'Norm' conditional probability to the quantum circuit (QC), specifically to the qubit at position 2. For our calculations, we summed all states with a 1 at qubit position 6 to represent the probability of survival, and states with a 0 to represent the probability of death. The system predicts survival when the probability of survival (p_surv) is greater than the probability of death (p_died), and predicts death otherwise.

Overall, the informational level of my quantum Bayesian network classifier achieved an accuracy of about 0.76.

![image](https://user-images.githubusercontent.com/62607343/132138747-922af691-c843-4d35-9600-5bfbff56ad74.png)

### Ch11 Quantum Phase

This section introduced a new concept called 'Phase,' which involves rotation around the Z-axis without affecting probability. Additionally, we learned about the Z gate and the Controlled Z gate, which essentially transform the state |+⟩ into |−⟩. The phenomenon of Phase Kickback was also covered.

Moreover, we explored the Bloch sphere, including how to use the 'plot_bloch_vector_spherical' function to visualize the sphere with an arrow indicating the quantum state.

- plot_bloch_vector_spherical: Visualize sphere with an arrow pointing quantum state

<div align="center">
<img width="25%" height="25%" src="https://user-images.githubusercontent.com/62607343/132150742-69e3a5f8-4654-4eb5-a13a-532ca8c2d316.png">
</div>

- plot_bloch_multivector:Also visualizes spheres with arrows each pointing quantum state

<div align="center">
<img width="600" alt="image" src="https://github.com/Snafkin547/QuantumComputing/assets/62607343/fe3d37b5-d174-4b29-8a78-c6e4bc36cdea">
</div>

 - plot_state_qsphere: Visualizes quantum Phase with color variations

<div align="center">
<img width="35%" height="35%" src="https://user-images.githubusercontent.com/62607343/132150700-7a455913-cfac-4a54-bf20-62dd90911d72.png">
</div>

Finaly, we built a circuit with Grover's iterate:

<div align="center">
 <table>
   <tr>
    <td colspan="2" align="center"><img src="https://github.com/Snafkin547/QuantumComputing/assets/62607343/d70fa606-e23b-44a2-87ea-1037de50948f"></td>
  </tr>
   <tr>
    <td align="center"><img width="480" alt="image" src="https://github.com/Snafkin547/QuantumComputing/assets/62607343/1f329e52-6cb2-4c71-9a30-32aaf309b93f"></td>
    <td align="center"><img src="https://user-images.githubusercontent.com/62607343/132150945-4511cd37-dcf3-4e4f-a076-201098600b49.png"></td>
  </tr>
 </table>
</div>


### Ch12 Application of Phase 

We constructed quantum circuits using an Oracle to identify each of the states |10⟩, |01⟩, |00⟩, and |11⟩.

To isolate the state |10⟩, the circuit proceeds as follows:

(1) Hadamard gates are applied to both qubits to create superpositions.
(2) A Controlled-Z gate is implemented between X gates on Qubit0. This encapsulation inverts the phase of the target state.
(3) A subsequent Hadamard gate on both qubits is used to flip the amplitudes of |10⟩ and |01⟩.
(4) Another Controlled-Z gate is applied in the same configuration to further manipulate the phase.
(5) A final Hadamard gate on both qubits completes the process, effectively isolating the state |10⟩ from the others.

![image](https://user-images.githubusercontent.com/62607343/132270184-dc49a38f-9243-485c-855b-737ac072836d.png)


