# Efficient Feature Envy Detection and Refactoring Based on Graph Neural Network

## Project Guidelines By Prof

### Phase 1
What exactly do we want to do with this project.  
“I am going to duplicate what the paper did.”  
Its like a contract between prof and us.  
Eventually we can take it up as a capstone or thesis or whatever.  
If we can identify a mislocated method, where they create a graph of all the mislocated methods by constructing a GNN.  

### Phase 2
Premature solution  

### Phase 3
Improved solution based on feedback from prof and all that  

### Phase 4
Documentation, writing what we did.  

## Phase 1
The amount of slides, probably 5. Titles are to be the questions below.  

### Introduction:
- Technical concepts & keywords for the current topic  
- What do we investigate? What questions do we need to answer?  
- What is the problem statement? Let’s define it ourselves.  
- Give details about the problem presented.  
- Talk about what makes it challenging.  
- What’s the proposed solution talking about? Convey it properly.  

### Abstract:
- "Method-Class relationships"  
- Two soln’s proposed: **SCG (SMOTE Call Graph)** & **SFFL(Symmetric Feature Fusion Learning)**  
- Captures the **“Strength of method invocations.”**  
- Converts the method-method call graph into a method-class call graph, and recommends the smelly method to the external class with the highest calling strength.  
- Focusing on **Refactoring feature envy** directly, SFFL leverages four heterogeneous graphs to represent method-class relationships.  
- **SFFL** → obtains representations for methods-classes.  
- **Three new metrics introduced:** precision2, recall2, F1score2.  
- **SCG & SFFL** was tested on **5 open-source projects** to demonstrate its superiority.  
- SCG transforms the feature envy detection problem into a **binary classification task** on a method call graph. It predicts the weights of edges, termed **calling strength**, to capture the strength of method invocations.  
- Additionally, it converts the method-method call graph into a method-class call graph and recommends the smelly method to the external class with the highest calling strength.  

### Introduction:
- **Code smells**, one of the most serious forms of **TD (Technical Debt)**, refer to the confusing, complex, or even harmful design patterns in source code.  
- **2 well-known code smells:** long methods & god classes (Both of these classes violate the **Single Responsibility Principle**).  
- **Feature Envy** : most common code smells; refers to a method that is more interested in an external class than its enclosing class.  
  - This **“interest”** is manifested in two aspects:
    - Its invocation of methods in the class ( when the particular method in class A is calling too many methods from another class B)  
    - Its access to attributes of the class  
- **Feature envy reduces** the cohesion of its own class and **increases the coupling** b/w classes.  
- **Best way to handle it:** move the feature envy method to the class from which it is calling too many methods from.  
- **Problem introduced:** Feature envy, one of the most common code smells which is a form of **Technical Debt (TD).**  
- **What makes it challenging?** reduces cohesion of its own class and increases the coupling between classes.  
- **What’s coupling?** A class A is coupled with class B if A affects B and vice versa.  

### Solution proposed:
- They introduce **2 processes** for detecting and refactoring the feature envy problem.  
- Presents a **GNN-based approach** called **SCG (SMOTE Call Graph)**, to address the problems of detecting and refactoring feature envy.  
- **SCG formulates the detection task;**
  - **Node binary classification problem** on the method call graph.  
  - Introduces the concept of **“Calling strength”** to quantify the strength of method invocations in the call graph.  
  - Converts the method call graph to a **method-class call graph representation**.  
  - Recommends moving the **smelly method** to the external class with the **highest calling strength**.  
- Presents **SFFL (Symmetric Feature Fusion Learning)** to address the feature envy refactoring problem.  
- Collects the following stuff from a project:
  - **Invocation** (no. of times method is called)  
  - **Ownership** (the class which it belongs to)  
  - **Position** (where the method is located in the code)  
  - **Semantic information** of a project  
- Encodes them into **four directed heterogeneous graphs**:
  - **Nodes:** methods & classes  
  - **Edges:** invocation or ownership relationships between them.  
- Hence **SFFL introduces a link prediction** to generate the adjacency matrix representing **ownership relationships b/w methods and classes**.  

### Why two methods though?
- They found out that **both the methods can complement each other.**  
- **SFFL exhibits better detection and refactoring performance than SCG**  
- In the case of **SCG**, when the training samples are **extremely imbalanced**, the detection performance of **SCG is superior to SFFL**.  
- **It builds upon their previous work where:**  
  - They propose a **novel approach** named **SCG based on Graph Neural Network (GNN)**, which effectively captures the pattern of **feature envy** and recommends its refactoring through **calling strength**.  
  - A **holistic approach named SFFL** is introduced based on the reconstruction of **method-class ownership heterogeneous graph**, which bypasses the detection step but reveals the essence of feature envy refactoring.  
  - **Three new evaluation metrics** are proposed, which have been proved to be effective for both **feature envy detection and refactoring.**  
  - **Extensive experiments on five open-source projects** demonstrate the **superiority of SCG and SFFL over three competitors.**  
  - **The current solution introduces a new approach named SFFL** that is based on the reconstruction of **method-class ownership heterogeneous graph**.  

## Next Steps
- **We will start working on the ppt tomorrow (2/11/2025).**  
- This is what we looked into so far.
