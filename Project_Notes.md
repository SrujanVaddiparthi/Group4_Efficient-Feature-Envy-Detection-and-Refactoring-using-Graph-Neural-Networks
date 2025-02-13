# Efficient Feature Envy Detection and Refactoring Based on Graph Neural Network

## Project Guidelines By Prof

### Phase 1
What exactly we want to do with this project.  
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

### Phase 1
The amount of slides, probably 5. Titles are to be the questions below.  

### Introduction:
- Technical concepts & keywords for the current topic  
### What do we investigate? What questions do we need to answer?  
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
- Developers aim to write efficient , readable and maintainable code.
-  Real - world constraints like time and budget pressures force compromises.
- These leads to Technical Debt (TD) .which is a hidden cost !!!
- **Code smells**, one of the most serious forms of **TD (Technical Debt)**, refer to the confusing, complex, or even harmful design patterns in source code.  
- **2 well-known code smells:** - long methods & god classes
- a).  Long Method- A Method that does too many tasks and is excessively long.
- b). God Class - A Class that takes on too many responsibility.
- Studies show-both code smells often appear together!
-  (Both of these classes violate the **Single Responsibility Principle**).  
- **Feature Envy** : most common code smells; refers to a method that is more interested in an external class than its enclosing class.  
  - This **“interest”** is manifested in two aspects:
    - Its invocation of methods in the class ( when the particular method in class A is calling too many methods from another class B)  
    - Its access to attributes of the class  
- **Feature envy reduces** the cohesion of its own class and **increases the coupling** b/w classes.  
- **Best way to handle it:** move the feature envy method to the class from which it is calling too many methods from.  
- **Problem introduced:** Feature envy, one of the most common code smells which is a form of **Technical Debt (TD).**  
- **What makes it challenging?** reduces cohesion of its own class and increases the coupling between classes.  
- **What’s coupling?** A class A is coupled with class B if A affects B and vice versa.  

### Open Investigation

#### Understanding Feature Envy and Its Challenges  
Feature Envy happens when a method depends too much on another class instead of its own. This creates problems because it makes code harder to manage and modify. A method should ideally work within its own class, but when it relies on another class for most of its work, it increases complexity.  

This issue leads to three main problems:  

- Low cohesion – The method doesn’t belong in its current class, making the class less organized.  
- High coupling – The method is too connected to another class, making changes more difficult.  
- Technical debt – As these issues build up, maintaining and improving the code becomes more challenging.  

Developers try to write efficient, readable, and maintainable code, but real-world factors like tight deadlines and limited budgets often force them to take shortcuts. These shortcuts can lead to code that works in the short term but becomes harder to improve over time. Feature Envy is one of the common side effects of these decisions.  

#### What Have Previous Studies Done?  
To detect Feature Envy, many tools like SonarQube, PMD, and FindBugs use static analysis. They look at:  

- How often a method calls another class compared to its own.  
- How often a method accesses attributes from another class.  

While these methods can catch some cases, they have several problems:  

- They follow strict rules that don’t always apply to different codebases.  
- They often produce false positives, flagging methods that aren’t actually misplaced.  
- They don’t analyze the actual purpose of a method, only its structure.  
- They only detect Feature Envy but don’t provide suggestions on how to fix it.  

Newer research has explored the use of Graph Neural Networks (GNNs) to improve Feature Envy detection. GNNs analyze method relationships dynamically instead of relying on fixed rules. This makes them more flexible, but most studies using GNNs still only focus on finding Feature Envy, not solving it.  

#### Limitations of Existing Approaches  
Even with improved techniques, Feature Envy detection still has some major issues:  

1. Static analysis methods often fail to consider deeper relationships between methods and classes.  
2. Many tools don’t actually understand what a method does, only how it interacts with other parts of the code.  
3. If a tool detects Feature Envy, it still requires a developer to manually decide where to move the method.  
4. Most detection models struggle when applied to a new project with a different code structure.  

#### Why Do We Need a Better Approach?  
Simply detecting Feature Envy is not enough. We need a system that can:  

- Find Feature Envy more accurately by analyzing method relationships.  
- Suggest the best class to move the method to, instead of just flagging an issue.  
- Work well across different projects without requiring major adjustments.  

Graph Neural Networks (GNNs) offer a way to solve this problem. Instead of just counting method calls, they can learn patterns from actual projects and make smarter decisions about where methods should go. This could make Feature Envy detection more reliable and help automate the refactoring process.  

By improving detection and automating refactoring, we can reduce the time developers spend fixing misplaced methods and make code easier to maintain in the long run.  


### Solution proposed:
- They introduce **2 processes** for detecting and refactoring the feature envy problem.  
- Presents a **GNN-based approach** called **SCG (SMOTE Call Graph)**, to address the problems of detecting and refactoring feature envy.  
- **SCG formulates the detection task;**
  - **Node binary classification problem** on the method call graph.  
  - Introduces the concept of **“Calling strength”** to quantify the strength of method invocations in the call graph.  
  - ***This is how it builds upon the existing studies***: Converts the method call graph to a **method-class call graph representation**.  
  - Recommends moving the **smelly method** to the external class with the **highest calling strength**.  
- Presents **SFFL (Symmetric Feature Fusion Learning)** to address the feature envy refactoring problem.  
- Collects the following stuff from a project:
  - **Invocation** (Indicates which method calls which)  
  - **Ownership** (Indicates which method belongs to which class)  
  - **Position** (where the method is located in the code)  
  - **Semantic information** ( Meaning of the method and its purpose) 
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
