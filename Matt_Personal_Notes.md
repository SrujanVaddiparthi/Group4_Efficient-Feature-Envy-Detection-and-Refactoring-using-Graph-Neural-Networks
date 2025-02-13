### Open Investigations
#### Understanding Feature Envy and Its Challenges  
- Feature Envy is when a method depends more on another class than its own.  
- This makes code harder to maintain because it increases dependencies.  
- Ideally, a method should belong to the class it interacts with the most.  
- But sometimes, due to poor design or project constraints, methods end up in the wrong place.  

#### Why is this a problem?  
- **Low cohesion** – The method doesn’t fit well in its current class.  
- **High coupling** – It creates unnecessary dependencies between classes.  
- **Technical debt** – Over time, this makes refactoring harder and slows down development.  

#### What Have Previous Studies Done?  
- Static analysis tools like **SonarQube, PMD, and FindBugs** try to detect Feature Envy.  
- They look at **how often a method calls another class vs. its own** and **attribute access patterns**.  
- These methods work to some extent but have major limitations:  
  - **Hardcoded rules** that don’t generalize well.  
  - **False positives**, flagging methods that are actually fine.  
  - **No refactoring suggestions**, just detection.  

- Some newer studies use **Graph Neural Networks (GNNs)** to analyze method relationships dynamically.  
- GNNs work better than static tools, but they mostly **detect** Feature Envy and don’t help with refactoring.  

#### Limitations of Existing Approaches  
1. **Static analysis is too rigid** – It flags issues based on surface-level patterns.  
2. **No real understanding of method behavior** – Just counting method calls isn’t enough.  
3. **No automated fix** – Tools tell you there’s an issue but don’t move the method for you.  
4. **Poor generalization** – A model trained on one codebase doesn’t always work well on another.  

#### Why Do We Need a Better Approach?  
- Just detecting Feature Envy isn’t enough. We need a solution that:  
  - **Finds Feature Envy more accurately** using relationship-based analysis.  
  - **Suggests where to move the method**, not just flagging an issue.  
  - **Works across different types of projects** without heavy tweaking.  

- **Graph Neural Networks (GNNs) might be the answer.**  
- Instead of just scanning code for predefined rules, they **learn from real projects** and detect patterns in method interactions.  
- A good GNN-based solution could **both detect and recommend fixes**, making refactoring easier.  

#### Summary  
- Feature Envy makes code harder to maintain.  
- Traditional tools detect it but don’t provide refactoring help.  
- GNN-based approaches could improve detection and **automate refactoring**.  
- The goal is to build a system that **finds misplaced methods and suggests where to move them**.  
