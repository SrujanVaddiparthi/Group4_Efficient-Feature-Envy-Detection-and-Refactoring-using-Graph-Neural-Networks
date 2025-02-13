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
