**INTRODUCTION**

In the field of software development, developers strive to write efficient,readable and maintainable code.
However, in real world, we have constraints such as time,budget,etc..., which often force compromises.
This leads to **Technical Debt**.This TD refers to a hidden cost that makes future maintenance more difficult.
One of the most serious forms of TD is code smells.This code smells refers to confusing,complex,
or poorly designed patterns in source code.Among the most well-known code smells are : **Long Methods** and **God Classes** .
A Long method performs too many tasks and becomes excessively lengthy, while a God Class takes too many responsibilitiesviolating
the **Single Responsibility Principle**.Studies show that these two code smells frequently appear together,making the code harder to manage.
One of the most common and problematics code smells is **Featur Envy**. This occurs when a method in one class becomes overly reliant on another class,
calling its method or accessing its attributes way too much. This reduces **Cohesion** within its class and increases **Coupling** between different classes,
making the code less maintainable. Coupling occurs when changes in one class directly impact another class. 
The best way to fix Feature Envy is to move the method to the class it's really interested in, which helps improve the code's organization and maintainability.
but spotting and fixing the feature envy is not manageable in the long run , making it a significant aspect of Technical Debt that developers must address. 
