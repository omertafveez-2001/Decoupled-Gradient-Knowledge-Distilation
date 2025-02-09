# Decoupled-Gradient-Knowledge-Distillation

This repository implements *Decoupled Gradient Knowledge Distillation*: an extension of the existing Knowledge Distillation technique known as "Decoupled Knowledge Distillation." [Paper Link](https://arxiv.org/abs/2203.08679) <br> 
In this implementation, we introduce the gradients of the Target Class Logits and Non-Target class logits in the loss function and maximize their mean-squared error loss. The loss function is given as follows. 

```math 
\text{Loss\_{DKD}} = \alpha * TCKD + \beta * NCKD - \epsilon \text{MSE(Target class gradients, Non target class gradients)}  
```

While it is non-trivial to maximize mean-squared-error loss, the experimentations led to better performance on maximizing this term. This was because maximizing the mean-squared error loss led to the student's target-class logits matching closely with the teacher's target class logits, therefore matching the **fidelity** of the system. <br> 

We noticed the following observations by maximizing the mean-squared error term in the loss function:
- The loss remains nearly identical whether the student's target class logit is confident enough to make a correct prediction or highly confident. For instance, consider **Scenario A**, where the student's target class probability is **0.6** and the teacher's is **0.8**, and **Scenario B**, where these probabilities are reversed. In both cases, the loss value remains the same. This indicates that the loss encourages confidence in the target class **only up to the threshold needed for a correct prediction**, preventing excessive overfitting.  

- When the student makes a correct prediction but the teacher does not, the **mean-squared error (MSE) loss** is higher. This suggests that the loss function prioritizes aligning the student's logits with the teacher’s, emphasizing fidelity over correctness alone.  

- The student's **intra-class feature representations** trained with this loss function are more compact compared to prior methods in the literature. This leads to a **faster neural collapse**, improving overall model efficiency and generalization. 



