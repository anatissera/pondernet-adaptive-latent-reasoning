# PonderNet: Learning to Ponder

**Authors:** Andrea Banino*, Jan Balaguer*, Charles Blundell

DeepMind, London, UK  
📧 abanino@deepmind.com · jan@deepmind.com · cblundell@deepmind.com

*contributed equally  
© 2021 Andrea Banino, Jan Balaguer, Charles Blundell

**arXiv:** 2107.05407v2 [cs.LG] — 2 Sep 2021  
**Venue:** 8th ICML Workshop on Automated Machine Learning (2021)

---

## Abstract

In standard neural networks the amount of computation used grows with the size of the inputs, but not with the complexity of the problem being learnt. To overcome this limitation we introduce PonderNet, a new algorithm that learns to adapt the amount of computation based on the complexity of the problem at hand. PonderNet learns end-to-end the number of computational steps to achieve an effective compromise between training prediction accuracy, computational cost and generalization. On a complex synthetic problem, PonderNet dramatically improves performance over previous adaptive computation methods and additionally succeeds at extrapolation tests where traditional neural networks fail. Also, our method matched the current state of the art results on a real world question and answering dataset, but using less compute. Finally, PonderNet reached state of the art results on a complex task designed to test the reasoning capabilities of neural networks.

---

## 1 Introduction

The time required to solve a problem is a function of more than just the size of the inputs. Commonly problems also have an inherent complexity that is independent of the input size: it is faster to add two numbers than to divide them. Most machine learning algorithms do not adjust their computational budget based on the complexity of the task they are learning to solve, or arguably, such adaptation is done manually by the machine learning practitioner. This adaptation is known as pondering. In prior work, Adaptive Computation Time (ACT; Graves, 2016) automatically learns to scale the required computation time via a scalar halting probability. This halting probability modulates the number of computational steps, called the "ponder time", needed for each input. Unfortunately ACT is notably unstable and sensitive to the choice of a hyper-parameter that trades-off accuracy and computation cost. Additionally, the gradient for the cost of computation can only back-propagate through the last computational step, leading to a biased estimation of the gradient. Another approach is represented by Adaptive Early Exit Networks (Bolukbasi et al., 2017) where the forward pass of an existing network is terminated at evaluation time if it is likely that the part of the network used so far already predicts the correct answer. More recently, work has investigated the use of REINFORCE (Williams, 1992) to perform conditional computation. A discrete latent variable is used to dynamically adjust the number of computation steps. This approach has been applied to recurrent neural networks (Chung et al., 2016; Banino et al., 2020), but has the downside that the estimated gradients have high variance, requiring large batch sizes to train them.

In this paper we present PonderNet that builds on these previous ideas. PonderNet is fully differentiable which allows for low-variance gradient estimates (unlike REINFORCE). It has unbiased gradient estimates (unlike ACT). We achieve this by reformulating the halting policy as a probabilistic model. This has consequences in all aspects of the model:

1. **Architecture**: in PonderNet, the halting node predicts the probability of halting conditional on not having halted before. We exactly compute the overall probability of halting at each step as a geometric distribution.
2. **Loss**: we don't regularize PonderNet to explicitly minimize the number of computing steps, but incentivize exploration instead. The pressure of using computation efficiently happens naturally as a form of Occam's razor.
3. **Inference**: PonderNet is probabilistic both in terms of number of computational steps and the prediction produced by the network.

---

## 2 Methods

### 2.1 Problem setting

We consider a supervised setting, where we want to learn a function f : x → y from data (x, y), with x = {x^{(1)}, ..., x^{(k)}} and y = {y^{(1)}, ..., y^{(k)}}. We propose a new general architecture for neural networks that modifies the forward pass, as well as a novel loss function to train it.

### 2.2 Step recurrence and halting process

The *PonderNet* architecture requires a *step function* s of the form ŷ_n, h_{n+1}, λ_n = s(x, h_n), as well as an initial state h₀. The output ŷ_n and λ_n are respectively the network's prediction and scalar probability of halting at step n. The step function s can be any neural network, such as MLPs, LSTMs, or encoder-decoder architectures such as transformers. We apply the step function recurrently up to N times.

The output ŷ_n is a learned prediction conditioned on the dynamic number of steps n ∈ {1, ..., N}. We rely on the value of λ_n to learn the optimal value of n. We define a Bernoulli random variable Λ_n in order to represent a Markov process for the halting with two states "continue" (Λ_n = 0) and "halt" (Λ_n = 1). The decision process starts from state "continue" (Λ₀ = 0). We set the transition probability:

P(Λ_n = 1 | Λ_{n-1} = 0) = λ_n,  ∀ 1 ≤ n ≤ N   (1)

that is the conditional probability of entering state "halt" at step n conditioned that there has been no previous halting. Note that "halt" is a terminal state. We can then estimate the unconditioned probability that the halting happened in steps 0, 1, 2, ..., N where N is the maximum number of steps allowed before halting. We derive this probability distribution p_n as a generalization of the geometric distribution:

p_n = λ_n ∏_{j=1}^{n-1} (1 - λ_j)   (2)

which is a valid probability distribution if we integrate over an infinite number of possible computation steps (N → ∞).

The prediction ŷ ~ Ŷ made by PonderNet is sampled from a random variable Ŷ with probability distribution P(Ŷ = y_n) = p_n. In other words, the prediction of PonderNet is the prediction made at the step n at which it halts. This is in contrast with ACT, where model predictions are always weighted averages across steps. Additionally, PonderNet is more generic in this regard: if one wishes to do so, it is straightforward to calculate the expected prediction across steps, similar to how it is done in ACT.

### 2.3 Maximum number of pondering steps

Since in practice we can only unroll the step function for a limited number of iterations, we must correct for this so that the sum of probabilities p_n sums to 1. We can do this in two ways. One option here is to normalize the probabilities p_n so that they sum up to 1 (this is equivalent to conditioning the probability of halting under the knowledge that n ≤ N). Alternatively, we could assign any remaining halting probability to the last step, so that p_N = 1 - ∑_{n=1}^{N-1} p_n instead of as previously defined.

In our experiments, we specify the maximum number of steps using two different criteria. In evaluation, and under known temporal or computational limitations, N can be set naively as a constant (or not set any limit, i.e. N → ∞). For training, we found that a more effective (and interpretable) way of parameterizing N is by defining a minimum cumulative probability of halting. N is then the smallest value of n such that ∑_{j=1}^{n} p_j > 1 - ε, with the hyper-parameter ε positive near 0 (in our experiments 0.05).

### 2.4 Training loss

The total loss is composed of reconstruction L_{Rec} and regularization L_{Reg} terms:

L = ∑_{n=1}^{N} p_n L(y, ŷ_n)  +  β · KL(p_n ‖ p_G(λ_p))   (3)
         ̲L̲_̲R̲e̲c̲              ̲L̲_̲R̲e̲g̲

where L is a pre-defined loss for the prediction (usually mean squared error, or cross-entropy); and λ_p is a hyper-parameter that defines a geometric prior distribution p_G(λ_p) on the halting policy (truncated at N). L_{Rec} is the expectation of the pre-defined reconstruction loss L across halting steps. L_{Reg} is the KL divergence between the distribution of halting probabilities p_n and the prior (a geometric distribution truncated at N, parameterized by λ_p). This hyper-parameter defines a prior on how likely it is that the network will halt at each step. This regularisation serves two purposes. First, it biases the network towards the expected prior number of steps 1/λ_p. Second, it provides an incentive to give a non-zero probability to all possible number of steps, thus promoting exploration.

### 2.5 Evaluation sampling

At evaluation, the network samples on a step basis from the halting Bernoulli random variable Λ_n ~ B(p = λ_n) to decide whether to continue or to halt. This process is repeated on every step n until a "halt" outcome is sampled, at which point the output y = y_n becomes the final prediction of the network. If a maximum number of steps N is reached, the network is automatically halted and produces a prediction y = y_N.

---

## 3 Results

### 3.1 Parity

In this section we are reporting results on the parity task as introduced in the original ACT paper (Graves, 2016). Out of the four tasks presented in that paper we decided to focus on parity as it was the one showing greater benefit from adaptive compute. In our instantiation of the parity problem the input vectors had 64 elements, of which a random number from 1 to 64 were randomly set to 1 or −1 and the rest were set to 0. The corresponding target was 1 if there was an odd number of ones and 0 if there was an even number of ones.

In figure 1a we can see that PonderNet achieved better accuracy than ACT on the parity task and it did so with a more efficient use of thinking time (1a at the bottom). Moreover, if we consider the total computation time during training (figure 1c) we can see that, in comparison to ACT, PonderNet employed less computation and achieved higher score.

Another analysis we performed on this version of the parity task was to look at the effect of the prior probability on performance. In figure 2b we show that the only case where PonderNet could not solve the task is when the prior (λ_p) was set to 0.9, that is when the average number of thinking steps given as prior was roughly 1 (1/0.9). Interestingly, when the prior (λ_p) was set to 0.1, hence starting with a prior average thinking time of 10 steps (1/0.1), the network managed to overcome this and settled to a more efficient average thinking time of roughly 3 steps (figure 2c). These results are important as they show that our method is particularly robust with respect to the prior, and a clear advancement in comparison to ACT, where the τ parameter is difficult to set and it is a source of training instability. One advantage of setting a prior probability is that this parameter is easy to interpret as the inverse of the "number of ponder steps", whereas the τ parameter does not have any straightforward interpretation, which makes it harder to define a priori.

Next we moved to test the ability of PonderNet to allow extrapolation. To do this we consider input vectors of 96 elements instead. We train the network on input vectors up from integers ranging from 1 to 48 elements and we then evaluate on integers between 49 and 96. Figure 1b shows that PonderNet was able to achieve almost perfect accuracy on this hard extrapolation task, whereas ACT remained at chance level. It is interesting to see how PonderNet increased its thinking time to 5 steps, which is almost twice as much as the ones in the interpolation set (see Fig. 1a), showing the capability of our method to adapt its computation to the complexity of the task.

### 3.2 bAbI

We then turn our attention to the bAbI question answering dataset (Weston et al., 2015), which consists of 20 different tasks. This task was chosen as it proved to be difficult for standard neural network architecture that do not employ adaptive computation (Dehghani et al., 2018). In particular we trained our model on the joint 10k training set.

Table 1 reports the averaged accuracy of our model and the other baselines on bAbI. Our model is able to match state of the art results, but it achieves them faster and with a lower average error. The comparison with Universal transformer (Dehghani et al., 2018, UT) is interesting as it uses the same transformer architecture as PonderNet, but the compute time is optimised with ACT. Interestingly, to solve 20 tasks, Universal Transformer takes 10161 steps, whereas our methods 1658, hence confirming that approach uses less compute than ACT.

| Architecture                                  | Average Error | Tasks Solved |
|----------------------------------------------|---------------|--------------|
| Memory Networks (Sukhbaatar et al., 2015)    | 4.2 ± 0.2     | 17           |
| DNC (Graves, 2016)                           | 3.8 ± 0.6     | 18           |
| Universal Transformer (Dehghani et al., 2018)| 0.29 ± 1.4    | 20           |
| Transformer + PonderNet                       | **0.15 ± 0.9**| 20           |

### 3.3 Paired associative inference

Finally, we tested PonderNet on the Paired associative inference task (PAI) (Banino et al., 2020). This task is thought to capture the essence of reasoning – the appreciation of distant relationships among elements distributed across multiple facts or memories and it has been shown to benefit from the addition of adaptive computation.

Table 2 reports the averaged accuracy of our model and the other baselines on PAI. Our model is able to match the results of MEMO, which was specifically designed with this task in mind. More interestingly, our model although is using the same architecture as UT (Dehghani et al., 2018) is able to achieve higher accuracy.

| Length                                        | UT    | MEMO          | PonderNet       |
|----------------------------------------------|-------|---------------|-----------------|
| 3 items (trained on: A-B-C - accuracy on A-C)| 85.60 | 98.26(0.67)   | **97.86(3.78)** |

---

## 4 Discussion

We introduced PonderNet, a new algorithm for learning to adapt the computational complexity of neural networks. It optimizes a novel objective function that combines prediction accuracy with a regularization term that incentivizes exploration over the pondering time. We demonstrated on the parity task that a neural network equipped with PonderNet can increase its computation to extrapolate beyond the data seen during training. Also, we showed that our methods achieved the highest accuracy in complex domains such as question answering and multi-step reasoning. Finally, adapting existing recurrent architectures to work with PonderNet is very easy: it simply requires to augment the step function with an additional halting unit, and to add an extra term to the loss. Critically, we showed that this extra loss term is robust to the choice of λ_p, the hyper-parameter that defines a prior on how likely is that the network will halt, which is an important advancement over ACT.

---

## References

- Banino et al. (2020). MEMO: A deep network for flexible combination of episodic memories. In ICLR, 2020.
- Bolukbasi et al. (2017). Adaptive neural networks for efficient inference. In Proceedings of the 34th ICML.
- Campos Camunez et al. (2018). Skip RNN: learning to skip state updates in recurrent neural networks. In ICLR, 2018.
- Chung et al. (2016). Hierarchical multiscale recurrent neural networks. arXiv:1609.01704.
- Dehghani et al. (2018). Universal transformers. arXiv:1807.03819.
- Graves (2016). Adaptive computation time for recurrent neural networks. arXiv:1603.08983.
- Kingma and Ba (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
- Sukhbaatar et al. (2015). End-to-end memory networks. In NeurIPS.
- Vaswani et al. (2017). Attention is all you need. In NeurIPS.
- Veličković et al. (2019). Neural execution of graph algorithms. arXiv:1910.10593.
- Weston et al. (2015). Towards AI-complete question answering: A set of prerequisite toy tasks. arXiv:1502.05698.
- Williams (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4):229–256.
- Yu et al. (2017). Learning to skim text. arXiv:1704.06877.

---

## Appendix A. Comparison to ACT

PonderNet builds on the ideas introduced in Adaptive Computation Time (ACT; Graves, 2016). The main contribution of this paper is to reformulate how the network learns to halt in a probabilistic way. This has consequences in all aspects of the model, including: the architecture and forward computation; the loss used to train the network; the deployment of the model; and the limitation of how multiple pondering modules can be combined.

### A.1 Forward computation

PonderNet's step function (that is computed on every step) is identical to the one proposed in ACT. They both assume a mapping y_n, h_{n+1}, λ_n = s(x, h_n). The main difference between ACT and PonderNet's forward computation is how the halting node λ_n is used.

In ACT, the network is unrolled for a number of steps N_{ACT} = min{N : ∑_{n=1}^{N} λ_n ≥ 1 − ε}. ACT's halting nodes learn to predict the overall probability that the network halted at step n, so that λ_n = p_n. The value of the halting node in the last step is replaced with a *remainder* quantity R = λ_N = p_N = 1 − ∑_{n=1}^{N-1} λ_n.

In PonderNet, any sufficiently high value of N can be used, and the unroll length of the network at training is distinguished from the learning of the halting policy (which is most critical for saving computation when deployed at evaluation).

The output of ACT is not treated probabilistically but as a weighted average ŷ_{ACT} = ∑_{n=1}^{N_{ACT}} ŷ_n λ_n over the outputs at each step. The halting, as well as the output, are computed identically for training and evaluation. In PonderNet, the output is probabilistic. In training, we compute the output and halting probabilities across many steps so that we can compute a weighted average of the *loss*. In evaluation, the network returns its prediction as soon as a halt state is sampled.

### A.2 Training loss

ACT proposes a heuristic training loss that combines two intuitive costs: the accuracy of the model, and the cost of computation. These two costs are in different units, and not easily comparable. Since N_{ACT} is not differentiable with respect to λ_n, ACT utilizes the remainder R = 1 − ∑_{n=1}^{N-1} as a proxy for minimizing the total number of computational steps.

In PonderNet, however, we propose that naively minimizing the number of steps (subject to good performance) is not necessarily a good objective. Instead, we propose that matching a prior halting distribution has multiple benefits: a) it provides an incentive for exploring alternative halting strategies; b) it provides robustness of the learnt step function, which may improve generalization; c) the KL is in same units as information-theoretic losses such as cross-entropy; and d) it provides an incentive to not ponder for longer than the prior.

Note that in PonderNet, we compute the loss for every possible number of computational steps, and then minimize the expectation (weighted average) over those. This is unlike in ACT where the expectation is taken over the predictions, and a loss is computed by comparing the average prediction with the target.

In PonderNet we have introduced two loss hyper-parameters λ_p and β, in comparison to a single hyper-parameter τ in ACT that trades-off accuracy with computational complexity. We note that, while τ and β are superficially similar (they both apply a weight to the regularization term), their effect is not equivalent because the regularization of ACT and PonderNet have different interpretation.

### A.3 Evaluation

ACT's predictions are computed identically during training and evaluation. In both contexts, the maximum number of steps N_{ACT} is determined based on the inputs, and the prediction is computed as a weighted average over the predictions in all steps. In PonderNet, training and evaluation are performed differently. During evaluation, the network halts probabilistically by sampling Λ_n, and either outputs the current prediction or performs an additional computational step.

---

## Appendix B. Parity.

### B.1 Training and evaluation details

For this experiment we used the Parity task as explained by Graves (2016). All the models used the same architecture, a simple RNN with a single hidden layer containing 128 tanh units and a single logistic sigmoid output unit. All models were optimized using Adam (Kingma and Ba, 2014), with learning rate fixed to 0.0003. For PonderNet we sampled uniformly 10 values of λ_p in the range (0, 1]. For ACT we sampled uniformly 19 values of τ in the range [2e-4, 2e-2] and we added also 0, which correspond to not penalising the halting unit at all. For both ACT and Ponder, N was set to 20. For PonderNet β was fixed to 0.01.

---

## Appendix C. bAbI.

### C.1 Training and evaluation details

For this experiment we used the English Question Answer dataset (Weston et al., 2015). All text is converted to lowercase; periods and interrogation marks are ignored; blank spaces are taken as word separation tokens; commas only appear in answers, and they are *not* ignored; all the questions are stripped out from the text and put separately (given as "queries" to our system).

At training time, we sample a mini-batch of 128 queries from the training dataset. For evaluation, we sample a batch of 10,000 elements from the dataset. The network was trained for 2e4 epochs, each one formed by 100 batch updates.

### C.2 Transformer architecture and hyperparameters

We use the same architecture as described in Dehghani et al. (2018), the 'universal_transformer_small' configuration.

**Hyperparameters used for bAbI experiments:**

| Parameter name             | Value                       |
|---------------------------|-----------------------------|
| Optimizer algorithm        | Adam                        |
| Learning rate              | 3e-4                        |
| Input embedding size       | 128                         |
| Attention type             | as in Vaswani et al. (2017) |
| Attention hidden size      | 512                         |
| Attention number of heads  | 8                           |
| Transition function        | MLP(1 Layer)                |
| Transition hidden size     | 128                         |
| Attention dropout rate     | 0.1                         |
| Activation function        | RELU                        |
| N                          | 10                          |
| β                          | 0.01                        |

---

## Appendix D. Paired Associative Inference

### D.1 PAI - Task details

For this task we used the dataset published in Banino et al. (2020). To build the dataset, Banino et al. (2020) started from raw images from the ImageNet dataset (Deng et al., 2009), which were embedded using a pre-trained ResNet (He et al., 2016), resulting in embeddings of size 1000. Here we are focusing on the dataset with sequences of length three (i.e. A − B − C) items.

A single entry in the batch is built by selecting N = 16 sequences from the relevant pool (e.g. training) and it's composed by three items: a memory, a query, and a target. Each memory content is created by storing all the possible pairwise association between the items in the sequence, e.g. A₁B₁ and B₁C₁, A₂B₂ and B₂C₂, ..., A_N B_N and B_N C_N. With N = 16, this process results in a memory with M = 32 rows each one with 2 embeddings of size 1000. Each query is composed of 3 images: the cue, the match, and the lure.

The queries are presented to the network as a concatenation of three image embedding vectors (the cue, the match and the lure), that is a 3 × 1000 dimensional vector.

**Hyperparameters used for PAI experiments:**

| Parameter name             | Value                       |
|---------------------------|-----------------------------|
| Optimizer algorithm        | Adam                        |
| Input embedding size       | 256                         |
| Attention type             | as in Vaswani et al. (2017) |
| Attention hidden size      | 512                         |
| Attention number of heads  | 8                           |
| Transition function        | MLP(2 Layers)               |
| Transition hidden size     | 128                         |
| Attention dropout rate     | 0.1                         |
| β                          | 0.01                        |

### D.3 PAI - Results based on query type

**Table 7: Paired Associative - length 3: A-B-C**

| Trial Type | MEMO          | UT    | PonderNet       |
|-----------|---------------|-------|-----------------|
| A-B       | 99.82(0.30)   | 97.43 | 98.01(2.39)     |
| B-C       | 99.76(0.38)   | 98.28 | 97.43(1.97)     |
| A-C       | 98.26(0.67)   | 85.60 | **97.86(3.78)** |

---

## Appendix E. Broader impact statement

In this work we introduced PonderNet, a new method that enables neural networks to adapt their computational complexity to the task they are trying to solve. Neural networks achieve state of the art in a wide range of applications, including natural language processing, reinforcement learning, computer vision and more. Currently, they require much time, expensive hardware and energy to train and to deploy. They also often fail to generalize and to extrapolate to conditions beyond their training.

PonderNet expands the capabilities of neural networks, by letting them decide to ponder for an indefinite amount of time (analogous to how both humans and computers think). This can be used to reduce the amount of training compute and energy at inference time, which makes it particularly well suited for platforms with limited resources such as mobile phones. Additionally, our experiments show that enabling neural networks to adapt their computational complexity has also benefits for their performance (beyond the computational requirements) when evaluating outside of the training distribution, which is one of the limiting factors when applying neural networks for real-world problems.

We encourage other researchers to pursue the questions we have considered on this work. We believe that biasing neural network architectures to behave more like algorithms, and less like "flat" mappings, will help develop deep learning methods to their the full potential.

---

## Key Concepts Summary

| Concept                  | PonderNet Definition                                                      |
|--------------------------|---------------------------------------------------------------------------|
| **Step function** s      | Shared recurrent module: (ŷ_n, h_{n+1}, λ_n) = s(x, h_n)                |
| **λ_n**                  | Probability of halting at step n, given no prior halt                     |
| **p_n**                  | Unconditioned probability of halting exactly at step n (geometric-like)   |
| **L_{Rec}**              | Expectation of task loss across halting steps: ∑ p_n L(y, ŷ_n)          |
| **L_{Reg}**              | KL divergence from halting distribution p_n to geometric prior p_G(λ_p) |
| **λ_p**                  | Prior hyper-parameter; 1/λ_p = expected prior number of steps            |
| **β**                    | Weight on the regularization term L_{Reg}                                |
| **N**                    | Maximum pondering steps (set by ε threshold during training)             |
| **Inference**            | Sequential Bernoulli sampling on λ_n until first "halt"                  |
| **vs. ACT**              | Fully differentiable; unbiased gradients; probabilistic output           |
