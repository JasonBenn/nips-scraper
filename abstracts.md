### #1: Wider and Deeper, Cheaper and Faster: Tensorized LSTMs for Sequence Learning

### #2: Concentration of Multilinear Functions of the Ising Model with Applications to Network Data

### #3: Deep Subspace Clustering Network

### #4: Attentional Pooling for Action Recognition

### #5: On the Consistency of Quick Shift

### #6: Rethinking Feature Discrimination and Polymerization for Large-scale Recognition

### #7: Breaking the Nonsmooth Barrier: A Scalable Parallel Method for Composite Optimization
_Fabian Pedregosa,  Rémi Leblond,  Simon Lacoste-Julien_

Due to their simplicity and excellent performance, parallel asynchronous variants of stochastic gradient descent have become popular methods to solve a wide range of large-scale optimization problems on multi-core architectures. Yet, despite their practical success, support for nonsmooth objectives is still lacking, making them unsuitable for many problems of interest in machine learning, such as the Lasso, group Lasso or empirical risk minimization with convex constraints. In this work, we propose and analyze ProxASAGA, a fully asynchronous sparse method inspired by SAGA, a variance reduced incremental gradient algorithm. The proposed method is easy to implement and significantly outperforms the state of the art on several nonsmooth, large-scale problems. We prove that our method achieves a theoretical linear speedup with respect to the sequential version under assumptions on the sparsity of gradients and block-separability of the proximal term. Empirical benchmarks on a multi-core architecture illustrate practical speedups of up to 12x on a 20-core machine.
[Abstract](https://arxiv.org/abs/1707.06468), [PDF](https://arxiv.org/pdf/1707.06468)


### #8: Dual-Agent GANs for Photorealistic and Identity Preserving Profile Face Synthesis

### #9: Dilated Recurrent Neural Networks

### #10: Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs

### #11: Scalable Generalized Linear Bandits: Online Computation and Hashing
_Kwang-Sung Jun,  Aniruddha Bhargava,  Robert Nowak,  Rebecca Willett_

Generalized Linear Bandits (GLBs), a natural extension of the stochastic linear bandits, has been popular and successful in recent years. However, existing GLBs scale poorly with the number of rounds and the number of arms, limiting their utility in practice. This paper proposes new, scalable solutions to the GLB problem in two respects. First, unlike existing GLBs, whose per-time-step space and time complexity grow at least linearly with time $t$, we propose a new algorithm that performs online computations to enjoy a constant space and time complexity. At its heart is a novel Generalized Linear extension of the Online-to-confidence-set Conversion (GLOC method) that takes \emph{any} online learning algorithm and turns it into a GLB algorithm. As a special case, we apply GLOC to the online Newton step algorithm, which results in a low-regret GLB algorithm with much lower time and memory complexity than prior work. Second, for the case where the number $N$ of arms is very large, we propose new algorithms in which each next arm is selected via an inner product search. Such methods can be implemented via hashing algorithms (i.e., "hash-amenable") and result in a time complexity sublinear in $N$. While a Thompson sampling extension of GLOC is hash-amenable, its regret bound for $d$-dimensional arm sets scales with $d^{3/2}$, whereas GLOC's regret bound is linear in $d$. Towards closing this gap, we propose a new hash-amenable algorithm whose regret bound scales with $d^{5/4}$. Finally, we propose a fast approximate hash-key computation (inner product) that has a better accuracy than the state-of-the-art, which can be of independent interest. We conclude the paper with preliminary experimental results confirming the merits of our methods.
[Abstract](https://arxiv.org/abs/1706.00136), [PDF](https://arxiv.org/pdf/1706.00136)


### #12: Probabilistic Models for Integration Error in the Assessment of Functional Cardiac Models

### #13:  Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent

### #14: Dynamic Safe Interruptibility for Decentralized Multi-Agent Reinforcement Learning

### #15: Interactive Submodular Bandit

### #16: Scene Physics Acquisition via Visual De-animation

### #17: Label Efficient Learning of Transferable Representations acrosss Domains and Tasks

### #18: Decoding with Value Networks for Neural Machine Translation

### #19: Parametric Simplex Method for Sparse Learning

### #20: Group Sparse Additive Machine

### #21: Uprooting and Rerooting Higher-order Graphical Models

### #22: The Unreasonable Effectiveness of Structured Random Orthogonal Embeddings

### #23: From Parity to Preference: Learning with Cost-effective Notions of Fairness

### #24: Inferring Generative Model Structure with Static Analysis
_Paroma Varma,  Bryan He,  Payal Bajaj,  Imon Banerjee,  Nishith Khandwala,  Daniel L. Rubin,  Christopher Ré_

Obtaining enough labeled data to robustly train complex discriminative models is a major bottleneck in the machine learning pipeline. A popular solution is combining multiple sources of weak supervision using generative models. The structure of these models affects training label quality, but is difficult to learn without any ground truth labels. We instead rely on these weak supervision sources having some structure by virtue of being encoded programmatically. We present Coral, a paradigm that infers generative model structure by statically analyzing the code for these heuristics, thus reducing the data required to learn structure significantly. We prove that Coral's sample complexity scales quasilinearly with the number of heuristics and number of relations found, improving over the standard sample complexity, which is exponential in $n$ for identifying $n^{\textrm{th}}$ degree relations. Experimentally, Coral matches or outperforms traditional structure learning approaches by up to 3.81 F1 points. Using Coral to model dependencies instead of assuming independence results in better performance than a fully supervised model by 3.07 accuracy points when heuristics are used to label radiology data without ground truth labels.
[Abstract](https://arxiv.org/abs/1709.02477), [PDF](https://arxiv.org/pdf/1709.02477)


### #25: Structured Embedding Models for Grouped Data

### #26: A Linear-Time Kernel Goodness-of-Fit Test
_Wittawat Jitkrittum,  Wenkai Xu,  Zoltan Szabo,  Kenji Fukumizu,  Arthur Gretton_

We propose a novel adaptive test of goodness-of-fit, with computational cost linear in the number of samples. We learn the test features that best indicate the differences between observed samples and a reference model, by minimizing the false negative rate. These features are constructed via Stein's method, meaning that it is not necessary to compute the normalising constant of the model. We analyse the asymptotic Bahadur efficiency of the new test, and prove that under a mean-shift alternative, our test always has greater relative efficiency than a previous linear-time kernel test, regardless of the choice of parameters for that test. In experiments, the performance of our method exceeds that of the earlier linear-time test, and matches or exceeds the power of a quadratic-time kernel test. In high dimensions and where model structure may be exploited, our goodness of fit test performs far better than a quadratic-time two-sample test based on the Maximum Mean Discrepancy, with samples drawn from the model.
[Abstract](https://arxiv.org/abs/1705.07673), [PDF](https://arxiv.org/pdf/1705.07673)


### #27: Cortical microcircuits as gated-recurrent neural networks

### #28: k-Support and Ordered Weighted Sparsity for Overlapping Groups: Hardness and Algorithms

### #29: A simple model of recognition and recall memory

### #30: On Structured Prediction Theory with Calibrated Convex Surrogate Losses
_Anton Osokin,  Francis Bach,  Simon Lacoste-Julien_

We provide novel theoretical insights on structured prediction in the context of efficient convex surrogate loss minimization with consistency guarantees. For any task loss, we construct a convex surrogate that can be optimized via stochastic gradient descent and we prove tight bounds on the so-called "calibration function" relating the excess surrogate risk to the actual risk. In contrast to prior related work, we carefully monitor the effect of the exponential number of classes in the learning guarantees as well as on the optimization complexity. As an interesting consequence, we formalize the intuition that some task losses make learning harder than others, and that the classical 0-1 loss is ill-suited for general structured prediction.
[Abstract](https://arxiv.org/abs/1703.02403), [PDF](https://arxiv.org/pdf/1703.02403)


### #31: Best of Both Worlds: Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model
_Jiasen Lu,  Anitha Kannan,  Jianwei Yang,  Devi Parikh,  Dhruv Batra_

We present a novel training framework for neural sequence models, particularly for grounded dialog generation. The standard training paradigm for these models is maximum likelihood estimation (MLE), or minimizing the cross-entropy of the human responses. Across a variety of domains, a recurring problem with MLE trained generative neural dialog models (G) is that they tend to produce 'safe' and generic responses ("I don't know", "I can't tell"). In contrast, discriminative dialog models (D) that are trained to rank a list of candidate human responses outperform their generative counterparts; in terms of automatic metrics, diversity, and informativeness of the responses. However, D is not useful in practice since it can not be deployed to have real conversations with users. Our work aims to achieve the best of both worlds -- the practical usefulness of G and the strong performance of D -- via knowledge transfer from D to G. Our primary contribution is an end-to-end trainable generative visual dialog model, where G receives gradients from D as a perceptual (not adversarial) loss of the sequence sampled from G. We leverage the recently proposed Gumbel-Softmax (GS) approximation to the discrete distribution -- specifically, a RNN augmented with a sequence of GS samplers, coupled with the straight-through gradient estimator to enable end-to-end differentiability. We also introduce a stronger encoder for visual dialog, and employ a self-attention mechanism for answer encoding along with a metric learning loss to aid D in better capturing semantic similarities in answer responses. Overall, our proposed model outperforms state-of-the-art on the VisDial dataset by a significant margin (2.67% on recall@10).
[Abstract](https://arxiv.org/abs/1706.01554), [PDF](https://arxiv.org/pdf/1706.01554)


### #32: MaskRNN: Instance Level Video Object Segmentation

### #33: Gated Recurrent Convolution Neural Network for OCR

### #34: Towards Accurate Binary Convolutional Neural Network

### #35: Semi-Supervised Learning for Optical Flow with Generative Adversarial Networks

### #36: Learning a Multi-View Stereo Machine
_Abhishek Kar,  Christian Häne,  Jitendra Malik_

We present a learnt system for multi-view stereopsis. In contrast to recent learning based methods for 3D reconstruction, we leverage the underlying 3D geometry of the problem through feature projection and unprojection along viewing rays. By formulating these operations in a differentiable manner, we are able to learn the system end-to-end for the task of metric 3D reconstruction. End-to-end learning allows us to jointly reason about shape priors while conforming geometric constraints, enabling reconstruction from much fewer images (even a single image) than required by classical approaches as well as completion of unseen surfaces. We thoroughly evaluate our approach on the ShapeNet dataset and demonstrate the benefits over classical approaches as well as recent learning based methods.
[Abstract](https://arxiv.org/abs/1708.05375), [PDF](https://arxiv.org/pdf/1708.05375)


### #37: Phase Transitions in the Pooled Data Problem

### #38: Universal Style Transfer via Feature Transforms
_Yijun Li,  Chen Fang,  Jimei Yang,  Zhaowen Wang,  Xin Lu,  Ming-Hsuan Yang_

Universal style transfer aims to transfer any arbitrary visual styles to content images. Existing feed-forward based methods, while enjoying the inference efficiency, are mainly limited by inability of generalizing to unseen styles or compromised visual quality. In this paper, we present a simple yet effective method that tackles these limitations without training on any pre-defined styles. The key ingredient of our method is a pair of feature transforms, whitening and coloring, that are embedded to an image reconstruction network. The whitening and coloring transforms reflect a direct matching of feature covariance of the content image to a given style image, which shares similar spirits with the optimization of Gram matrix based cost in neural style transfer. We demonstrate the effectiveness of our algorithm by generating high-quality stylized images with comparisons to a number of recent methods. We also analyze our method by visualizing the whitened features and synthesizing textures via simple feature coloring.
[Abstract](https://arxiv.org/abs/1705.08086), [PDF](https://arxiv.org/pdf/1705.08086)


### #39: On the Model Shrinkage Effect of Gamma Process Edge Partition Models

### #40: Pose Guided Person Image Generation
_Liqian Ma,  Qianru Sun,  Xu Jia,  Bernt Schiele,  Tinne Tuytelaars,  Luc Van Gool_

This paper proposes the novel Pose Guided Person Generation Network (PG$^2$) that allows to synthesize person images in arbitrary poses, based on an image of that person and a novel pose. Our generation framework PG$^2$ utilizes the pose information explicitly and consists of two key stages: pose integration and image refinement. In the first stage the condition image and the target pose are fed into a U-Net-like network to generate an initial but coarse image of the person with the target pose. The second stage then refines the initial and blurry result by training a U-Net-like generator in an adversarial way. Extensive experimental results on both 128$\times$64 re-identification images and 256$\times$256 fashion photos show that our model generates high-quality person images with convincing details.
[Abstract](https://arxiv.org/abs/1705.09368), [PDF](https://arxiv.org/pdf/1705.09368)


### #41: Inference in Graphical Models via Semidefinite Programming Hierarchies

### #42: Variable Importance Using Decision Trees

### #43: Preventing Gradient Explosions in Gated Recurrent Units

### #44: On the Power of Truncated SVD for General High-rank Matrix Estimation Problems
_Simon S. Du,  Yining Wang,  Aarti Singh_

We show that given an estimate $\widehat{A}$ that is close to a general high-rank positive semi-definite (PSD) matrix $A$ in spectral norm (i.e., $\|\widehat{A}-A\|_2 \leq \delta$), the simple truncated SVD of $\widehat{A}$ produces a multiplicative approximation of $A$ in Frobenius norm. This observation leads to many interesting results on general high-rank matrix estimation problems, which we briefly summarize below ($A$ is an $n\times n$ high-rank PSD matrix and $A_k$ is the best rank-$k$ approximation of $A$): (1) High-rank matrix completion: By observing $\Omega(\frac{n\max\{\epsilon^{-4},k^2\}\mu_0^2\|A\|_F^2\log n}{\sigma_{k+1}(A)^2})$ elements of $A$ where $\sigma_{k+1}\left(A\right)$ is the $\left(k+1\right)$-th singular value of $A$ and $\mu_0$ is the incoherence, the truncated SVD on a zero-filled matrix satisfies $\|\widehat{A}_k-A\|_F \leq (1+O(\epsilon))\|A-A_k\|_F$ with high probability. (2)High-rank matrix de-noising: Let $\widehat{A}=A+E$ where $E$ is a Gaussian random noise matrix with zero mean and $\nu^2/n$ variance on each entry. Then the truncated SVD of $\widehat{A}$ satisfies $\|\widehat{A}_k-A\|_F \leq (1+O(\sqrt{\nu/\sigma_{k+1}(A)}))\|A-A_k\|_F + O(\sqrt{k}\nu)$. (3) Low-rank Estimation of high-dimensional covariance: Given $N$ i.i.d.~samples $X_1,\cdots,X_N\sim\mathcal N_n(0,A)$, can we estimate $A$ with a relative-error Frobenius norm bound? We show that if $N = \Omega\left(n\max\{\epsilon^{-4},k^2\}\gamma_k(A)^2\log N\right)$ for $\gamma_k(A)=\sigma_1(A)/\sigma_{k+1}(A)$, then $\|\widehat{A}_k-A\|_F \leq (1+O(\epsilon))\|A-A_k\|_F$ with high probability, where $\widehat{A}=\frac{1}{N}\sum_{i=1}^N{X_iX_i^\top}$ is the sample covariance.
[Abstract](https://arxiv.org/abs/1702.06861), [PDF](https://arxiv.org/pdf/1702.06861)


### #45: f-GANs in an Information Geometric Nutshell
_Richard Nock,  Zac Cranko,  Aditya Krishna Menon,  Lizhen Qu,  Robert C. Williamson_

Nowozin \textit{et al} showed last year how to extend the GAN \textit{principle} to all $f$-divergences. The approach is elegant but falls short of a full description of the supervised game, and says little about the key player, the generator: for example, what does the generator actually converge to if solving the GAN game means convergence in some space of parameters? How does that provide hints on the generator's design and compare to the flourishing but almost exclusively experimental literature on the subject? In this paper, we unveil a broad class of distributions for which such convergence happens --- namely, deformed exponential families, a wide superset of exponential families --- and show tight connections with the three other key GAN parameters: loss, game and architecture. In particular, we show that current deep architectures are able to factorize a very large number of such densities using an especially compact design, hence displaying the power of deep architectures and their concinnity in the $f$-GAN game. This result holds given a sufficient condition on \textit{activation functions} --- which turns out to be satisfied by popular choices. The key to our results is a variational generalization of an old theorem that relates the KL divergence between regular exponential families and divergences between their natural parameters. We complete this picture with additional results and experimental insights on how these results may be used to ground further improvements of GAN architectures, via (i) a principled design of the activation functions in the generator and (ii) an explicit integration of proper composite losses' link function in the discriminator.
[Abstract](https://arxiv.org/abs/1707.04385), [PDF](https://arxiv.org/pdf/1707.04385)


### #46: Multimodal Image-to-Image Translation by Enforcing Bi-Cycle Consistency

### #47: Mixture-Rank Matrix Approximation for Collaborative Filtering

### #48: Non-monotone Continuous DR-submodular  Maximization: Structure and Algorithms

### #49: Learning with Average Top-k Loss
_Yanbo Fan,  Siwei Lyu,  Yiming Ying,  Bao-Gang Hu_

In this work, we introduce the average top-$k$ (AT$_k$) loss as a new ensemble loss for supervised learning, which is the average over the $k$ largest individual losses over a training dataset. We show that the AT$_k$ loss is a natural generalization of the two widely used ensemble losses, namely the average loss and the maximum loss, but can combines their advantages and mitigate their drawbacks to better adapt to different data distributions. Furthermore, it remains a convex function over all individual losses, which can lead to convex optimization problems that can be solved effectively with conventional gradient-based method. We provide an intuitive interpretation of the AT$_k$ loss based on its equivalent effect on the continuous individual loss functions, suggesting that it can reduce the penalty on correctly classified data. We further give a learning theory analysis of MAT$_k$ learning on the classification calibration of the AT$_k$ loss and the error bounds of AT$_k$-SVM. We demonstrate the applicability of minimum average top-$k$ learning for binary classification and regression using synthetic and real datasets.
[Abstract](https://arxiv.org/abs/1705.08826), [PDF](https://arxiv.org/pdf/1705.08826)


### #50: Learning multiple visual domains with residual adapters
_Sylvestre-Alvise Rebuffi,  Hakan Bilen,  Andrea Vedaldi_

There is a growing interest in learning data representations that work well for many different types of problems and data. In this paper, we look in particular at the task of learning a single visual representation that can be successfully utilized in the analysis of very different types of images, from dog breeds to stop signs and digits. Inspired by recent work on learning networks that predict the parameters of another, we develop a tunable deep network architecture that, by means of adapter residual modules, can be steered on the fly to diverse visual domains. Our method achieves a high degree of parameter sharing while maintaining or even improving the accuracy of domain-specific representations. We also introduce the Visual Decathlon Challenge, a benchmark that evaluates the ability of representations to capture simultaneously ten very different visual domains and measures their ability to recognize well uniformly.
[Abstract](https://arxiv.org/abs/1705.08045), [PDF](https://arxiv.org/pdf/1705.08045)


### #51: Dykstra's Algorithm, ADMM, and Coordinate Descent: Connections, Insights, and Extensions
_Ryan J. Tibshirani_

We study connections between Dykstra's algorithm for projecting onto an intersection of convex sets, the augmented Lagrangian method of multipliers or ADMM, and block coordinate descent. We prove that coordinate descent for a regularized regression problem, in which the (separable) penalty functions are seminorms, is exactly equivalent to Dykstra's algorithm applied to the dual problem. ADMM on the dual problem is also seen to be equivalent, in the special case of two sets, with one being a linear subspace. These connections, aside from being interesting in their own right, suggest new ways of analyzing and extending coordinate descent. For example, from existing convergence theory on Dykstra's algorithm over polyhedra, we discern that coordinate descent for the lasso problem converges at an (asymptotically) linear rate. We also develop two parallel versions of coordinate descent, based on the Dykstra and ADMM connections.
[Abstract](https://arxiv.org/abs/1705.04768), [PDF](https://arxiv.org/pdf/1705.04768)


### #52: Flat2Sphere: Learning Spherical Convolution for Fast Features from 360° Imagery

### #53: 3D Shape Reconstruction by Modeling 2.5D Sketch

### #54: Multimodal Learning and Reasoning for Visual Question Answering

### #55: Adversarial Surrogate Losses for Ordinal Regression

### #56: Hypothesis Transfer Learning via Transformation Functions

### #57: Adversarial Invariant Feature Learning

### #58: Convergence Analysis of Two-layer Neural Networks with ReLU Activation

### #59: Doubly Accelerated Stochastic Variance Reduced Dual Averaging Method for Regularized Empirical Risk Minimization

### #60: Langevin Dynamics with Continuous Tempering for Training Deep Neural Networks

### #61: Efficient Online Linear Optimization with Approximation Algorithms

### #62: Geometric Descent Method for Convex Composite Minimization

### #63: Diffusion Approximations for Online Principal Component Estimation and Global Convergence

### #64:  Avoiding Discrimination through Causal Reasoning

### #65: Nonparametric Online Regression while Learning the Metric

### #66: Recycling for Fairness: Learning with Conditional Distribution Matching Constraints

### #67: Safe and Nested Subgame Solving for Imperfect-Information Games

### #68: Unsupervised Image-to-Image Translation Networks

### #69: Coded Distributed Computing for Inverse Problems

### #70: A Screening Rule for l1-Regularized Ising Model Estimation

### #71: Improved Dynamic Regret for Non-degeneracy Functions

### #72: Learning Efficient Object Detection Models with Knowledge Distillation

### #73: One-Sided Unsupervised Domain Mapping
_Sagie Benaim,  Lior Wolf_

In unsupervised domain mapping, the learner is given two unmatched datasets $A$ and $B$. The goal is to learn a mapping $G_{AB}$ that translates a sample in $A$ to the analog sample in $B$. Recent approaches have shown that when learning simultaneously both $G_{AB}$ and the inverse mapping $G_{BA}$, convincing mappings are obtained. In this work, we present a method of learning $G_{AB}$ without learning $G_{BA}$. This is done by learning a mapping that maintains the distance between a pair of samples. Moreover, good mappings are obtained, even by maintaining the distance between different parts of the same sample before and after mapping. We present experimental results that the new method not only allows for one sided mapping learning, but also leads to preferable numerical results over the existing circularity-based constraint. Our entire code is made publicly available at this https URL .
[Abstract](https://arxiv.org/abs/1706.00826), [PDF](https://arxiv.org/pdf/1706.00826)


### #74: Deep Mean-Shift Priors for Image Restoration
_Siavash Arjomand Bigdeli,  Meiguang Jin,  Paolo Favaro,  Matthias Zwicker_

In this paper we introduce a natural image prior that directly represents a Gaussian-smoothed version of the natural image distribution. We include our prior in a formulation of image restoration as a Bayes estimator that also allows us to solve noise-blind image restoration problems. We show that the gradient of our prior corresponds to the mean-shift vector on the natural image distribution. In addition, we learn the mean-shift vector field using denoising autoencoders, and use it in a gradient descent approach to perform Bayes risk minimization. We demonstrate competitive results for noise-blind deblurring, super-resolution, and demosaicing.
[Abstract](http://arxiv.org/abs/1709.03749), [PDF](http://arxiv.org/pdf/1709.03749)


### #75: Greedy Algorithms for Cone Constrained Optimization with Convergence Guarantees
_Francesco Locatello,  Michael Tschannen,  Gunnar Rätsch,  Martin Jaggi_

Greedy optimization methods such as Matching Pursuit (MP) and Frank-Wolfe (FW) algorithms regained popularity in recent years due to their simplicity, effectiveness and theoretical guarantees. MP and FW address optimization over the linear span and the convex hull of a set of atoms, respectively. In this paper, we consider the intermediate case of optimization over the convex cone, parametrized as the conic hull of a generic atom set, leading to the first principled definitions of non-negative MP algorithms for which we give explicit convergence rates and demonstrate excellent empirical performance. In particular, we derive sublinear ($\mathcal{O}(1/t)$) convergence on general smooth and convex objectives, and linear convergence ($\mathcal{O}(e^{-t})$) on strongly convex objectives, in both cases for general sets of atoms. Furthermore, we establish a clear correspondence of our algorithms to known algorithms from the MP and FW literature. Our novel algorithms and analyses target general atom sets and general objective functions, and hence are directly applicable to a large variety of learning settings.
[Abstract](https://arxiv.org/abs/1705.11041), [PDF](https://arxiv.org/pdf/1705.11041)


### #76: A New Theory for Nonconvex Matrix Completion

### #77: Robust Hypothesis Test for Functional Effect with Gaussian Processes

### #78: Lower bounds on the robustness to adversarial perturbations

### #79: Minimizing a Submodular Function from Samples

### #80: Introspective Classification with Convolutional Nets

### #81: Label Distribution Learning Forests

### #82: Unsupervised object learning from dense equivariant image labelling

### #83: Compression-aware Training of Deep Neural Networks

### #84: Multiscale Semi-Markov Dynamics for Intracortical Brain-Computer Interfaces

### #85: PredRNN: Recurrent Neural Networks for Video Prediction using Spatiotemporal LSTMs

### #86: Detrended Partial Cross Correlation for Brain Connectivity Analysis

### #87: Contrastive Learning for Image Captioning

### #88: Safe Model-based Reinforcement Learning with Stability Guarantees
_Felix Berkenkamp,  Matteo Turchetta,  Angela P. Schoellig,  Andreas Krause_

Reinforcement learning is a powerful paradigm for learning optimal policies from experimental data. However, to find optimal policies, most reinforcement learning algorithms explore all possible actions, which may be harmful for real-world systems. As a consequence, learning algorithms are rarely applied on safety-critical systems in the real world. In this paper, we present a learning algorithm that explicitly considers safety in terms of stability guarantees. Specifically, we extend control theoretic results on Lyapunov stability verification and show how to use statistical models of the dynamics to obtain high-performance control policies with provable stability certificates. Moreover, under additional regularity assumptions in terms of a Gaussian process prior, we prove that one can effectively and safely collect data in order to learn about the dynamics and thus both improve control performance and expand the safe region of the state space. In our experiments, we show how the resulting algorithm can safely optimize a neural network policy on a simulated inverted pendulum, without the pendulum ever falling down.
[Abstract](https://arxiv.org/abs/1705.08551), [PDF](https://arxiv.org/pdf/1705.08551)


### #89: Online multiclass boosting

### #90: Matching on Balanced Nonlinear Representations for Treatment Effects Estimation

### #91: Learning Overcomplete HMMs

### #92: GP CaKe: Effective brain connectivity with causal kernels
_Luca Ambrogioni,  Max Hinne,  Marcel van Gerven,  Eric Maris_

A fundamental goal in network neuroscience is to understand how activity in one region drives activity elsewhere, a process referred to as effective connectivity. Here we propose to model this causal interaction using integro-differential equations and causal kernels that allow for a rich analysis of effective connectivity. The approach combines the tractability and flexibility of autoregressive modeling with the biophysical interpretability of dynamic causal modeling. The causal kernels are learned nonparametrically using Gaussian process regression, yielding an efficient framework for causal inference. We construct a novel class of causal covariance functions that enforce the desired properties of the causal kernels, an approach which we call GP CaKe. By construction, the model and its hyperparameters have biophysical meaning and are therefore easily interpretable. We demonstrate the efficacy of GP CaKe on a number of simulations and give an example of a realistic application on magnetoencephalography (MEG) data.
[Abstract](https://arxiv.org/abs/1705.05603), [PDF](https://arxiv.org/pdf/1705.05603)


### #93: Decoupling "when to update" from "how to update"

### #94: Self-Normalizing Neural Networks
_Günter Klambauer,  Thomas Unterthiner,  Andreas Mayr,  Sepp Hochreiter_

Deep Learning has revolutionized vision via convolutional neural networks (CNNs) and natural language processing via recurrent neural networks (RNNs). However, success stories of Deep Learning with standard feed-forward neural networks (FNNs) are rare. FNNs that perform well are typically shallow and, therefore cannot exploit many levels of abstract representations. We introduce self-normalizing neural networks (SNNs) to enable high-level abstract representations. While batch normalization requires explicit normalization, neuron activations of SNNs automatically converge towards zero mean and unit variance. The activation function of SNNs are "scaled exponential linear units" (SELUs), which induce self-normalizing properties. Using the Banach fixed-point theorem, we prove that activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance -- even under the presence of noise and perturbations. This convergence property of SNNs allows to (1) train deep networks with many layers, (2) employ strong regularization, and (3) to make learning highly robust. Furthermore, for activations not close to unit variance, we prove an upper and lower bound on the variance, thus, vanishing and exploding gradients are impossible. We compared SNNs on (a) 121 tasks from the UCI machine learning repository, on (b) drug discovery benchmarks, and on (c) astronomy tasks with standard FNNs and other machine learning methods such as random forests and support vector machines. SNNs significantly outperformed all competing FNN methods at 121 UCI tasks, outperformed all competing methods at the Tox21 dataset, and set a new record at an astronomy data set. The winning SNN architectures are often very deep. Implementations are available at: github.com/bioinf-jku/SNNs.
[Abstract](https://arxiv.org/abs/1706.02515), [PDF](https://arxiv.org/pdf/1706.02515)


### #95: Learning to Pivot with Adversarial Networks
_Gilles Louppe,  Michael Kagan,  Kyle Cranmer_

Several techniques for domain adaptation have been proposed to account for differences in the distribution of the data used for training and testing. The majority of this work focuses on a binary domain label. Similar problems occur in a scientific context where there may be a continuous family of plausible data generation processes associated to the presence of systematic uncertainties. Robust inference is possible if it is based on a pivot -- a quantity whose distribution does not depend on the unknown values of the nuisance parameters that parametrize this family of data generation processes. In this work, we introduce and derive theoretical results for a training procedure based on adversarial networks for enforcing the pivotal property (or, equivalently, fairness with respect to continuous attributes) on a predictive model. The method includes a hyperparameter to control the trade-off between accuracy and robustness. We demonstrate the effectiveness of this approach with a toy example and examples from particle physics.
[Abstract](https://arxiv.org/abs/1611.01046), [PDF](https://arxiv.org/pdf/1611.01046)


### #96: MolecuLeNet: A continuous-filter convolutional neural network for modeling quantum interactions
_Kristof T. Schütt,  Pieter-Jan Kindermans,  Huziel E. Sauceda,  Stefan Chmiela,  Alexandre Tkatchenko,  Klaus-Robert Müller_

Deep learning has the potential to revolutionize quantum chemistry as it is ideally suited to learn representations for structured data and speed up the exploration of chemical space. While convolutional neural networks have proven to be the first choice for images, audio and video data, the atoms in molecules are not restricted to a grid. Instead, their precise locations contain essential physical information, that would get lost if discretized. Thus, we propose to use continuous-filter convolutional layers to be able to model local correlations without requiring the data to lie on a grid. We apply those layers in SchNet: a novel deep learning architecture modeling quantum interactions in molecules. We obtain a joint model for the total energy and interatomic forces that follows fundamental quantum-chemical principles. This includes rotationally invariant energy predictions and a smooth, differentiable potential energy surface. Our architecture achieves state-of-the-art performance for benchmarks of equilibrium molecules and molecular dynamics trajectories. Finally, we introduce a more challenging benchmark with chemical and structural variations that suggests the path for further work.
[Abstract](https://arxiv.org/abs/1706.08566), [PDF](https://arxiv.org/pdf/1706.08566)


### #97: Active Bias: Training a More Accurate Neural Network by Emphasizing High Variance Samples
_Haw-Shiuan Chang,  Erik Learned-Miller,  Andrew McCallum_

Self-paced learning and hard example mining re-weight training instances to improve learning accuracy. This paper presents two improved alternatives based on lightweight estimates of sample uncertainty in stochastic gradient descent (SGD): the variance in predicted probability of the correct class across iterations of mini-batch SGD, and the proximity of the correct class probability to the decision threshold. Extensive experimental results on six datasets show that our methods reliably improve accuracy in various network architectures, including additional gains on top of other popular training techniques, such as residual learning, momentum, ADAM, batch normalization, dropout, and distillation.
[Abstract](https://arxiv.org/abs/1704.07433), [PDF](https://arxiv.org/pdf/1704.07433)


### #98: Differentiable Learning of Submodular Functions

### #99: Inductive Representation Learning on Large Graphs
_William L. Hamilton,  Rex Ying,  Jure Leskovec_

Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during training of the embeddings; these previous approaches are inherently transductive and do not naturally generalize to unseen nodes. Here we present GraphSAGE, a general, inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, we learn a function that generates embeddings by sampling and aggregating features from a node's local neighborhood. Our algorithm outperforms strong baselines on three inductive node-classification benchmarks: we classify the category of unseen nodes in evolving information graphs based on citation and Reddit post data, and we show that our algorithm generalizes to completely unseen graphs using a multi-graph dataset of protein-protein interactions.
[Abstract](https://arxiv.org/abs/1706.02216), [PDF](https://arxiv.org/pdf/1706.02216)


### #100: Subset Selection for Sequential Data

### #101: Question Asking as Program Generation

### #102: Revisiting Perceptron: Efficient and Label-Optimal Learning of Halfspaces
_Songbai Yan,  Chicheng Zhang_

It has been a long-standing problem to efficiently learn a linear separator using as few labels as possible. In this work, we propose an efficient perceptron-based algorithm for actively learning homogeneous linear separators under uniform distribution. Under bounded noise, where each label is flipped with probability at most $\eta$, our algorithm achieves near-optimal $\tilde{O}\left(\frac{d}{(1-2\eta)^2}\log\frac{1}{\epsilon}\right)$ label complexity in time $\tilde{O}\left(\frac{d^2}{\epsilon(1-2\eta)^2}\right)$, and significantly improves over the best known result (Awasthi et al., 2016). Under adversarial noise, where at most $\nu$ fraction of labels can be flipped, our algorithm achieves near-optimal $\tilde{O}\left(d\log\frac{1}{\epsilon}\right)$ label complexity in time $\tilde{O}\left(\frac{d^2}{\epsilon}\right)$, which is better than the best known label complexity and time complexity in Awasthi et al. (2014).
[Abstract](https://arxiv.org/abs/1702.05581), [PDF](https://arxiv.org/pdf/1702.05581)


### #103: Gradient Descent Can Take Exponential Time to Escape Saddle Points
_Simon S. Du,  Chi Jin,  Jason D. Lee,  Michael I. Jordan,  Barnabas Poczos,  Aarti Singh_

Although gradient descent (GD) almost always escapes saddle points asymptotically [Lee et al., 2016], this paper shows that even with fairly natural random initialization schemes and non-pathological functions, GD can be significantly slowed down by saddle points, taking exponential time to escape. On the other hand, gradient descent with perturbations [Ge et al., 2015, Jin et al., 2017] is not slowed down by saddle points - it can find an approximate local minimizer in polynomial time. This result implies that GD is inherently slower than perturbed GD, and justifies the importance of adding perturbations for efficient non-convex optimization. While our focus is theoretical, we also present experiments that illustrate our theoretical findings.
[Abstract](https://arxiv.org/abs/1705.10412), [PDF](https://arxiv.org/pdf/1705.10412)


### #104: Union of Intersections (UoI) for Interpretable Data Driven Discovery and Prediction
_Kristofer E. Bouchard,  Alejandro F. Bujan,  Farbod Roosta-Khorasani,  Shashanka Ubaru,  Prabhat,  Antoine M. Snijders,  Jian-Hua Mao,  Edward F. Chang,  Michael W. Mahoney,  Sharmodeep Bhattacharyya_

The increasing size and complexity of scientific data could dramatically enhance discovery and prediction for basic scientific applications. Realizing this potential, however, requires novel statistical analysis methods that are both interpretable and predictive. We introduce Union of Intersections (UoI), a flexible, modular, and scalable framework for enhanced model selection and estimation. Methods based on UoI perform model selection and model estimation through intersection and union operations, respectively. We show that UoI-based methods achieve low-variance and nearly unbiased estimation of a small number of interpretable features, while maintaining high-quality prediction accuracy. We perform extensive numerical investigation to evaluate a UoI algorithm ($UoI_{Lasso}$) on synthetic and real data. In doing so, we demonstrate the extraction of interpretable functional networks from human electrophysiology recordings as well as accurate prediction of phenotypes from genotype-phenotype data with reduced features. We also show (with the $UoI_{L1Logistic}$ and $UoI_{CUR}$ variants of the basic framework) improved prediction parsimony for classification and matrix factorization on several benchmark biomedical data sets. These results suggest that methods based on the UoI framework could improve interpretation and prediction in data-driven discovery across scientific fields.
[Abstract](https://arxiv.org/abs/1705.07585), [PDF](https://arxiv.org/pdf/1705.07585)


### #105: One-Shot Imitation Learning
_Yan Duan,  Marcin Andrychowicz,  Bradly C. Stadie,  Jonathan Ho,  Jonas Schneider,  Ilya Sutskever,  Pieter Abbeel,  Wojciech Zaremba_

Imitation learning has been commonly applied to solve different tasks in isolation. This usually requires either careful feature engineering, or a significant number of samples. This is far from what we desire: ideally, robots should be able to learn from very few demonstrations of any given task, and instantly generalize to new situations of the same task, without requiring task-specific engineering. In this paper, we propose a meta-learning framework for achieving such capability, which we call one-shot imitation learning. Specifically, we consider the setting where there is a very large set of tasks, and each task has many instantiations. For example, a task could be to stack all blocks on a table into a single tower, another task could be to place all blocks on a table into two-block towers, etc. In each case, different instances of the task would consist of different sets of blocks with different initial states. At training time, our algorithm is presented with pairs of demonstrations for a subset of all tasks. A neural net is trained that takes as input one demonstration and the current state (which initially is the initial state of the other demonstration of the pair), and outputs an action with the goal that the resulting sequence of states and actions matches as closely as possible with the second demonstration. At test time, a demonstration of a single instance of a new task is presented, and the neural net is expected to perform well on new instances of this new task. The use of soft attention allows the model to generalize to conditions and tasks unseen in the training data. We anticipate that by training this model on a much greater variety of tasks and settings, we will obtain a general system that can turn any demonstrations into robust policies that can accomplish an overwhelming variety of tasks. Videos available at this https URL .
[Abstract](https://arxiv.org/abs/1703.07326), [PDF](https://arxiv.org/pdf/1703.07326)


### #106: Learning the Morphology of Brain Signals Using Alpha-Stable Convolutional Sparse Coding
_Mainak Jas,  Tom Dupré La Tour,  Umut Şimşekli,  Alexandre Gramfort_

Neural time-series data contain a wide variety of prototypical signal waveforms (atoms) that are of significant importance in clinical and cognitive research. One of the goals for analyzing such data is hence to extract such 'shift-invariant' atoms. Even though some success has been reported with existing algorithms, they are limited in applicability due to their heuristic nature. Moreover, they are often vulnerable to artifacts and impulsive noise, which are typically present in raw neural recordings. In this study, we address these issues and propose a novel probabilistic convolutional sparse coding (CSC) model for learning shift-invariant atoms from raw neural signals containing potentially severe artifacts. In the core of our model, which we call $\alpha$CSC, lies a family of heavy-tailed distributions called $\alpha$-stable distributions. We develop a novel, computationally efficient Monte Carlo expectation-maximization algorithm for inference. The maximization step boils down to a weighted CSC problem, for which we develop a computationally efficient optimization algorithm. Our results show that the proposed algorithm achieves state-of-the-art convergence speeds. Besides, $\alpha$CSC is significantly more robust to artifacts when compared to three competing algorithms: it can extract spike bursts, oscillations, and even reveal more subtle phenomena such as cross-frequency coupling when applied to noisy neural time series.
[Abstract](https://arxiv.org/abs/1705.08006), [PDF](https://arxiv.org/pdf/1705.08006)


### #107: Integration Methods and Optimization Algorithms

### #108: Sharpness, Restart and Acceleration
_Vincent Roulet,  Alexandre d'Aspremont_

The {\L}ojasievicz inequality shows that sharpness bounds on the minimum of convex optimization problems hold almost generically. Here, we show that sharpness directly controls the performance of restart schemes. The constants quantifying sharpness are of course unobservable, but we show that optimal restart strategies are fairly robust, and searching for the best scheme only increases the complexity by a logarithmic factor compared to the optimal bound. Overall then, restart schemes generically accelerate accelerated methods.
[Abstract](https://arxiv.org/abs/1702.03828), [PDF](https://arxiv.org/pdf/1702.03828)


### #109: Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition

### #110: Soft-to-Hard Vector Quantization for End-to-End Learning Compressible Representations
_Eirikur Agustsson,  Fabian Mentzer,  Michael Tschannen,  Lukas Cavigelli,  Radu Timofte,  Luca Benini,  Luc Van Gool_

We present a new approach to learn compressible representations in deep architectures with an end-to-end training strategy. Our method is based on a soft (continuous) relaxation of quantization and entropy, which we anneal to their discrete counterparts throughout training. We showcase this method for two challenging applications: Image compression and neural network compression. While these tasks have typically been approached with different methods, our soft-to-hard quantization approach gives results competitive with the state-of-the-art for both.
[Abstract](https://arxiv.org/abs/1704.00648), [PDF](https://arxiv.org/pdf/1704.00648)


### #111: Learning spatiotemporal piecewise-geodesic trajectories from longitudinal manifold-valued data

### #112: Improving Regret Bounds for Combinatorial Semi-Bandits with Probabilistically Triggered Arms and Its Applications

### #113: Predictive-State Decoders: Encoding the Future into Recurrent Networks

### #114: Posterior sampling for reinforcement learning: worst-case regret bounds
_Shipra Agrawal,  Randy Jia_

We present an algorithm based on posterior sampling (aka Thompson sampling) that achieves near-optimal worst-case regret bounds when the underlying Markov Decision Process (MDP) is communicating with a finite, though unknown, diameter. Our main result is a high probability regret upper bound of $\tilde{O}(D\sqrt{SAT})$ for any communicating MDP with $S$ states, $A$ actions and diameter $D$, when $T\ge S^5A$. Here, regret compares the total reward achieved by the algorithm to the total expected reward of an optimal infinite-horizon undiscounted average reward policy, in time horizon $T$. This result improves over the best previously known upper bound of $\tilde{O}(DS\sqrt{AT})$ achieved by any algorithm in this setting, and matches the dependence on $S$ in the established lower bound of $\Omega(\sqrt{DSAT})$ for this problem. Our techniques involve proving some novel results about the anti-concentration of Dirichlet distribution, which may be of independent interest.
[Abstract](https://arxiv.org/abs/1705.07041), [PDF](https://arxiv.org/pdf/1705.07041)


### #115: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results

### #116: Matching neural paths: transfer from recognition to correspondence search
_Nikolay Savinov,  Lubor Ladicky,  Marc Pollefeys_

Many machine learning tasks require finding per-part correspondences between objects. In this work we focus on low-level correspondences - a highly ambiguous matching problem. We propose to use a hierarchical semantic representation of the objects, coming from a convolutional neural network, to solve this ambiguity. Training it for low-level correspondence prediction directly might not be an option in some domains where the ground-truth correspondences are hard to obtain. We show how transfer from recognition can be used to avoid such training. Our idea is to mark parts as "matching" if their features are close to each other at all the levels of convolutional feature hierarchy (neural paths). Although the overall number of such paths is exponential in the number of layers, we propose a polynomial algorithm for aggregating all of them in a single backward pass. The empirical validation is done on the task of stereo correspondence and demonstrates that we achieve competitive results among the methods which do not use labeled target domain data.
[Abstract](https://arxiv.org/abs/1705.08272), [PDF](https://arxiv.org/pdf/1705.08272)


### #117: Linearly constrained Gaussian processes
_Carl Jidling,  Niklas Wahlström,  Adrian Wills,  Thomas B. Schön_

We consider a modification of the covariance function in Gaussian processes to correctly account for known linear constraints. By modelling the target function as a transformation of an underlying function, the constraints are explicitly incorporated in the model such that they are guaranteed to be fulfilled by any sample drawn or prediction made. We also propose a constructive procedure for designing the transformation operator and illustrate the result on both simulated and real-data examples.
[Abstract](https://arxiv.org/abs/1703.00787), [PDF](https://arxiv.org/pdf/1703.00787)


### #118: Fixed-Rank Approximation of a Positive-Semidefinite Matrix from Streaming Data
_Joel A. Tropp,  Alp Yurtsever,  Madeleine Udell,  Volkan Cevher_

Several important applications, such as streaming PCA and semidefinite programming, involve a large-scale positive-semidefinite (psd) matrix that is presented as a sequence of linear updates. Because of storage limitations, it may only be possible to retain a sketch of the psd matrix. This paper develops a new algorithm for fixed-rank psd approximation from a sketch. The approach combines the Nystrom approximation with a novel mechanism for rank truncation. Theoretical analysis establishes that the proposed method can achieve any prescribed relative error in the Schatten 1-norm and that it exploits the spectral decay of the input matrix. Computer experiments show that the proposed method dominates alternative techniques for fixed-rank psd matrix approximation across a wide range of examples.
[Abstract](https://arxiv.org/abs/1706.05736), [PDF](https://arxiv.org/pdf/1706.05736)


### #119: Multi-Modal Imitation Learning from Unstructured Demonstrations using Generative Adversarial Nets
_Karol Hausman,  Yevgen Chebotar,  Stefan Schaal,  Gaurav Sukhatme,  Joseph Lim_

Imitation learning has traditionally been applied to learn a single task from demonstrations thereof. The requirement of structured and isolated demonstrations limits the scalability of imitation learning approaches as they are difficult to apply to real-world scenarios, where robots have to be able to execute a multitude of tasks. In this paper, we propose a multi-modal imitation learning framework that is able to segment and imitate skills from unlabelled and unstructured demonstrations by learning skill segmentation and imitation learning jointly. The extensive simulation results indicate that our method can efficiently separate the demonstrations into individual skills and learn to imitate them using a single multi-modal policy. The video of our experiments is available at this http URL
[Abstract](https://arxiv.org/abs/1705.10479), [PDF](https://arxiv.org/pdf/1705.10479)


### #120: Learning to Inpaint for Image Compression

### #121: Adaptive Bayesian Sampling with Monte Carlo EM

### #122: No More Fixed Penalty  Parameter in ADMM: Faster Convergence with New Adaptive Penalization

### #123: Shape and Material from Sound

### #124: Flexible statistical inference for mechanistic models of neural dynamics

### #125: Online Prediction with Selfish Experts
_Tim Roughgarden,  Okke Schrijvers_

We consider the problem of binary prediction with expert advice in settings where experts have agency and seek to maximize their credibility. This paper makes three main contributions. First, it defines a model to reason formally about settings with selfish experts, and demonstrates that "incentive compatible" (IC) algorithms are closely related to the design of proper scoring rules. Designing a good IC algorithm is easy if the designer's loss function is quadratic, but for other loss functions, novel techniques are required. Second, we design IC algorithms with good performance guarantees for the absolute loss function. Third, we give a formal separation between the power of online prediction with selfish experts and online prediction with honest experts by proving lower bounds for both IC and non-IC algorithms. In particular, with selfish experts and the absolute loss function, there is no (randomized) algorithm for online prediction-IC or otherwise-with asymptotically vanishing regret.
[Abstract](https://arxiv.org/abs/1702.03615), [PDF](https://arxiv.org/pdf/1702.03615)


### #126: Tensor Biclustering

### #127: DPSCREEN: Dynamic Personalized Screening

### #128: Learning Unknown Markov Decision Processes: A Thompson Sampling Approach
_Yi Ouyang,  Mukul Gagrani,  Ashutosh Nayyar,  Rahul Jain_

We consider the problem of learning an unknown Markov Decision Process (MDP) that is weakly communicating in the infinite horizon setting. We propose a Thompson Sampling-based reinforcement learning algorithm with dynamic episodes (TSDE). At the beginning of each episode, the algorithm generates a sample from the posterior distribution over the unknown model parameters. It then follows the optimal stationary policy for the sampled model for the rest of the episode. The duration of each episode is dynamically determined by two stopping criteria. The first stopping criterion controls the growth rate of episode length. The second stopping criterion happens when the number of visits to any state-action pair is doubled. We establish $\tilde O(HS\sqrt{AT})$ bounds on expected regret under a Bayesian setting, where $S$ and $A$ are the sizes of the state and action spaces, $T$ is time, and $H$ is the bound of the span. This regret bound matches the best available bound for weakly communicating MDPs. Numerical results show it to perform better than existing algorithms for infinite horizon MDPs.
[Abstract](https://arxiv.org/abs/1709.04570), [PDF](https://arxiv.org/pdf/1709.04570)


### #129: Testing and Learning on Distributions with Symmetric Noise Invariance
_Ho Chung Leon Law,  Christopher Yau,  Dino Sejdinovic_

Kernel embeddings of distributions and the Maximum Mean Discrepancy (MMD), the resulting distance between distributions, are useful tools for fully nonparametric two-sample testing and learning on distributions. However, it is rarely that all possible differences between samples are of interest -- discovered differences can be due to different types of measurement noise, data collection artefacts or other irrelevant sources of variability. We propose distances between distributions which encode invariance to additive symmetric noise, aimed at testing whether the assumed true underlying processes differ. Moreover, we construct invariant features of distributions, leading to learning algorithms robust to the impairment of the input distributions with symmetric additive noise. Such features lend themselves to a straightforward neural network implementation and can thus also be learned given a supervised signal.
[Abstract](https://arxiv.org/abs/1703.07596), [PDF](https://arxiv.org/pdf/1703.07596)


### #130: A Dirichlet Mixture Model of Hawkes Processes for Event Sequence Clustering
_Hongteng Xu,  Hongyuan Zha_

We propose an effective method to solve the event sequence clustering problems based on a novel Dirichlet mixture model of a special but significant type of point processes --- Hawkes process. In this model, each event sequence belonging to a cluster is generated via the same Hawkes process with specific parameters, and different clusters correspond to different Hawkes processes. The prior distribution of the Hawkes processes is controlled via a Dirichlet distribution. We learn the model via a maximum likelihood estimator (MLE) and propose an effective variational Bayesian inference algorithm. We specifically analyze the resulting EM-type algorithm in the context of inner-outer iterations and discuss several inner iteration allocation strategies. The identifiability of our model, the convergence of our learning method, and its sample complexity are analyzed in both theoretical and empirical ways, which demonstrate the superiority of our method to other competitors. The proposed method learns the number of clusters automatically and is robust to model misspecification. Experiments on both synthetic and real-world data show that our method can learn diverse triggering patterns hidden in asynchronous event sequences and achieve encouraging performance on clustering purity and consistency.
[Abstract](https://arxiv.org/abs/1701.09177), [PDF](https://arxiv.org/pdf/1701.09177)


### #131: Deanonymization in the Bitcoin P2P Network

### #132: Accelerated consensus via Min-Sum Splitting

### #133: Generalized Linear Model Regression under Distance-to-set Penalties

### #134: Adaptive sampling for a population of neurons

### #135: Nonbacktracking Bounds on the Influence in Independent Cascade Models
_Emmanuel Abbe,  Sanjeev Kulkarni,  Eun Jee Lee_

This paper develops upper and lower bounds on the influence measure in a network, more precisely, the expected number of nodes that a seed set can influence in the independent cascade model. In particular, our bounds exploit nonbacktracking walks, Fortuin-Kasteleyn-Ginibre (FKG) type inequalities, and are computed by message passing implementation. Nonbacktracking walks have recently allowed for headways in community detection, and this paper shows that their use can also impact the influence computation. Further, we provide a knob to control the trade-off between the efficiency and the accuracy of the bounds. Finally, the tightness of the bounds is illustrated with simulations on various network models.
[Abstract](https://arxiv.org/abs/1706.05295), [PDF](https://arxiv.org/pdf/1706.05295)


### #136: Learning with Feature Evolvable Streams
_Bo-Jian Hou,  Lijun Zhang,  Zhi-Hua Zhou_

Learning with streaming data has attracted much attention during the past few years. Though most studies consider data stream with fixed features, in real practice the features may be evolvable. For example, features of data gathered by limited-lifespan sensors will change when these sensors are substituted by new ones. In this paper, we propose a novel learning paradigm: Feature Evolvable Streaming Learning where old features would vanish and new features will occur. Rather than relying on only the current features, we attempt to recover the vanished features and exploit it to improve performance. Specifically, we learn two models from the recovered features and the current features, respectively. To benefit from the recovered features, we develop two ensemble methods. In the first method, we combine the predictions from two models and theoretically show that with assistance of old features, the performance on new features can be improved. In the second approach, we dynamically select the best single prediction and establish a better performance guarantee when the best model switches. Experiments on both synthetic and real data validate the effectiveness of our proposal.
[Abstract](https://arxiv.org/abs/1706.05259), [PDF](https://arxiv.org/pdf/1706.05259)


### #137: Online Convex Optimization with Stochastic Constraints
_Hao Yu,  Michael J. Neely,  Xiaohan Wei_

This paper considers online convex optimization (OCO) with stochastic constraints, which generalizes Zinkevich's OCO over a known simple fixed set by introducing multiple stochastic functional constraints that are i.i.d. generated at each round and are disclosed to the decision maker only after the decision is made. This formulation arises naturally when decisions are restricted by stochastic environments or deterministic environments with noisy observations. It also includes many important problems as special cases, such as OCO with long term constraints, stochastic constrained convex optimization, and deterministic constrained convex optimization. To solve this problem, this paper proposes a new algorithm that achieves $O(\sqrt{T})$ expected regret and constraint violations and $O(\sqrt{T}\log(T))$ high probability regret and constraint violations. Experiments on a real-world data center scheduling problem further verify the performance of the new algorithm.
[Abstract](https://arxiv.org/abs/1708.03741), [PDF](https://arxiv.org/pdf/1708.03741)


### #138: Max-Margin Invariant Features from Transformed Unlabelled Data

### #139: Cognitive Impairment Prediction in Alzheimer’s Disease with Regularized Modal Regression

### #140: Translation Synchronization via Truncated Least Squares

### #141: From which world is your graph

### #142: A New Alternating Direction Method for Linear Programming

### #143: Regret Analysis for Continuous Dueling Bandit

### #144: Best Response Regression

### #145: TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning

### #146: Learning Affinity via Spatial Propagation Networks

### #147: Linear regression without correspondence

### #148: NeuralFDR: Learning Discovery Thresholds from Hypothesis Features

### #149: Cost efficient gradient boosting

### #150: Probabilistic Rule Realization and Selection

### #151: Nearest-Neighbor Sample Compression: Efficiency, Consistency, Infinite Dimensions

### #152: A Scale Free Algorithm for Stochastic Bandits with Bounded Kurtosis

### #153: Learning Multiple Tasks with Deep Relationship Networks

### #154: Deep Hyperalignment

### #155: Online to Offline Conversions and Adaptive Minibatch Sizes

### #156: Stochastic Optimization with Variance Reduction for Infinite Datasets with Finite Sum Structure

### #157: Deep Learning with Topological Signatures

### #158: Predicting User Activity Level In Point Process Models With Mass Transport Equation

### #159: Submultiplicative Glivenko-Cantelli and Uniform Convergence of Revenues

### #160: Deep Dynamic Poisson Factorization Model

### #161: Positive-Unlabeled Learning with Non-Negative Risk Estimator

### #162: Optimal Sample Complexity of M-wise Data for Top-K Ranking

### #163: What-If Reasoning using Counterfactual Gaussian Processes

### #164: Communication-Efficient Stochastic Gradient Descent, with Applications to Neural Networks

### #165: On the Convergence of Block Coordinate Descent in Training DNNs with Tikhonov Regularization

### #166: Train longer, generalize better: closing the generalization gap in large batch training of neural networks

### #167: Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks

### #168: Model evidence from nonequilibrium simulations

### #169: Minimal Exploration in Structured Stochastic Bandits

### #170: Learned D-AMP: Principled Neural-network-based Compressive Image Recovery

### #171: Deliberation Networks: Sequence Generation Beyond One-Pass Decoding

### #172: Adaptive Clustering through Semidefinite Programming

### #173: Log-normality and Skewness of Estimated State/Action Values in Reinforcement Learning

### #174: Repeated Inverse Reinforcement Learning
_Kareem Amin,  Nan Jiang,  Satinder Singh_

How detailed should we make the goals we prescribe to AI agents acting on our behalf in complex environments? Detailed and low-level specification of goals can be tedious and expensive to create, and abstract and high-level goals could lead to negative surprises as the agent may find behaviors that we would not want it to do, i.e., lead to unsafe AI. One approach to addressing this dilemma is for the agent to infer human goals by observing human behavior. This is the Inverse Reinforcement Learning (IRL) problem. However, IRL is generally ill-posed for there are typically many reward functions for which the observed behavior is optimal. While the use of heuristics to select from among the set of feasible reward functions has led to successful applications of IRL to learning from demonstration, such heuristics do not address AI safety. In this paper we introduce a novel repeated IRL problem that captures an aspect of AI safety as follows. The agent has to act on behalf of a human in a sequence of tasks and wishes to minimize the number of tasks that it surprises the human. Each time the human is surprised the agent is provided a demonstration of the desired behavior by the human. We formalize this problem, including how the sequence of tasks is chosen, in a few different ways and provide some foundational results.
[Abstract](https://arxiv.org/abs/1705.05427), [PDF](https://arxiv.org/pdf/1705.05427)


### #175: The Numerics of GANs

### #176: Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search
_Luigi Acerbi,  Wei Ji Ma_

Computational models in fields such as computational neuroscience are often evaluated via stochastic simulation or numerical approximation. Fitting these models implies a difficult optimization problem over complex, possibly noisy parameter landscapes. Bayesian optimization (BO) has been successfully applied to solving expensive black-box problems in engineering and machine learning. Here we explore whether BO can be applied as a general tool for model fitting. First, we present a novel BO algorithm, Bayesian adaptive direct search (BADS), that achieves competitive performance with an affordable computational overhead for the running time of typical models. We then perform an extensive benchmark of BADS vs. many common and state-of-the-art nonconvex, derivative-free optimizers on a set of model-fitting problems with real data and models from six studies in behavioral, cognitive, and computational neuroscience. With default settings, BADS consistently finds comparable or better solutions than other methods, showing great promise for BO, and BADS in particular, as a general model-fitting tool.
[Abstract](https://arxiv.org/abs/1705.04405), [PDF](https://arxiv.org/pdf/1705.04405)


### #177: Learning Chordal Markov Networks via Branch and Bound

### #178: Revenue Optimization with Approximate Bid Predictions
_Andrés Muñoz Medina,  Sergei Vassilvitskii_

In the context of advertising auctions, finding good reserve prices is a notoriously challenging learning problem. This is due to the heterogeneity of ad opportunity types and the non-convexity of the objective function. In this work, we show how to reduce reserve price optimization to the standard setting of prediction under squared loss, a well understood problem in the learning community. We further bound the gap between the expected bid and revenue in terms of the average loss of the predictor. This is the first result that formally relates the revenue gained to the quality of a standard machine learned model.
[Abstract](https://arxiv.org/abs/1706.04732), [PDF](https://arxiv.org/pdf/1706.04732)


### #179: Solving (Almost) all Systems of Random Quadratic Equations

### #180: Unsupervised Learning of Disentangled Latent Representations from Sequential Data

### #181: Lookahead  Bayesian Optimization with Inequality Constraints

### #182: Hierarchical Methods of Moments

### #183: Interpretable and Globally Optimal Prediction for Textual Grounding using Image Concepts

### #184: Revisit Fuzzy Neural Network: Demystifying Batch Normalization and ReLU with Generalized Hamming Network

### #185: Speeding Up Latent Variable Gaussian Graphical Model Estimation via Nonconvex Optimization
_Pan Xu,  Jian Ma,  Quanquan Gu_

We study the estimation of the latent variable Gaussian graphical model (LVGGM), where the precision matrix is the superposition of a sparse matrix and a low-rank matrix. In order to speed up the estimation of the sparse plus low-rank components, we propose a sparsity constrained maximum likelihood estimator based on matrix factorization, and an efficient alternating gradient descent algorithm with hard thresholding to solve it. Our algorithm is orders of magnitude faster than the convex relaxation based methods for LVGGM. In addition, we prove that our algorithm is guaranteed to linearly converge to the unknown sparse and low-rank components up to the optimal statistical precision. Experiments on both synthetic and genomic data demonstrate the superiority of our algorithm over the state-of-the-art algorithms and corroborate our theory.
[Abstract](https://arxiv.org/abs/1702.08651), [PDF](https://arxiv.org/pdf/1702.08651)


### #186: Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models
_Sergey Ioffe_

Batch Normalization is quite effective at accelerating and improving the training of deep models. However, its effectiveness diminishes when the training minibatches are small, or do not consist of independent samples. We hypothesize that this is due to the dependence of model layer inputs on all the examples in the minibatch, and different activations being produced between training and inference. We propose Batch Renormalization, a simple and effective extension to ensure that the training and inference models generate the same outputs that depend on individual examples rather than the entire minibatch. Models trained with Batch Renormalization perform substantially better than batchnorm when training with small or non-i.i.d. minibatches. At the same time, Batch Renormalization retains the benefits of batchnorm such as insensitivity to initialization and training efficiency.
[Abstract](https://arxiv.org/abs/1702.03275), [PDF](https://arxiv.org/pdf/1702.03275)


### #187: Generating steganographic images via adversarial training

### #188: Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration

### #189: PixelGAN Autoencoders
_Alireza Makhzani,  Brendan Frey_

In this paper, we describe the "PixelGAN autoencoder", a generative autoencoder in which the generative path is a convolutional autoregressive neural network on pixels (PixelCNN) that is conditioned on a latent code, and the recognition path uses a generative adversarial network (GAN) to impose a prior distribution on the latent code. We show that different priors result in different decompositions of information between the latent code and the autoregressive decoder. For example, by imposing a Gaussian distribution as the prior, we can achieve a global vs. local decomposition, or by imposing a categorical distribution as the prior, we can disentangle the style and content information of images in an unsupervised fashion. We further show how the PixelGAN autoencoder with a categorical prior can be directly used in semi-supervised settings and achieve competitive semi-supervised classification results on the MNIST, SVHN and NORB datasets.
[Abstract](https://arxiv.org/abs/1706.00531), [PDF](https://arxiv.org/pdf/1706.00531)


### #190: Consistent Multitask Learning with Nonlinear Output Relations
_Carlo Ciliberto,  Alessandro Rudi,  Lorenzo Rosasco,  Massimiliano Pontil_

Key to multitask learning is exploiting relationships between different tasks to improve prediction performance. If the relations are linear, regularization approaches can be used successfully. However, in practice assuming the tasks to be linearly related might be restrictive, and allowing for nonlinear structures is a challenge. In this paper, we tackle this issue by casting the problem within the framework of structured prediction. Our main contribution is a novel algorithm for learning multiple tasks which are related by a system of nonlinear equations that their joint outputs need to satisfy. We show that the algorithm is consistent and can be efficiently implemented. Experimental results show the potential of the proposed method.
[Abstract](https://arxiv.org/abs/1705.08118), [PDF](https://arxiv.org/pdf/1705.08118)


### #191: Fast Alternating Minimization Algorithms for Dictionary Learning

### #192: Learning ReLUs via Gradient Descent
_Mahdi Soltanolkotabi_

In this paper we study the problem of learning Rectified Linear Units (ReLUs) which are functions of the form $max(0,<w,x>)$ with $w$ denoting the weight vector. We study this problem in the high-dimensional regime where the number of observations are fewer than the dimension of the weight vector. We assume that the weight vector belongs to some closed set (convex or nonconvex) which captures known side-information about its structure. We focus on the realizable model where the inputs are chosen i.i.d.~from a Gaussian distribution and the labels are generated according to a planted weight vector. We show that projected gradient descent, when initialization at 0, converges at a linear rate to the planted model with a number of samples that is optimal up to numerical constants. Our results on the dynamics of convergence of these very shallow neural nets may provide some insights towards understanding the dynamics of deeper architectures.
[Abstract](https://arxiv.org/abs/1705.04591), [PDF](https://arxiv.org/pdf/1705.04591)


### #193: Stabilizing Training of Generative Adversarial Networks through Regularization
_Kevin Roth,  Aurelien Lucchi,  Sebastian Nowozin,  Thomas Hofmann_

Deep generative models based on Generative Adversarial Networks (GANs) have demonstrated impressive sample quality but in order to work they require a careful choice of architecture, parameter initialization, and selection of hyper-parameters. This fragility is in part due to a dimensional mismatch between the model distribution and the true distribution, causing their density ratio and the associated f-divergence to be undefined. We overcome this fundamental limitation and propose a new regularization approach with low computational cost that yields a stable GAN training procedure. We demonstrate the effectiveness of this approach on several datasets including common benchmark image generation tasks. Our approach turns GAN models into reliable building blocks for deep learning.
[Abstract](https://arxiv.org/abs/1705.09367), [PDF](https://arxiv.org/pdf/1705.09367)


### #194: Expectation Propagation with Stochastic Kinetic Model in Complex Interaction Systems

### #195: Data-Efficient Reinforcement Learning in Continuous State-Action Gaussian-POMDPs

### #196: Compatible Reward Inverse Reinforcement Learning

### #197: First-Order Adaptive Sample Size Methods to Reduce Complexity of Empirical Risk Minimization

### #198: Hiding Images in Plain Sight: Deep Steganography

### #199: Neural Program Meta-Induction

### #200: Bayesian Dyadic Trees and Histograms for  Regression
_Stephanie van der Pas,  Veronika Rockova_

Many machine learning tools for regression are based on recursive partitioning of the covariate space into smaller regions, where the regression function can be estimated locally. Among these, regression trees and their ensembles have demonstrated impressive empirical performance. In this work, we shed light on the machinery behind Bayesian variants of these methods. In particular, we study Bayesian regression histograms, such as Bayesian dyadic trees, in the simple regression case with just one predictor. We focus on the reconstruction of regression surfaces that are piecewise constant, where the number of jumps is unknown. We show that with suitably designed priors, posterior distributions concentrate around the true step regression function at the minimax rate (up to a log factor). These results do not require the knowledge of the true number of steps, nor the width of the true partitioning cells. Thus, Bayesian dyadic regression trees are fully adaptive and can recover the true piecewise regression function nearly as well as if we knew the exact number and location of jumps. Our results constitute the first step towards understanding why Bayesian trees and their ensembles have worked so well in practice. As an aside, we discuss prior distributions on balanced interval partitions and how they relate to a problem in geometric probability. Namely, we quantify the probability of covering the circumference of a circle with random arcs whose endpoints are confined to a grid, a new variant of the original problem.
[Abstract](https://arxiv.org/abs/1708.00078), [PDF](https://arxiv.org/pdf/1708.00078)


### #201: A graph-theoretic approach to multitasking

### #202: Consistent Robust Regression

### #203:  Natural value approximators: learning when to trust past estimates

### #204: Bandits Dueling on Partially Ordered Sets

### #205: Elementary Symmetric Polynomials for Optimal Experimental Design
_Zelda Mariet,  Suvrit Sra_

We revisit the classical problem of optimal experimental design (OED) under a new mathematical model grounded in a geometric motivation. Specifically, we introduce models based on elementary symmetric polynomials; these polynomials capture "partial volumes" and offer a graded interpolation between the widely used A-optimal design and D-optimal design models, obtaining each of them as special cases. We analyze properties of our models, and derive both greedy and convex-relaxation algorithms for computing the associated designs. Our analysis establishes approximation guarantees on these algorithms, while our empirical results substantiate our claims and demonstrate a curious phenomenon concerning our greedy method. Finally, as a byproduct, we obtain new results on the theory of elementary symmetric polynomials that may be of independent interest.
[Abstract](https://arxiv.org/abs/1705.09677), [PDF](https://arxiv.org/pdf/1705.09677)


### #206: Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols
_Serhii Havrylov,  Ivan Titov_

Learning to communicate through interaction, rather than relying on explicit supervision, is often considered a prerequisite for developing a general AI. We study a setting where two agents engage in playing a referential game and, from scratch, develop a communication protocol necessary to succeed in this game. Unlike previous work, we require that messages they exchange, both at train and test time, are in the form of a language (i.e. sequences of discrete symbols). We compare a reinforcement learning approach and one using a differentiable relaxation (straight-through Gumbel-softmax estimator) and observe that the latter is much faster to converge and it results in more effective protocols. Interestingly, we also observe that the protocol we induce by optimizing the communication success exhibits a degree of compositionality and variability (i.e. the same information can be phrased in different ways), both properties characteristic of natural languages. As the ultimate goal is to ensure that communication is accomplished in natural language, we also perform experiments where we inject prior information about natural language into our model and study properties of the resulting protocol.
[Abstract](https://arxiv.org/abs/1705.11192), [PDF](https://arxiv.org/pdf/1705.11192)


### #207: Backprop without Learning Rates Through Coin Betting
_Francesco Orabona,  Tatiana Tommasi_

Deep learning methods achieve state-of-the-art performance in many application scenarios. Yet, these methods require a significant amount of hyperparameters tuning in order to achieve the best results. In particular, tuning the learning rates in the stochastic optimization process is still one of the main bottlenecks. In this paper, we propose a new stochastic gradient descent procedure for deep networks that does not require any learning rate setting. Contrary to previous methods, we do not adapt the learning rates nor we make use of the assumed curvature of the objective function. Instead, we reduce the optimization process to a game of betting on a coin and propose a learning rate free optimal algorithm for this scenario. Theoretical convergence is proven for convex and quasi-convex functions and empirical evidence shows the advantage of our algorithm over popular stochastic gradient algorithms.
[Abstract](https://arxiv.org/abs/1705.07795), [PDF](https://arxiv.org/pdf/1705.07795)


### #208: Pixels to Graphs by Associative Embedding
_Alejandro Newell,  Jia Deng_

Graphs are a useful abstraction of image content. Not only can graphs represent details about individual objects in a scene but they can capture the interactions between pairs of objects. We present a method for training a convolutional neural network such that it takes in an input image and produces a full graph. This is done end-to-end in a single stage with the use of associative embeddings. The network learns to simultaneously identify all of the elements that make up a graph and piece them together. We benchmark on the Visual Genome dataset, and report a Recall@50 of 9.7% compared to the prior state-of-the-art at 3.4%, a nearly threefold improvement on the challenging task of scene graph generation.
[Abstract](https://arxiv.org/abs/1706.07365), [PDF](https://arxiv.org/pdf/1706.07365)


### #209: Runtime Neural Pruning

### #210: Compressing the Gram Matrix for Learning Neural Networks in Polynomial Time

### #211: MMD GAN: Towards Deeper Understanding of Moment Matching Network
_Chun-Liang Li,  Wei-Cheng Chang,  Yu Cheng,  Yiming Yang,  Barnabás Póczos_

Generative moment matching network (GMMN) is a deep generative model that differs from Generative Adversarial Network (GAN) by replacing the discriminator in GAN with a two-sample test based on kernel maximum mean discrepancy (MMD). Although some theoretical guarantees of MMD have been studied, the empirical performance of GMMN is still not as competitive as that of GAN on challenging and large benchmark datasets. The computational efficiency of GMMN is also less desirable in comparison with GAN, partially due to its requirement for a rather large batch size during the training. In this paper, we propose to improve both the model expressiveness of GMMN and its computational efficiency by introducing adversarial kernel learning techniques, as the replacement of a fixed Gaussian kernel in the original GMMN. The new approach combines the key ideas in both GMMN and GAN, hence we name it MMD-GAN. The new distance measure in MMD-GAN is a meaningful loss that enjoys the advantage of weak topology and can be optimized via gradient descent with relatively small batch sizes. In our evaluation on multiple benchmark datasets, including MNIST, CIFAR- 10, CelebA and LSUN, the performance of MMD-GAN significantly outperforms GMMN, and is competitive with other representative GAN works.
[Abstract](https://arxiv.org/abs/1705.08584), [PDF](https://arxiv.org/pdf/1705.08584)


### #212: The Reversible Residual Network: Backpropagation Without Storing Activations
_Aidan N. Gomez,  Mengye Ren,  Raquel Urtasun,  Roger B. Grosse_

Deep residual networks (ResNets) have significantly pushed forward the state-of-the-art on image classification, increasing in performance as networks grow both deeper and wider. However, memory consumption becomes a bottleneck, as one needs to store the activations in order to calculate gradients using backpropagation. We present the Reversible Residual Network (RevNet), a variant of ResNets where each layer's activations can be reconstructed exactly from the next layer's. Therefore, the activations for most layers need not be stored in memory during backpropagation. We demonstrate the effectiveness of RevNets on CIFAR-10, CIFAR-100, and ImageNet, establishing nearly identical classification accuracy to equally-sized ResNets, even though the activation storage requirements are independent of depth.
[Abstract](https://arxiv.org/abs/1707.04585), [PDF](https://arxiv.org/pdf/1707.04585)


### #213: Fast Rates for Bandit Optimization with Upper-Confidence Frank-Wolfe

### #214: Zap Q-Learning

### #215: Expectation Propagation for t-Exponential Family Using Q-Algebra
_Futoshi Futami,  Issei Sato,  Masashi Sugiyama_

Exponential family distributions are highly useful in machine learning since their calculation can be performed efficiently through natural parameters. The exponential family has recently been extended to the t-exponential family, which contains Student-t distributions as family members and thus allows us to handle noisy data well. However, since the t-exponential family is denied by the deformed exponential, we cannot derive an efficient learning algorithm for the t-exponential family such as expectation propagation (EP). In this paper, we borrow the mathematical tools of q-algebra from statistical physics and show that the pseudo additivity of distributions allows us to perform calculation of t-exponential family distributions through natural parameters. We then develop an expectation propagation (EP) algorithm for the t-exponential family, which provides a deterministic approximation to the posterior or predictive distribution with simple moment matching. We finally apply the proposed EP algorithm to the Bayes point machine and Student-t process classication, and demonstrate their performance numerically.
[Abstract](https://arxiv.org/abs/1705.09046), [PDF](https://arxiv.org/pdf/1705.09046)


### #216: Few-Shot Learning Through an Information Retrieval  Lens
_Eleni Triantafillou,  Richard Zemel,  Raquel Urtasun_

Few-shot learning refers to understanding new concepts from only a few examples. We propose an information retrieval-inspired approach for this problem that is motivated by the increased importance of maximally leveraging all the available information in this low-data regime. We define a training objective that aims to extract as much information as possible from each training batch by effectively optimizing over all relative orderings of the batch points simultaneously. In particular, we view each batch point as a `query' that ranks the remaining ones based on its predicted relevance to them and we define a model within the framework of structured prediction to optimize mean Average Precision over these rankings. Our method produces state-of-the-art results on standard few-shot learning benchmarks.
[Abstract](https://arxiv.org/abs/1707.02610), [PDF](https://arxiv.org/pdf/1707.02610)


### #217: Formal Guarantees on the Robustness of a Classifier against Adversarial Manipulation
_Matthias Hein,  Maksym Andriushchenko_

Recent work has shown that state-of-the-art classifiers are quite brittle, in the sense that a small adversarial change of an originally with high confidence correctly classified input leads to a wrong classification again with high confidence. This raises concerns that such classifiers are vulnerable to attacks and calls into question their usage in safety-critical systems. We show in this paper for the first time formal guarantees on the robustness of a classifier by giving instance-specific lower bounds on the norm of the input manipulation required to change the classifier decision. Based on this analysis we propose the Cross-Lipschitz regularization functional. We show that using this form of regularization in kernel methods resp. neural networks improves the robustness of the classifier without any loss in prediction performance.
[Abstract](https://arxiv.org/abs/1705.08475), [PDF](https://arxiv.org/pdf/1705.08475)


### #218: Associative Embedding: End-to-End Learning for Joint Detection and Grouping
_Alejandro Newell,  Zhiao Huang,  Jia Deng_

We introduce associative embedding, a novel method for supervising convolutional neural networks for the task of detection and grouping. A number of computer vision problems can be framed in this manner including multi-person pose estimation, instance segmentation, and multi-object tracking. Usually the grouping of detections is achieved with multi-stage pipelines, instead we propose an approach that teaches a network to simultaneously output detections and group assignments. This technique can be easily integrated into any state-of-the-art network architecture that produces pixel-wise predictions. We show how to apply this method to both multi-person pose estimation and instance segmentation and report state-of-the-art performance for multi-person pose on the MPII and MS-COCO datasets.
[Abstract](https://arxiv.org/abs/1611.05424), [PDF](https://arxiv.org/pdf/1611.05424)


### #219: Practical Locally Private Heavy Hitters
_Raef Bassily,  Kobbi Nissim,  Uri Stemmer,  Abhradeep Thakurta_

We present new practical local differentially private heavy hitters algorithms achieving optimal or near-optimal worst-case error and running time -- TreeHist and Bitstogram. In both algorithms, server running time is $\tilde O(n)$ and user running time is $\tilde O(1)$, hence improving on the prior state-of-the-art result of Bassily and Smith [STOC 2015] requiring $O(n^{5/2})$ server time and $O(n^{3/2})$ user time. With a typically large number of participants in local algorithms ($n$ in the millions), this reduction in time complexity, in particular at the user side, is crucial for making locally private heavy hitters algorithms usable in practice. We implemented Algorithm TreeHist to verify our theoretical analysis and compared its performance with the performance of Google's RAPPOR code.
[Abstract](https://arxiv.org/abs/1707.04982), [PDF](https://arxiv.org/pdf/1707.04982)


### #220: Large-Scale Quadratically Constrained Quadratic Program via Low-Discrepancy Sequences

### #221: Inhomogoenous Hypergraph Clustering with Applications
_Pan Li,  Olgica Milenkovic_

Hypergraph partitioning is an important problem in machine learning, computer vision and network analytics. A widely used method for hypergraph partitioning relies on minimizing a normalized sum of the costs of partitioning hyperedges across clusters. Algorithmic solutions based on this approach assume that different partitions of a hyperedge incur the same cost. However, this assumption fails to leverage the fact that different subsets of vertices within the same hyperedge may have different structural importance. We hence propose a new hypergraph clustering technique, termed inhomogeneous hypergraph partitioning, which assigns different costs to different hyperedge cuts. We prove that inhomogeneous partitioning produces a quadratic approximation to the optimal solution if the inhomogeneous costs satisfy submodularity constraints. Moreover, we demonstrate that inhomogenous partitioning offers significant performance improvements in applications such as structure learning of rankings, subspace segmentation and motif clustering.
[Abstract](https://arxiv.org/abs/1709.01249), [PDF](https://arxiv.org/pdf/1709.01249)


### #222: Differentiable Learning of Logical Rules for Knowledge Base Reasoning
_Fan Yang,  Zhilin Yang,  William W. Cohen_

We study the problem of learning probabilistic first-order logical rules for knowledge base reasoning. This learning problem is difficult because it requires learning the parameters in a continuous space as well as the structure in a discrete space. We propose a framework, Neural Logic Programming, that combines the parameter and structure learning of first-order logical rules in an end-to-end differentiable model. This approach is inspired by a recently-developed differentiable logic called TensorLog, where inference tasks can be compiled into sequences of differentiable operations. We design a neural controller system that learns to compose these operations. Empirically, our method obtains state-of-the-art results on multiple knowledge base benchmark datasets, including Freebase and WikiMovies.
[Abstract](https://arxiv.org/abs/1702.08367), [PDF](https://arxiv.org/pdf/1702.08367)


### #223: Deep Multi-task Gaussian Processes for Survival Analysis with Competing Risks

### #224: Masked Autoregressive Flow for Density Estimation
_George Papamakarios,  Theo Pavlakou,  Iain Murray_

Autoregressive models are among the best performing neural density estimators. We describe an approach for increasing the flexibility of an autoregressive model, based on modelling the random numbers that the model uses internally when generating data. By constructing a stack of autoregressive models, each modelling the random numbers of the next model in the stack, we obtain a type of normalizing flow suitable for density estimation, which we call Masked Autoregressive Flow. This type of flow is closely related to Inverse Autoregressive Flow and is a generalization of Real NVP. Masked Autoregressive Flow achieves state-of-the-art performance in a range of general-purpose density estimation tasks.
[Abstract](https://arxiv.org/abs/1705.07057), [PDF](https://arxiv.org/pdf/1705.07057)


### #225: Non-convex Finite-Sum Optimization Via SCSG Methods

### #226: Beyond normality: Learning sparse probabilistic graphical models in the non-Gaussian setting

### #227: Inner-loop free ADMM using Auxiliary Deep Neural Networks

### #228: OnACID: Online Analysis of Calcium Imaging Data in Real Time

### #229: Collaborative PAC Learning

### #230: Fast Black-box Variational Inference through Stochastic Trust-Region Optimization

### #231: Scalable Demand-Aware Recommendation

### #232: SGD Learns the Conjugate Kernel Class of the Network

### #233: Noise-Tolerant Interactive Learning Using Pairwise Comparisons

### #234: Analyzing Hidden Representations in End-to-End Automatic Speech Recognition Systems

### #235: Generative Local Metric Learning for Kernel Regression

### #236: Information Theoretic Properties of Markov Random Fields, and their Algorithmic Applications

### #237: Fitting Low-Rank Tensors in Constant Time

### #238: Deep supervised discrete hashing

### #239: Using Options and Covariance Testing for Long Horizon Off-Policy Policy Evaluation

### #240: How regularization affects the critical points in linear networks

### #241: Fisher GAN

### #242: Information-theoretic analysis of generalization capability of learning algorithms

### #243: Sparse Approximate Conic Hulls

### #244: Rigorous Dynamics and Consistent Estimation in Arbitrarily Conditioned Linear Systems

### #245: Toward Goal-Driven Neural Network Models for the Rodent Whisker-Trigeminal System

### #246: Accuracy First: Selecting a Differential Privacy Level for Accuracy Constrained ERM

### #247: EX2: Exploration with Exemplar Models for Deep Reinforcement Learning

### #248: Multitask Spectral Learning of Weighted Automata

### #249: Multi-way Interacting Regression via Factorization Machines

### #250: Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

### #251: Practical Data-Dependent Metric Compression with Provable Guarantees

### #252: REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models

### #253: Nonlinear random matrix theory for deep learning

### #254: Parallel Streaming Wasserstein Barycenters

### #255: ELF: An Extensive, Lightweight and Flexible Research Platform for Real-time Strategy Games

### #256: Dual Discriminator Generative Adversarial Nets

### #257: Dynamic Revenue Sharing

### #258: Decomposition-Invariant Conditional Gradient for General Polytopes with Line Search

### #259: Multi-agent Predictive Modeling with Attentional CommNets

### #260: An Empirical Bayes Approach to Optimizing Machine Learning Algorithms

### #261: Differentially Private Empirical Risk Minimization Revisited: Faster and More General

### #262: Variational Inference via $\chi$ Upper Bound Minimization

### #263: On Quadratic Convergence of DC Proximal Newton Algorithm in Nonconvex Sparse Learning
_Xingguo Li,  Lin F. Yang,  Jason Ge,  Jarvis Haupt,  Tong Zhang,  Tuo Zhao_

We propose a DC proximal Newton algorithm for solving nonconvex regularized sparse learning problems in high dimensions. Our proposed algorithm integrates the proximal Newton algorithm with multi-stage convex relaxation based on difference of convex (DC) programming, and enjoys both strong computational and statistical guarantees. Specifically, by leveraging a sophisticated characterization of sparse modeling structures/assumptions (i.e., local restricted strong convexity and Hessian smoothness), we prove that within each stage of convex relaxation, our proposed algorithm achieves (local) quadratic convergence, and eventually obtains a sparse approximate local optimum with optimal statistical properties after only a few convex relaxations. Numerical experiments are provided to support our theory.
[Abstract](https://arxiv.org/abs/1706.06066), [PDF](https://arxiv.org/pdf/1706.06066)


### #264: #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning

### #265: An Empirical Study on The Properties of Random Bases for Kernel Methods

### #266: Bridging the Gap Between Value and Policy Based Reinforcement Learning

### #267: Premise Selection for Theorem Proving by Deep Graph Embedding

### #268: A Bayesian Data Augmentation Approach for Learning Deep Models

### #269: Principles of Riemannian Geometry  in Neural Networks

### #270: Cold-Start Reinforcement Learning with Softmax Policy Gradients

### #271: Online Dynamic Programming

### #272: Alternating Estimation for Structured High-Dimensional Multi-Response Models

### #273: Convolutional Gaussian Processes

### #274: Estimation of the covariance structure of heavy-tailed distributions

### #275: Mean Field Residual Networks: On the Edge of Chaos

### #276: Decomposable Submodular Function Minimization: Discrete and Continuous

### #277: Gauging Variational Inference

### #278: Deep Recurrent Neural Network-Based Identification of Precursor microRNAs

### #279: Robust Estimation of Neural Signals in Calcium Imaging

### #280: State Aware Imitation Learning

### #281: Beyond Parity: Fairness Objectives for Collaborative Filtering

### #282: A PAC-Bayesian Analysis of Randomized Learning with Application to Stochastic Gradient Descent

### #283: Fully Decentralized Policies for Multi-Agent Systems: An Information Theoretic Approach

### #284: Model-Powered Conditional Independence Test

### #285: Deep Voice 2: Multi-Speaker Neural Text-to-Speech

### #286: Variance-based Regularization with Convex Objectives

### #287: Deep Lattice Networks and Partial Monotonic Functions

### #288: Continual Learning with Deep Generative Replay

### #289: AIDE: An algorithm for measuring the accuracy of probabilistic inference algorithms

### #290: Learning Causal Structures Using Regression Invariance

### #291: Online Influence Maximization under Independent Cascade Model with Semi-Bandit Feedback

### #292: Minimax Optimal Players for the Finite-Time 3-Expert Prediction Problem

### #293: Reinforcement Learning under Model Mismatch

### #294: Hierarchical Attentive Recurrent Tracking

### #295: Tomography of the London Underground: a Scalable Model for Origin-Destination Data

### #296: Rotting Bandits

### #297: Unbiased estimates for linear regression via volume sampling

### #298: An Applied Algorithmic Foundation for Hierarchical Clustering

### #299: Adaptive Accelerated Gradient Converging Method under H\"{o}lderian Error Bound Condition

### #300: Stein Variational Gradient Descent as Gradient Flow

### #301: Partial Hard Thresholding: A Towards Unified Analysis of Support Recovery

### #302: Shallow Updates for Deep Reinforcement Learning

### #303: A Highly Efficient Gradient Boosting Decision Tree

### #304: Adversarial Ranking for Language Generation

### #305: Regret Minimization in MDPs with Options without Prior Knowledge

### #306: Net-Trim: Convex Pruning of Deep Neural Networks with Performance Guarantee

### #307: Graph Matching via Multiplicative Update Algorithm

### #308: Dynamic Importance Sampling for Anytime Bounds of the Partition Function

### #309: Is the Bellman residual a bad proxy?

### #310: Generalization Properties of Learning with Random Features

### #311: Differentially private Bayesian learning on distributed data

### #312: Learning to Compose Domain-Specific Transformations for Data Augmentation

### #313: Wasserstein Learning of Deep Generative Point Process Models

### #314: Ensemble Sampling

### #315: Language modeling with recurrent highway hypernetworks

### #316: Searching in the Dark: Practical SVRG Methods under Error Bound Conditions with  Guarantee

### #317: Bayesian Compression for Deep Learning

### #318: Streaming Sparse Gaussian Process Approximations

### #319: VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning

### #320: Sparse k-Means Embedding

### #321: Utile Context Tree Weighting

### #322: A Regularized Framework for Sparse and Structured Neural Attention

### #323: Multi-output Polynomial Networks and Factorization Machines

### #324: Clustering Billions of Reads for DNA Data Storage

### #325: Multi-Objective Non-parametric Sequential Prediction

### #326: A Universal Analysis of Large-Scale Regularized Least Squares Solutions

### #327: Deep Sets

### #328: ExtremeWeather: A large-scale climate dataset for semi-supervised detection, localization, and understanding of extreme weather events

### #329: Process-constrained batch Bayesian optimisation

### #330: Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes

### #331: Spherical convolutions and their application in molecular modelling

### #332: Efficient Optimization for Linear Dynamical Systems with Applications to Clustering and Sparse Coding

### #333: On Optimal Generalizability in Parametric Learning

### #334: Near Optimal Sketching of Low-Rank Tensor Regression

### #335: Tractability in Structured Probability Spaces

### #336: Model-based Bayesian inference of neural activity and connectivity from all-optical interrogation of a neural circuit

### #337: Gaussian process based nonlinear latent structure discovery in multivariate spike train data

### #338: Neural system identification for large populations separating "what" and "where"

### #339: Certified Defenses for Data Poisoning Attacks

### #340: Eigen-Distortions of Hierarchical Representations

### #341: Limitations on Variance-Reduction and Acceleration Schemes for Finite Sums Optimization

### #342: Unsupervised Sequence Classification using Sequential Output Statistics

### #343: Subset Selection under Noise

### #344: Collecting Telemetry Data Privately

### #345: Concrete Dropout

### #346: Adaptive Batch Size for Safe Policy Gradients

### #347: A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning

### #348: PASS-GLM: polynomial approximate sufficient statistics for scalable Bayesian GLM inference

### #349: Bayesian GANs

### #350: Off-policy evaluation for slate recommendation

### #351: A multi-agent reinforcement learning model of common-pool resource appropriation

### #352: On the Optimization Landscape of Tensor Decompositions

### #353: High-Order Attention Models for Visual Question Answering

### #354: Sparse convolutional coding for neuronal assembly detection

### #355: Quantifying how much sensory information in a neural code is relevant for behavior

### #356: Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks

### #357: Reducing Reparameterization Gradient Variance

### #358: Visual Reference Resolution using Attention Memory for Visual Dialog

### #359: Joint distribution optimal transportation for domain adaptation

### #360: Multiresolution Kernel Approximation for Gaussian Process Regression

### #361: Collapsed variational Bayes for Markov jump processes

### #362: Universal consistency and minimax rates for online Mondrian Forest

### #363: Efficiency Guarantees from Data

### #364: Diving into the shallows: a computational perspective on large-scale shallow learning

### #365: End-to-end Differentiable Proving

### #366: Influence Maximization with $\varepsilon$-Almost Submodular Threshold Function

### #367: Inferring The Latent Structure of Human Decision-Making from Raw Visual Inputs

### #368: Variational Laws of Visual Attention for Dynamic Scenes

### #369: Recursive Sampling for the Nystrom Method

### #370: Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning

### #371: Dynamic Routing Between Capsules

### #372: Incorporating Side Information by Adaptive Convolution

### #373: Conic Scan Coverage algorithm for nonparametric topic modeling

### #374: FALKON: An Optimal Large Scale Kernel Method

### #375: Structured Generative Adversarial Networks

### #376: Conservative Contextual Linear Bandits

### #377: Variational Memory Addressing in Generative Models

### #378: On Tensor Train Rank Minimization : Statistical Efficiency and Scalable Algorithm

### #379: Scalable Levy Process Priors for Spectral Kernel Learning

### #380: Deep Hyperspherical Learning

### #381: Learning Deep Structured Multi-Scale Features using Attention-Gated CRFs for Contour Prediction

### #382: On-the-fly Operation Batching in Dynamic Computation Graphs

### #383: Nonlinear Acceleration of Stochastic Algorithms

### #384: Optimized Pre-Processing for Discrimination Prevention

### #385: YASS: Yet Another Spike Sorter

### #386: Independence clustering (without a matrix)

### #387: Fast amortized inference of neural activity from calcium imaging data with variational autoencoders

### #388: Adaptive Active Hypothesis Testing under Limited Information

### #389: Streaming Weak Submodularity: Interpreting Neural Networks on the Fly

### #390: Successor Features for Transfer in Reinforcement Learning

### #391: Counterfactual Fairness

### #392: Prototypical Networks for Few-shot Learning

### #393: Triple Generative Adversarial Nets

### #394: Efficient Sublinear-Regret Algorithms for Online Sparse Linear Regression

### #395: Mapping distinct timescales of functional interactions among brain networks

### #396: Multi-Armed Bandits with Metric Movement Costs

### #397: Learning A Structured Optimal Bipartite Graph for Co-Clustering

### #398: Learning Low-Dimensional Metrics

### #399: The Marginal Value of Adaptive Gradient Methods in Machine Learning

### #400: Aggressive Sampling for Multi-class to Binary Reduction with Applications to Text Classification

### #401: Deconvolutional Paragraph Representation Learning

### #402: Random Permutation Online Isotonic Regression

### #403: A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning

### #404: Inverse Filtering for Hidden Markov Models

### #405: Non-parametric Neural Networks

### #406: Learning Active Learning from Data

### #407: VAE Learning via Stein Variational Gradient Descent

### #408: Deep adversarial neural decoding

### #409: Efficient Use of Limited-Memory Resources to Accelerate Linear Learning

### #410: Temporal Coherency based Criteria for Predicting Video Frames using Deep Multi-stage Generative Adversarial Networks

### #411: Sobolev Training for Neural Networks

### #412: Multi-Information Source Optimization

### #413: Deep Reinforcement Learning from Human Preferences

### #414: On the Fine-Grained Complexity of Empirical Risk Minimization: Kernel Methods and Neural Networks

### #415: Policy Gradient With Value Function Approximation For Collective Multiagent Planning

### #416: Adversarial Symmetric Variational Autoencoder

### #417: Tensor encoding and decomposition of brain connectomes with application to tractography evaluation

### #418: A Minimax Optimal Algorithm for Crowdsourcing

### #419: Estimating Accuracy from Unlabeled Data: A Probabilistic Logic Approach

### #420: A Decomposition of Forecast Error in Prediction Markets

### #421: Safe Adaptive Importance Sampling

### #422: Variational Walkback: Learning a Transition Operator as a Stochastic Recurrent Net

### #423: Polynomial Codes: an Optimal Design for High-Dimensional Coded Matrix Multiplication

### #424: Unsupervised Learning of Disentangled Representations from Video

### #425: Federated Multi-Task Learning

### #426: Is Input Sparsity Time Possible for Kernel Low-Rank Approximation?

### #427: The Expxorcist: Nonparametric Graphical Models Via Conditional Exponential Densities

### #428: Improved Graph Laplacian via Geometric Self-Consistency

### #429: Dual Path Networks

### #430: Faster and Non-ergodic O(1/K) Stochastic Alternating Direction Method of Multipliers

### #431: A Probabilistic Framework for Nonlinearities in Stochastic Neural Networks

### #432: DisTraL: Robust multitask reinforcement learning

### #433: Online Learning of Optimal Bidding Strategy in Repeated Multi-Commodity Auctions

### #434: Trimmed Density Ratio Estimation

### #435: Training recurrent networks to generate hypotheses about how the brain solves hard navigation problems

### #436: Visual Interaction Networks

### #437: Reconstruct & Crush Network

### #438: Streaming Robust Submodular Maximization:A Partitioned Thresholding Approach

### #439: Simple strategies for recovering inner products from coarsely quantized random projections

### #440: Discovering Potential Influence via Information Bottleneck

### #441: Doubly Stochastic Variational Inference for Deep Gaussian Processes

### #442: Ranking Data with Continuous Labels through Oriented Recursive Partitions

### #443: Scalable Model Selection for Belief Networks

### #444: Targeting EEG/LFP Synchrony with Neural Nets

### #445: Near-Optimal Edge Evaluation in Explicit Generalized Binomial Graphs

### #446: Non-Stationary Spectral Kernels

### #447: Overcoming Catastrophic Forgetting by Incremental Moment Matching

### #448: Balancing information exposure in social networks

### #449: SafetyNets: Verifiable Execution of Deep Neural Networks on an Untrusted Cloud

### #450: Query Complexity of Clustering with Side Information

### #451: QMDP-Net: Deep Learning for Planning under Partial Observability

### #452: Robust Optimization for Non-Convex Objectives

### #453: Thy Friend is My Friend: Iterative Collaborative Filtering for Sparse Matrix Estimation

### #454: Adaptive Classification for Prediction Under a Budget

### #455: Convergence rates of a partition based Bayesian multivariate density estimation method

### #456: Affine-Invariant Online Optimization

### #457: Beyond Worst-case: A Probabilistic Analysis of Affine Policies in Dynamic Optimization

### #458: A unified approach to interpreting model predictions

### #459: Stochastic Approximation for Canonical Correlation Analysis

### #460: Investigating the learning dynamics of deep neural networks using random matrix theory

### #461: Sample and Computationally Efficient Learning Algorithms under S-Concave Distributions

### #462: Scalable Variational Inference for Dynamical Systems

### #463: Context Selection for Embedding Models

### #464: Working hard to know your neighbor's margins: Local descriptor learning loss

### #465: Accelerated Stochastic Greedy Coordinate Descent by Soft Thresholding Projection onto Simplex

### #466: Multi-Task Learning for Contextual Bandits

### #467: Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon

### #468: Accelerated First-order Methods for Geodesically Convex Optimization on Riemannian Manifolds

### #469: Selective Classification for Deep Neural Networks

### #470: Minimax Estimation of Bandable Precision Matrices

### #471: Monte-Carlo Tree Search by Best Arm Identification

### #472: Group Additive Structure Identification for Kernel Nonparametric Regression

### #473: Fast, Sample-Efficient Algorithms for Structured Phase Retrieval

### #474: Hash Embeddings for Efficient Word Representations

### #475: Online Learning for Multivariate Hawkes Processes

### #476: Maximum Margin Interval Trees

### #477: DropoutNet: Addressing Cold Start in Recommender Systems

### #478: A simple neural network module for relational reasoning

### #479: Q-LDA: Uncovering Latent Patterns in Text-based Sequential Decision Processes

### #480: Online Reinforcement Learning in Stochastic Games

### #481: Position-based Multiple-play Multi-armed Bandit Problem with Unknown Position Bias

### #482: Active Exploration for Learning Symbolic Representations

### #483: Clone MCMC: Parallel High-Dimensional Gaussian Gibbs Sampling

### #484: Fair Clustering Through Fairlets

### #485: Polynomial time algorithms for dual volume sampling

### #486: Hindsight Experience Replay

### #487: Stochastic and Adversarial Online Learning without Hyperparameters

### #488: Teaching Machines to Describe Images with Natural Language Feedback

### #489: Perturbative Black Box Variational Inference

### #490: GibbsNet: Iterative Adversarial Inference for Deep Graphical Models

### #491: PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

### #492: Regularizing Deep Neural Networks by Noise: Its Interpretation and Optimization

### #493: Learning Graph Embeddings with Embedding Propagation

### #494: Efficient Modeling of Latent Information in Supervised Learning using Gaussian Processes

### #495: A-NICE-MC: Adversarial Training for MCMC

### #496: Excess Risk Bounds for the Bayes Risk using Variational Inference in Latent Gaussian Models

### #497: Real-Time Bidding with Side Information

### #498: Saliency-based Sequential Image Attention with Multiset Prediction

### #499: Variational Inference for Gaussian Process Models with Linear Complexity

### #500: K-Medoids For K-Means Seeding

### #501: Identifying Outlier Arms in Multi-Armed Bandit

### #502: Online Learning with Transductive Regret

### #503: Riemannian approach to batch normalization

### #504: Self-supervised Learning of Motion Capture

### #505: Triangle Generative Adversarial Networks

### #506: Preserving Proximity and Global Ranking for Node Embedding

### #507: Bayesian Optimization with Gradients

### #508: Second-order Optimization in Deep Reinforcement Learning using Kronecker-factored Approximation

### #509: Renyi Differential Privacy Mechanisms for Posterior Sampling

### #510: Online Learning with a Hint

### #511: Identification of Gaussian Process State Space Models

### #512: Robust Imitation of Diverse Behaviors

### #513: Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent

### #514: Local Aggregative Games

### #515: A Sample Complexity Measure with Applications to Learning Optimal Auctions

### #516: Thinking Fast and Slow with Deep Learning and Tree Search

### #517: EEG-GRAPH: A Factor Graph Based Model for Capturing Spatial, Temporal, and Observational Relationships in Electroencephalograms

### #518: Improving the Expected Improvement Algorithm

### #519: Hybrid Reward Architecture for Reinforcement Learning

### #520: Approximate Supermodularity Bounds for Experimental Design

### #521: Maximizing Subset Accuracy with Recurrent Neural Networks in Multi-label Classification

### #522: AdaGAN: Boosting Generative Models

### #523: Straggler Mitigation in Distributed Optimization Through Data Encoding

### #524: Multi-View Decision Processes

### #525: A Greedy Approach for Budgeted Maximum Inner Product Search

### #526: SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks

### #527: Plan, Attend, Generate: Planning for Sequence-to-Sequence Models

### #528: Task-based End-to-end Model Learning in Stochastic Optimization

### #529: Towards Understanding Adversarial Learning for Joint Distribution Matching

### #530: Finite sample analysis of the GTD Policy Evaluation Algorithms in Markov Setting

### #531: On the Complexity of Learning Neural Networks

### #532: Hierarchical Implicit Models and Likelihood-Free Variational Inference

### #533: Improved Semi-supervised Learning with GANs using Manifold Invariances

### #534: Approximation and Convergence Properties of Generative Adversarial Learning

### #535: From Bayesian Sparsity to Gated Recurrent Nets

### #536: Min-Max Propagation

### #537: What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

### #538: Gradient descent GAN optimization is locally stable

### #539: Toward Robustness against Label Noise in Training Deep Discriminative Neural Networks

### #540: Dualing GANs

### #541: Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model

### #542: Do Deep Neural Networks Suffer from Crowding?

### #543: Learning from Complementary Labels

### #544: More powerful and flexible rules for online FDR control with memory and weights

### #545: Learning from uncertain curves: The 2-Wasserstein metric for Gaussian processes

### #546: Discriminative State Space Models

### #547: On Fairness and Calibration

### #548: Imagination-Augmented Agents for Deep Reinforcement Learning

### #549: Extracting low-dimensional dynamics from multiple large-scale neural population recordings by learning to predict correlations

### #550: Unifying PAC and Regret: Uniform PAC Bounds for Episodic Reinforcement Learning

### #551: Gradients of Generative Models for Improved Discriminative Analysis of Tandem Mass Spectra

### #552: Asynchronous Parallel Coordinate Minimization for MAP Inference

### #553: Multiscale Quantization for Fast Similarity Search

### #554: Diverse and Accurate Image Description Using a Variational Auto-Encoder with an Additive Gaussian Encoding Space

### #555: Improved Training of Wasserstein GANs

### #556: Optimally Learning Populations of Parameters

### #557: Clustering with Noisy Queries

### #558: Higher-Order Total Variation Classes on Grids: Minimax Theory and Trend Filtering Methods

### #559: Training Quantized Nets: A Deeper Understanding

### #560: Permutation-based Causal Inference Algorithms with Interventions

### #561: Time-dependent spatially varying graphical models, with application to brain fMRI data analysis

### #562: Gradient Methods for Submodular Maximization

### #563: Smooth Primal-Dual Coordinate Descent Algorithms for Nonsmooth Convex Optimization

### #564: Maximizing the Spread of Influence from Training Data

### #565: Multiplicative Weights Update with Constant Step-Size in Congestion Games:  Convergence, Limit Cycles and Chaos

### #566: Learning Neural Representations of Human Cognition across Many fMRI Studies

### #567: A KL-LUCB algorithm for Large-Scale Crowdsourcing

### #568: Collaborative Deep Learning in Fixed Topology Networks

### #569: Fast-Slow Recurrent Neural Networks

### #570: Learning Disentangled Representations with Semi-Supervised Deep Generative Models

### #571: Learning to Generalize Intrinsic Images with a Structured Disentangling Autoencoder

### #572: Exploring Generalization in Deep Learning

### #573: A framework for Multi-A(rmed)/B(andit) Testing with Online FDR Control

### #574: Fader Networks: Generating Image Variations by Sliding Attribute Values

### #575: Action Centered Contextual Bandits

### #576: Estimating Mutual Information for Discrete-Continuous Mixtures

### #577: Attention is All you Need

### #578: Recurrent Ladder Networks

### #579: Parameter-Free Online Learning via Model Selection

### #580: Bregman Divergence for Stochastic Variance Reduction: Saddle-Point and Adversarial Prediction

### #581: Unbounded cache model for online language modeling with open vocabulary

### #582: Predictive State Recurrent Neural Networks

### #583: Early stopping for kernel boosting algorithms: A general analysis with localized complexities

### #584: SVCCA: Singular Vector Canonical Correlation Analysis for Deep Understanding and Improvement

### #585: Convolutional Phase Retrieval

### #586: Estimating High-dimensional Non-Gaussian Multiple Index Models via Stein’s Lemma

### #587: Gaussian Quadrature for Kernel Features

### #588: Value Prediction Network

### #589: On Learning Errors of Structured Prediction with Approximate Inference

### #590: Efficient Second-Order Online Kernel Learning with Adaptive Embedding

### #591: Implicit Regularization in Matrix Factorization

### #592: Optimal Shrinkage of Singular Values Under Random Data Contamination

### #593: Delayed Mirror Descent in Continuous Games

### #594: Asynchronous Coordinate Descent under More Realistic Assumptions

### #595: Linear Convergence of a Frank-Wolfe Type Algorithm over Trace-Norm Balls

### #596: Hierarchical Clustering Beyond the Worst-Case

### #597: Invariance and Stability of Deep Convolutional Representations

### #598: Statistical Cost Sharing

### #599: The Expressive Power of Neural Networks: A View from the Width

### #600: Spectrally-normalized margin bounds for neural networks

### #601: Robust and Efficient Transfer Learning with Hidden Parameter Markov Decision Processes

### #602: Population Matching Discrepancy and Applications in Deep Learning

### #603: Scalable Planning with Tensorflow for Hybrid Nonlinear Domains

### #604: Boltzmann Exploration Done Right

### #605: Towards the ImageNet-CNN of NLP: Pretraining Sentence Encoders with Machine Translation

### #606: Neural Discrete Representation Learning

### #607: Generalizing GANs: A Turing Perspective

### #608: Scalable Log Determinants for Gaussian Process Kernel Learning

### #609: Poincaré Embeddings for Learning Hierarchical Representations

### #610: Learning Combinatorial Optimization Algorithms over Graphs

### #611: Robust Conditional Probabilities

### #612: Learning with Bandit Feedback in Potential Games

### #613: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments

### #614: Communication-Efficient Distributed Learning of Discrete Distributions

### #615: Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

### #616: When Worlds Collide: Integrating Different Counterfactual Assumptions in Fairness

### #617: Matrix Norm Estimation from a Few Entries

### #618: Deep Networks for Decoding Natural Images from Retinal Signals

### #619: Causal Effect Inference with Deep Latent Variable Models

### #620: Learning Identifiable Gaussian Bayesian Networks in Polynomial Time and Sample Complexity

### #621: Gradient Episodic Memory for Continuum Learning

### #622: Radon Machines: Effective Parallelisation for Machine Learning

### #623: Semisupervised Clustering, AND-Queries and Locally Encodable Source Coding

### #624: Clustering Stable Instances of Euclidean k-means.

### #625: Good Semi-supervised Learning That Requires a Bad GAN

### #626: On Blackbox Backpropagation and Jacobian Sensing

### #627: Protein Interface Prediction using Graph Convolutional Networks

### #628: Solid Harmonic Wavelet Scattering: Predicting Quantum Molecular Energy from Invariant Descriptors of 3D  Electronic Densities

### #629: Towards Generalization and Simplicity in Continuous Control

### #630: Random Projection Filter Bank for Time Series Data

### #631: Filtering Variational Objectives

### #632: On Frank-Wolfe and Equilibrium Computation

### #633: Modulating early visual processing by language

### #634: Learning Mixture of Gaussians with Streaming Data

### #635: Practical Hash Functions for Similarity Estimation and Dimensionality Reduction

### #636: Two Time-Scale Update Rule for Generative Adversarial Nets

### #637: The Scaling Limit of High-Dimensional Online Independent Component Analysis

### #638: Approximation Algorithms for $\ell_0$-Low Rank Approximation

### #639: The power of absolute discounting: all-dimensional distribution estimation

### #640: Supervised Adversarial Domain Adaptation

### #641: Spectral Mixture Kernels for Multi-Output Gaussian Processes

### #642: Neural Expectation Maximization

### #643: Online Learning of Linear Dynamical Systems

### #644: Z-Forcing: Training Stochastic Recurrent Networks

### #645: Thalamus Gated Recurrent Modules

### #646: Neural Variational Inference and Learning in Undirected Graphical Models

### #647: Subspace Clustering via Tangent Cones

### #648: The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process

### #649: Inverse Reward Design

### #650: Structured Bayesian Pruning via Log-Normal Multiplicative Noise

### #651: Attend and Predict: Understanding Gene Regulation by Selective Attention on Chromatin

### #652: Acceleration and Averaging in Stochastic Descent Dynamics

### #653: Kernel functions based on triplet comparisons

### #654: An Error Detection and Correction Framework for Connectomics

### #655: Style Transfer from Non-parallel Text by Cross-Alignment

### #656: Cross-Spectral Factor Analysis

### #657: Stochastic Submodular Maximization: The Case of Coverage Functions

### #658: On Distributed Hierarchical Clustering

### #659: Unsupervised Transformation Learning via Convex Relaxations

### #660: A Sharp Error Analysis for the Fused Lasso, with Implications to Broader Settings  and Approximate Screening

### #661: Efficient Computation of Moments in Sum-Product Networks

### #662: A Meta-Learning Perspective on Cold-Start Recommendations for Items

### #663: Predicting Scene Parsing and Motion Dynamics in the Future

### #664: Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference

### #665: Efficient Approximation Algorithms for Strings Kernel Based Sequence Classification

### #666: Kernel Feature Selection via Conditional Covariance Minimization

### #667:  Statistical Convergence Analysis of Gradient EM on General Gaussian Mixture Models

### #668: Real Time Image Saliency for Black Box Classifiers

### #669: Houdini: Democratizing Adversarial Examples

### #670: Efficient and Flexible Inference for Stochastic Systems

### #671: When Cyclic Coordinate Descent Beats Randomized Coordinate Descent

### #672: Active Learning from Peers

### #673: Learning Causal Graphs with Latent Variables

### #674: Learning to Model the Tail

### #675: Stochastic Mirror Descent for Non-Convex Optimization

### #676: On Separability of Loss Functions, and Revisiting Discriminative Vs Generative Models

### #677: Maxing and Ranking with Few Assumptions

### #678: On clustering network-valued data

### #679: A General Framework for Robust Interactive Learning

### #680: Multi-view Matrix Factorization for Linear Dynamical System Estimation

