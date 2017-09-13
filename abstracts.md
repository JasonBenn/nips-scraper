### #0: Wider and Deeper, Cheaper and Faster: Tensorized LSTMs for Sequence Learning
Not found on arxiv

### #1: Concentration of Multilinear Functions of the Ising Model with Applications to Network Data
Not found on arxiv

### #2: Deep Subspace Clustering Network
We present a novel deep neural network architecture for unsupervised subspace clustering. This architecture is built upon deep auto-encoders, which non-linearly map the input data into a latent space. Our key idea is to introduce a novel self-expressive layer between the encoder and the decoder to mimic the "self-expressiveness" property that has proven effective in traditional subspace clustering. Being differentiable, our new self-expressive layer provides a simple but effective way to learn pairwise affinities between all data points through a standard back-propagation procedure. Being nonlinear, our neural-network based method is able to cluster data points having complex (often nonlinear) structures. We further propose pre-training and fine-tuning strategies that let us effectively learn the parameters of our subspace clustering networks. Our experiments show that the proposed method significantly outperforms the state-of-the-art unsupervised subspace clustering methods.

### #3: Attentional Pooling for Action Recognition
Not found on arxiv

### #4: On the Consistency of Quick Shift
Not found on arxiv

### #5: Rethinking Feature Discrimination and Polymerization for Large-scale Recognition
Not found on arxiv

### #6: Breaking the Nonsmooth Barrier: A Scalable Parallel Method for Composite Optimization
Due to their simplicity and excellent performance, parallel asynchronous variants of stochastic gradient descent have become popular methods to solve a wide range of large-scale optimization problems on multi-core architectures. Yet, despite their practical success, support for nonsmooth objectives is still lacking, making them unsuitable for many problems of interest in machine learning, such as the Lasso, group Lasso or empirical risk minimization with convex constraints. In this work, we propose and analyze ProxASAGA, a fully asynchronous sparse method inspired by SAGA, a variance reduced incremental gradient algorithm. The proposed method is easy to implement and significantly outperforms the state of the art on several nonsmooth, large-scale problems. We prove that our method achieves a theoretical linear speedup with respect to the sequential version under assumptions on the sparsity of gradients and block-separability of the proximal term. Empirical benchmarks on a multi-core architecture illustrate practical speedups of up to 12x on a 20-core machine.

### #7: Dual-Agent GANs for Photorealistic and Identity Preserving Profile Face Synthesis
Not found on arxiv

### #8: Dilated Recurrent Neural Networks
Not found on arxiv

### #9: Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs
Not found on arxiv

### #10: Scalable Generalized Linear Bandits: Online Computation and Hashing
Generalized Linear Bandits (GLBs), a natural extension of the stochastic linear bandits, has been popular and successful in recent years. However, existing GLBs scale poorly with the number of rounds and the number of arms, limiting their utility in practice. This paper proposes new, scalable solutions to the GLB problem in two respects. First, unlike existing GLBs, whose per-time-step space and time complexity grow at least linearly with time $t$, we propose a new algorithm that performs online computations to enjoy a constant space and time complexity. At its heart is a novel Generalized Linear extension of the Online-to-confidence-set Conversion (GLOC method) that takes \emph{any} online learning algorithm and turns it into a GLB algorithm. As a special case, we apply GLOC to the online Newton step algorithm, which results in a low-regret GLB algorithm with much lower time and memory complexity than prior work. Second, for the case where the number $N$ of arms is very large, we propose new algorithms in which each next arm is selected via an inner product search. Such methods can be implemented via hashing algorithms (i.e., "hash-amenable") and result in a time complexity sublinear in $N$. While a Thompson sampling extension of GLOC is hash-amenable, its regret bound for $d$-dimensional arm sets scales with $d^{3/2}$, whereas GLOC's regret bound is linear in $d$. Towards closing this gap, we propose a new hash-amenable algorithm whose regret bound scales with $d^{5/4}$. Finally, we propose a fast approximate hash-key computation (inner product) that has a better accuracy than the state-of-the-art, which can be of independent interest. We conclude the paper with preliminary experimental results confirming the merits of our methods.

### #11: Probabilistic Models for Integration Error in the Assessment of Functional Cardiac Models
Not found on arxiv

### #12:  Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
Not found on arxiv

### #13: Dynamic Safe Interruptibility for Decentralized Multi-Agent Reinforcement Learning
Not found on arxiv

### #14: Interactive Submodular Bandit
Not found on arxiv

### #15: Scene Physics Acquisition via Visual De-animation
Not found on arxiv

### #16: Label Efficient Learning of Transferable Representations acrosss Domains and Tasks
Not found on arxiv

### #17: Decoding with Value Networks for Neural Machine Translation
Not found on arxiv

### #18: Parametric Simplex Method for Sparse Learning
High dimensional sparse learning has imposed a great computational challenge to large scale data analysis. In this paper, we are interested in a broad class of sparse learning approaches formulated as linear programs parametrized by a {\em regularization factor}, and solve them by the parametric simplex method (PSM). Our parametric simplex method offers significant advantages over other competing methods: (1) PSM naturally obtains the complete solution path for all values of the regularization parameter; (2) PSM provides a high precision dual certificate stopping criterion; (3) PSM yields sparse solutions through very few iterations, and the solution sparsity significantly reduces the computational cost per iteration. Particularly, we demonstrate the superiority of PSM over various sparse learning approaches, including Dantzig selector for sparse linear regression, LAD-Lasso for sparse robust linear regression, CLIME for sparse precision matrix estimation, sparse differential network estimation, and sparse Linear Programming Discriminant (LPD) analysis. We then provide sufficient conditions under which PSM always outputs sparse solutions such that its computational performance can be significantly boosted. Thorough numerical experiments are provided to demonstrate the outstanding performance of the PSM method.

### #19: Group Sparse Additive Machine
Not found on arxiv

### #20: Uprooting and Rerooting Higher-order Graphical Models
Not found on arxiv

### #21: The Unreasonable Effectiveness of Structured Random Orthogonal Embeddings
Not found on arxiv

### #22: From Parity to Preference: Learning with Cost-effective Notions of Fairness
Not found on arxiv

### #23: Inferring Generative Model Structure with Static Analysis
Obtaining enough labeled data to robustly train complex discriminative models is a major bottleneck in the machine learning pipeline. A popular solution is combining multiple sources of weak supervision using generative models. The structure of these models affects training label quality, but is difficult to learn without any ground truth labels. We instead rely on these weak supervision sources having some structure by virtue of being encoded programmatically. We present Coral, a paradigm that infers generative model structure by statically analyzing the code for these heuristics, thus reducing the data required to learn structure significantly. We prove that Coral's sample complexity scales quasilinearly with the number of heuristics and number of relations found, improving over the standard sample complexity, which is exponential in $n$ for identifying $n^{\textrm{th}}$ degree relations. Experimentally, Coral matches or outperforms traditional structure learning approaches by up to 3.81 F1 points. Using Coral to model dependencies instead of assuming independence results in better performance than a fully supervised model by 3.07 accuracy points when heuristics are used to label radiology data without ground truth labels.

### #24: Structured Embedding Models for Grouped Data
Not found on arxiv

### #25: A Linear-Time Kernel Goodness-of-Fit Test
We propose a novel adaptive test of goodness-of-fit, with computational cost linear in the number of samples. We learn the test features that best indicate the differences between observed samples and a reference model, by minimizing the false negative rate. These features are constructed via Stein's method, meaning that it is not necessary to compute the normalising constant of the model. We analyse the asymptotic Bahadur efficiency of the new test, and prove that under a mean-shift alternative, our test always has greater relative efficiency than a previous linear-time kernel test, regardless of the choice of parameters for that test. In experiments, the performance of our method exceeds that of the earlier linear-time test, and matches or exceeds the power of a quadratic-time kernel test. In high dimensions and where model structure may be exploited, our goodness of fit test performs far better than a quadratic-time two-sample test based on the Maximum Mean Discrepancy, with samples drawn from the model.

### #26: Cortical microcircuits as gated-recurrent neural networks
Not found on arxiv

### #27: k-Support and Ordered Weighted Sparsity for Overlapping Groups: Hardness and Algorithms
Not found on arxiv

### #28: A simple model of recognition and recall memory
Not found on arxiv

### #29: On Structured Prediction Theory with Calibrated Convex Surrogate Losses
We provide novel theoretical insights on structured prediction in the context of efficient convex surrogate loss minimization with consistency guarantees. For any task loss, we construct a convex surrogate that can be optimized via stochastic gradient descent and we prove tight bounds on the so-called "calibration function" relating the excess surrogate risk to the actual risk. In contrast to prior related work, we carefully monitor the effect of the exponential number of classes in the learning guarantees as well as on the optimization complexity. As an interesting consequence, we formalize the intuition that some task losses make learning harder than others, and that the classical 0-1 loss is ill-suited for general structured prediction.

### #30: Best of Both Worlds: Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model
We present a novel training framework for neural sequence models, particularly for grounded dialog generation. The standard training paradigm for these models is maximum likelihood estimation (MLE), or minimizing the cross-entropy of the human responses. Across a variety of domains, a recurring problem with MLE trained generative neural dialog models (G) is that they tend to produce 'safe' and generic responses ("I don't know", "I can't tell"). In contrast, discriminative dialog models (D) that are trained to rank a list of candidate human responses outperform their generative counterparts; in terms of automatic metrics, diversity, and informativeness of the responses. However, D is not useful in practice since it can not be deployed to have real conversations with users. Our work aims to achieve the best of both worlds -- the practical usefulness of G and the strong performance of D -- via knowledge transfer from D to G. Our primary contribution is an end-to-end trainable generative visual dialog model, where G receives gradients from D as a perceptual (not adversarial) loss of the sequence sampled from G. We leverage the recently proposed Gumbel-Softmax (GS) approximation to the discrete distribution -- specifically, a RNN augmented with a sequence of GS samplers, coupled with the straight-through gradient estimator to enable end-to-end differentiability. We also introduce a stronger encoder for visual dialog, and employ a self-attention mechanism for answer encoding along with a metric learning loss to aid D in better capturing semantic similarities in answer responses. Overall, our proposed model outperforms state-of-the-art on the VisDial dataset by a significant margin (2.67% on recall@10).

### #31: MaskRNN: Instance Level Video Object Segmentation
Not found on arxiv

### #32: Gated Recurrent Convolution Neural Network for OCR
Not found on arxiv

### #33: Towards Accurate Binary Convolutional Neural Network
Not found on arxiv

### #34: Semi-Supervised Learning for Optical Flow with Generative Adversarial Networks
Not found on arxiv

### #35: Learning a Multi-View Stereo Machine
We present a learnt system for multi-view stereopsis. In contrast to recent learning based methods for 3D reconstruction, we leverage the underlying 3D geometry of the problem through feature projection and unprojection along viewing rays. By formulating these operations in a differentiable manner, we are able to learn the system end-to-end for the task of metric 3D reconstruction. End-to-end learning allows us to jointly reason about shape priors while conforming geometric constraints, enabling reconstruction from much fewer images (even a single image) than required by classical approaches as well as completion of unseen surfaces. We thoroughly evaluate our approach on the ShapeNet dataset and demonstrate the benefits over classical approaches as well as recent learning based methods.

### #36: Phase Transitions in the Pooled Data Problem
Not found on arxiv

### #37: Universal Style Transfer via Feature Transforms
Universal style transfer aims to transfer any arbitrary visual styles to content images. Existing feed-forward based methods, while enjoying the inference efficiency, are mainly limited by inability of generalizing to unseen styles or compromised visual quality. In this paper, we present a simple yet effective method that tackles these limitations without training on any pre-defined styles. The key ingredient of our method is a pair of feature transforms, whitening and coloring, that are embedded to an image reconstruction network. The whitening and coloring transforms reflect a direct matching of feature covariance of the content image to a given style image, which shares similar spirits with the optimization of Gram matrix based cost in neural style transfer. We demonstrate the effectiveness of our algorithm by generating high-quality stylized images with comparisons to a number of recent methods. We also analyze our method by visualizing the whitened features and synthesizing textures via simple feature coloring.

### #38: On the Model Shrinkage Effect of Gamma Process Edge Partition Models
Not found on arxiv

### #39: Pose Guided Person Image Generation
This paper proposes the novel Pose Guided Person Generation Network (PG$^2$) that allows to synthesize person images in arbitrary poses, based on an image of that person and a novel pose. Our generation framework PG$^2$ utilizes the pose information explicitly and consists of two key stages: pose integration and image refinement. In the first stage the condition image and the target pose are fed into a U-Net-like network to generate an initial but coarse image of the person with the target pose. The second stage then refines the initial and blurry result by training a U-Net-like generator in an adversarial way. Extensive experimental results on both 128$\times$64 re-identification images and 256$\times$256 fashion photos show that our model generates high-quality person images with convincing details.

### #40: Inference in Graphical Models via Semidefinite Programming Hierarchies
Not found on arxiv

### #41: Variable Importance Using Decision Trees
Not found on arxiv

### #42: Preventing Gradient Explosions in Gated Recurrent Units
Not found on arxiv

### #43: On the Power of Truncated SVD for General High-rank Matrix Estimation Problems
We show that given an estimate $\widehat{A}$ that is close to a general high-rank positive semi-definite (PSD) matrix $A$ in spectral norm (i.e., $\|\widehat{A}-A\|_2 \leq \delta$), the simple truncated SVD of $\widehat{A}$ produces a multiplicative approximation of $A$ in Frobenius norm. This observation leads to many interesting results on general high-rank matrix estimation problems, which we briefly summarize below ($A$ is an $n\times n$ high-rank PSD matrix and $A_k$ is the best rank-$k$ approximation of $A$): (1) High-rank matrix completion: By observing $\Omega(\frac{n\max\{\epsilon^{-4},k^2\}\mu_0^2\|A\|_F^2\log n}{\sigma_{k+1}(A)^2})$ elements of $A$ where $\sigma_{k+1}\left(A\right)$ is the $\left(k+1\right)$-th singular value of $A$ and $\mu_0$ is the incoherence, the truncated SVD on a zero-filled matrix satisfies $\|\widehat{A}_k-A\|_F \leq (1+O(\epsilon))\|A-A_k\|_F$ with high probability. (2)High-rank matrix de-noising: Let $\widehat{A}=A+E$ where $E$ is a Gaussian random noise matrix with zero mean and $\nu^2/n$ variance on each entry. Then the truncated SVD of $\widehat{A}$ satisfies $\|\widehat{A}_k-A\|_F \leq (1+O(\sqrt{\nu/\sigma_{k+1}(A)}))\|A-A_k\|_F + O(\sqrt{k}\nu)$. (3) Low-rank Estimation of high-dimensional covariance: Given $N$ i.i.d.~samples $X_1,\cdots,X_N\sim\mathcal N_n(0,A)$, can we estimate $A$ with a relative-error Frobenius norm bound? We show that if $N = \Omega\left(n\max\{\epsilon^{-4},k^2\}\gamma_k(A)^2\log N\right)$ for $\gamma_k(A)=\sigma_1(A)/\sigma_{k+1}(A)$, then $\|\widehat{A}_k-A\|_F \leq (1+O(\epsilon))\|A-A_k\|_F$ with high probability, where $\widehat{A}=\frac{1}{N}\sum_{i=1}^N{X_iX_i^\top}$ is the sample covariance.

### #44: f-GANs in an Information Geometric Nutshell
Nowozin \textit{et al} showed last year how to extend the GAN \textit{principle} to all $f$-divergences. The approach is elegant but falls short of a full description of the supervised game, and says little about the key player, the generator: for example, what does the generator actually converge to if solving the GAN game means convergence in some space of parameters? How does that provide hints on the generator's design and compare to the flourishing but almost exclusively experimental literature on the subject? In this paper, we unveil a broad class of distributions for which such convergence happens --- namely, deformed exponential families, a wide superset of exponential families --- and show tight connections with the three other key GAN parameters: loss, game and architecture. In particular, we show that current deep architectures are able to factorize a very large number of such densities using an especially compact design, hence displaying the power of deep architectures and their concinnity in the $f$-GAN game. This result holds given a sufficient condition on \textit{activation functions} --- which turns out to be satisfied by popular choices. The key to our results is a variational generalization of an old theorem that relates the KL divergence between regular exponential families and divergences between their natural parameters. We complete this picture with additional results and experimental insights on how these results may be used to ground further improvements of GAN architectures, via (i) a principled design of the activation functions in the generator and (ii) an explicit integration of proper composite losses' link function in the discriminator.

### #45: Multimodal Image-to-Image Translation by Enforcing Bi-Cycle Consistency
Not found on arxiv

### #46: Mixture-Rank Matrix Approximation for Collaborative Filtering
Not found on arxiv

### #47: Non-monotone Continuous DR-submodular  Maximization: Structure and Algorithms
Not found on arxiv

### #48: Learning with Average Top-k Loss
In this work, we introduce the average top-$k$ (AT$_k$) loss as a new ensemble loss for supervised learning, which is the average over the $k$ largest individual losses over a training dataset. We show that the AT$_k$ loss is a natural generalization of the two widely used ensemble losses, namely the average loss and the maximum loss, but can combines their advantages and mitigate their drawbacks to better adapt to different data distributions. Furthermore, it remains a convex function over all individual losses, which can lead to convex optimization problems that can be solved effectively with conventional gradient-based method. We provide an intuitive interpretation of the AT$_k$ loss based on its equivalent effect on the continuous individual loss functions, suggesting that it can reduce the penalty on correctly classified data. We further give a learning theory analysis of MAT$_k$ learning on the classification calibration of the AT$_k$ loss and the error bounds of AT$_k$-SVM. We demonstrate the applicability of minimum average top-$k$ learning for binary classification and regression using synthetic and real datasets.

### #49: Learning multiple visual domains with residual adapters
There is a growing interest in learning data representations that work well for many different types of problems and data. In this paper, we look in particular at the task of learning a single visual representation that can be successfully utilized in the analysis of very different types of images, from dog breeds to stop signs and digits. Inspired by recent work on learning networks that predict the parameters of another, we develop a tunable deep network architecture that, by means of adapter residual modules, can be steered on the fly to diverse visual domains. Our method achieves a high degree of parameter sharing while maintaining or even improving the accuracy of domain-specific representations. We also introduce the Visual Decathlon Challenge, a benchmark that evaluates the ability of representations to capture simultaneously ten very different visual domains and measures their ability to recognize well uniformly.

### #50: Dykstra's Algorithm, ADMM, and Coordinate Descent: Connections, Insights, and Extensions
We study connections between Dykstra's algorithm for projecting onto an intersection of convex sets, the augmented Lagrangian method of multipliers or ADMM, and block coordinate descent. We prove that coordinate descent for a regularized regression problem, in which the (separable) penalty functions are seminorms, is exactly equivalent to Dykstra's algorithm applied to the dual problem. ADMM on the dual problem is also seen to be equivalent, in the special case of two sets, with one being a linear subspace. These connections, aside from being interesting in their own right, suggest new ways of analyzing and extending coordinate descent. For example, from existing convergence theory on Dykstra's algorithm over polyhedra, we discern that coordinate descent for the lasso problem converges at an (asymptotically) linear rate. We also develop two parallel versions of coordinate descent, based on the Dykstra and ADMM connections.

### #51: Flat2Sphere: Learning Spherical Convolution for Fast Features from 360° Imagery
While 360{\deg} cameras offer tremendous new possibilities in vision, graphics, and augmented reality, the spherical images they produce make core feature extraction non-trivial. Convolutional neural networks (CNNs) trained on images from perspective cameras yield "flat" filters, yet 360{\deg} images cannot be projected to a single plane without significant distortion. A naive solution that repeatedly projects the viewing sphere to all tangent planes is accurate, but much too computationally intensive for real problems. We propose to learn a spherical convolutional network that translates a planar CNN to process 360{\deg} imagery directly in its equirectangular projection. Our approach learns to reproduce the flat filter outputs on 360{\deg} data, sensitive to the varying distortion effects across the viewing sphere. The key benefits are 1) efficient feature extraction for 360{\deg} images and video, and 2) the ability to leverage powerful pre-trained networks researchers have carefully honed (together with massive labeled image training sets) for perspective images. We validate our approach compared to several alternative methods in terms of both raw CNN output accuracy as well as applying a state-of-the-art "flat" object detector to 360{\deg} data. Our method yields the most accurate results while saving orders of magnitude in computation versus the existing exact reprojection solution.

### #52: 3D Shape Reconstruction by Modeling 2.5D Sketch
Not found on arxiv

### #53: Multimodal Learning and Reasoning for Visual Question Answering
Not found on arxiv

### #54: Adversarial Surrogate Losses for Ordinal Regression
Not found on arxiv

### #55: Hypothesis Transfer Learning via Transformation Functions
We consider the Hypothesis Transfer Learning (HTL) problem where one incorporates a hypothesis trained on the source domain into the learning procedure of the target domain. Existing theoretical analysis either only studies specific algorithms or only presents upper bounds on the generalization error but not on the excess risk. In this paper, we propose a unified algorithm-dependent framework for HTL through a novel notion of transformation function, which characterizes the relation between the source and the target domains. We conduct a general risk analysis of this framework and in particular, we show for the first time, if two domains are related, HTL enjoys faster convergence rates of excess risks for Kernel Smoothing and Kernel Ridge Regression than those of the classical non-transfer learning settings. Experiments on real world data demonstrate the effectiveness of our framework.

### #56: Adversarial Invariant Feature Learning
Not found on arxiv

### #57: Convergence Analysis of Two-layer Neural Networks with ReLU Activation
In recent years, stochastic gradient descent (SGD) based techniques has become the standard tools for training neural networks. However, formal theoretical understanding of why SGD can train neural networks in practice is largely missing. In this paper, we make progress on understanding this mystery by providing a convergence analysis for SGD on a rich subset of two-layer feedforward networks with ReLU activations. This subset is characterized by a special structure called "identity mapping". We prove that, if input follows from Gaussian distribution, with standard $O(1/\sqrt{d})$ initialization of the weights, SGD converges to the global minimum in polynomial number of steps. Unlike normal vanilla networks, the "identity mapping" makes our network asymmetric and thus the global minimum is unique. To complement our theory, we are also able to show experimentally that multi-layer networks with this mapping have better performance compared with normal vanilla networks. Our convergence theorem differs from traditional non-convex optimization techniques. We show that SGD converges to optimal in "two phases": In phase I, the gradient points to the wrong direction, however, a potential function $g$ gradually decreases. Then in phase II, SGD enters a nice one point convex region and converges. We also show that the identity mapping is necessary for convergence, as it moves the initial point to a better place for optimization. Experiment verifies our claims.

### #58: Doubly Accelerated Stochastic Variance Reduced Dual Averaging Method for Regularized Empirical Risk Minimization
In this paper, we develop a new accelerated stochastic gradient method for efficiently solving the convex regularized empirical risk minimization problem in mini-batch settings. The use of mini-batches is becoming a golden standard in the machine learning community, because mini-batch settings stabilize the gradient estimate and can easily make good use of parallel computing. The core of our proposed method is the incorporation of our new "double acceleration" technique and variance reduction technique. We theoretically analyze our proposed method and show that our method much improves the mini-batch efficiencies of previous accelerated stochastic methods, and essentially only needs size $\sqrt{n}$ mini-batches for achieving the optimal iteration complexities for both non-strongly and strongly convex objectives, where $n$ is the training set size. Further, we show that even in non-mini-batch settings, our method surpasses the best known convergence rate for non-strongly convex objectives, and it achieves the one for strongly convex objectives.

### #59: Langevin Dynamics with Continuous Tempering for Training Deep Neural Networks
Minimizing non-convex and high-dimensional objective functions is challenging, especially when training modern deep neural networks. In this paper, a novel approach is proposed which divides the training process into two consecutive phases to obtain better generalization performance: Bayesian sampling and stochastic optimization. The first phase is to explore the energy landscape and to capture the "fat" modes; and the second one is to fine-tune the parameter learned from the first phase. In the Bayesian learning phase, we apply continuous tempering and stochastic approximation into the Langevin dynamics to create an efficient and effective sampler, in which the temperature is adjusted automatically according to the designed "temperature dynamics". These strategies can overcome the challenge of early trapping into bad local minima and have achieved remarkable improvements in various types of neural networks as shown in our theoretical analysis and empirical experiments.

### #60: Efficient Online Linear Optimization with Approximation Algorithms
Not found on arxiv

### #61: Geometric Descent Method for Convex Composite Minimization
Not found on arxiv

### #62: Diffusion Approximations for Online Principal Component Estimation and Global Convergence
Not found on arxiv

### #63:  Avoiding Discrimination through Causal Reasoning
Not found on arxiv

### #64: Nonparametric Online Regression while Learning the Metric
We study algorithms for online nonparametric regression that learn the directions along which the regression function is smoother. Our algorithm learns the Mahalanobis metric based on the gradient outer product matrix $\boldsymbol{G}$ of the regression function (automatically adapting to the effective rank of this matrix), while simultaneously bounding the regret ---on the same data sequence--- in terms of the spectrum of $\boldsymbol{G}$. As a preliminary step in our analysis, we generalize a nonparametric online learning algorithm by Hazan and Megiddo by enabling it to compete against functions whose Lipschitzness is measured with respect to an arbitrary Mahalanobis metric.

### #65: Recycling for Fairness: Learning with Conditional Distribution Matching Constraints
Not found on arxiv

### #66: Safe and Nested Subgame Solving for Imperfect-Information Games
Unlike perfect-information games, imperfect-information games cannot be solved by decomposing the game into subgames that are solved independently. Instead, all decisions must consider the strategy of the game as a whole. While it is not possible to solve an imperfect-information game exactly through decomposition, it is possible to approximate solutions, or improve existing strategies, by solving disjoint subgames. This process is referred to as subgame solving. We introduce subgame solving techniques that outperform prior methods both in theory and practice. We also show how to adapt them, and past subgame solving techniques, to respond to opponent actions that are outside the original action abstraction; this significantly outperforms the prior state-of-the-art approach, action translation. Finally, we show that subgame solving can be repeated as the game progresses down the tree, leading to lower exploitability. Subgame solving is a key component of Libratus, the first AI to defeat top humans in heads-up no-limit Texas hold'em poker.

### #67: Unsupervised Image-to-Image Translation Networks
Most of the existing image-to-image translation frameworks---mapping an image in one domain to a corresponding image in another---are based on supervised learning, i.e., pairs of corresponding images in two domains are required for learning the translation function. This largely limits their applications, because capturing corresponding images in two different domains is often a difficult task. To address the issue, we propose the UNsupervised Image-to-image Translation (UNIT) framework, which is based on variational autoencoders and generative adversarial networks. The proposed framework can learn the translation function without any corresponding images in two domains. We enable this learning capability by combining a weight-sharing constraint and an adversarial training objective. Through visualization results from various unsupervised image translation tasks, we verify the effectiveness of the proposed framework. An ablation study further reveals the critical design choices. Moreover, we apply the UNIT framework to the unsupervised domain adaptation task and achieve better results than competing algorithms do in benchmark datasets.

### #68: Coded Distributed Computing for Inverse Problems
Not found on arxiv

### #69: A Screening Rule for l1-Regularized Ising Model Estimation
Not found on arxiv

### #70: Improved Dynamic Regret for Non-degeneracy Functions
Not found on arxiv

### #71: Learning Efficient Object Detection Models with Knowledge Distillation
Not found on arxiv

### #72: One-Sided Unsupervised Domain Mapping
In unsupervised domain mapping, the learner is given two unmatched datasets $A$ and $B$. The goal is to learn a mapping $G_{AB}$ that translates a sample in $A$ to the analog sample in $B$. Recent approaches have shown that when learning simultaneously both $G_{AB}$ and the inverse mapping $G_{BA}$, convincing mappings are obtained. In this work, we present a method of learning $G_{AB}$ without learning $G_{BA}$. This is done by learning a mapping that maintains the distance between a pair of samples. Moreover, good mappings are obtained, even by maintaining the distance between different parts of the same sample before and after mapping. We present experimental results that the new method not only allows for one sided mapping learning, but also leads to preferable numerical results over the existing circularity-based constraint. Our entire code is made publicly available at this https URL .

### #73: Deep Mean-Shift Priors for Image Restoration
Not found on arxiv

### #74: Greedy Algorithms for Cone Constrained Optimization with Convergence Guarantees
Greedy optimization methods such as Matching Pursuit (MP) and Frank-Wolfe (FW) algorithms regained popularity in recent years due to their simplicity, effectiveness and theoretical guarantees. MP and FW address optimization over the linear span and the convex hull of a set of atoms, respectively. In this paper, we consider the intermediate case of optimization over the convex cone, parametrized as the conic hull of a generic atom set, leading to the first principled definitions of non-negative MP algorithms for which we give explicit convergence rates and demonstrate excellent empirical performance. In particular, we derive sublinear ($\mathcal{O}(1/t)$) convergence on general smooth and convex objectives, and linear convergence ($\mathcal{O}(e^{-t})$) on strongly convex objectives, in both cases for general sets of atoms. Furthermore, we establish a clear correspondence of our algorithms to known algorithms from the MP and FW literature. Our novel algorithms and analyses target general atom sets and general objective functions, and hence are directly applicable to a large variety of learning settings.

### #75: A New Theory for Nonconvex Matrix Completion
Not found on arxiv

### #76: Robust Hypothesis Test for Functional Effect with Gaussian Processes
Not found on arxiv

### #77: Lower bounds on the robustness to adversarial perturbations
Not found on arxiv

### #78: Minimizing a Submodular Function from Samples
Not found on arxiv

### #79: Introspective Classification with Convolutional Nets
Not found on arxiv

### #80: Label Distribution Learning Forests
Label distribution learning (LDL) is a general learning framework, which assigns to an instance a distribution over a set of labels rather than a single label or multiple labels. Current LDL methods have either restricted assumptions on the expression form of the label distribution or limitations in representation learning, e.g., to learn deep features in an end-to-end manner. This paper presents label distribution learning forests (LDLFs) - a novel label distribution learning algorithm based on differentiable decision trees, which have several advantages: 1) Decision trees have the potential to model any general form of label distributions by a mixture of leaf node predictions. 2) The learning of differentiable decision trees can be combined with representation learning. We define a distribution-based loss function for a forest, enabling all the trees to be learned jointly, and show that an update function for leaf node predictions, which guarantees a strict decrease of the loss function, can be derived by variational bounding. The effectiveness of the proposed LDLFs is verified on several LDL tasks and a computer vision application, showing significant improvements to the state-of-the-art LDL methods.

### #81: Unsupervised object learning from dense equivariant image labelling
One of the key challenges of visual perception is to extract abstract models of 3D objects and object categories from visual measurements, which are affected by complex nuisance factors such as viewpoint, occlusion, motion, and deformations. Starting from the recent idea of viewpoint factorization, we propose a new approach that, given a large number of images of an object and no other supervision, can extract a dense object-centric coordinate frame. This coordinate frame is invariant to deformations of the images and comes with a dense equivariant labelling neural network that can map image pixels to their corresponding object coordinates. We demonstrate the applicability of this method to simple articulated objects and deformable objects such as human faces, learning embeddings from random synthetic transformations or optical flow correspondences, all without any manual supervision.

### #82: Compression-aware Training of Deep Neural Networks
Not found on arxiv

### #83: Multiscale Semi-Markov Dynamics for Intracortical Brain-Computer Interfaces
Not found on arxiv

### #84: PredRNN: Recurrent Neural Networks for Video Prediction using Spatiotemporal LSTMs
Not found on arxiv

### #85: Detrended Partial Cross Correlation for Brain Connectivity Analysis
Not found on arxiv

### #86: Contrastive Learning for Image Captioning
Not found on arxiv

### #87: Safe Model-based Reinforcement Learning with Stability Guarantees
Reinforcement learning is a powerful paradigm for learning optimal policies from experimental data. However, to find optimal policies, most reinforcement learning algorithms explore all possible actions, which may be harmful for real-world systems. As a consequence, learning algorithms are rarely applied on safety-critical systems in the real world. In this paper, we present a learning algorithm that explicitly considers safety in terms of stability guarantees. Specifically, we extend control theoretic results on Lyapunov stability verification and show how to use statistical models of the dynamics to obtain high-performance control policies with provable stability certificates. Moreover, under additional regularity assumptions in terms of a Gaussian process prior, we prove that one can effectively and safely collect data in order to learn about the dynamics and thus both improve control performance and expand the safe region of the state space. In our experiments, we show how the resulting algorithm can safely optimize a neural network policy on a simulated inverted pendulum, without the pendulum ever falling down.

### #88: Online multiclass boosting
Not found on arxiv

### #89: Matching on Balanced Nonlinear Representations for Treatment Effects Estimation
Not found on arxiv

### #90: Learning Overcomplete HMMs
Not found on arxiv

### #91: GP CaKe: Effective brain connectivity with causal kernels
A fundamental goal in network neuroscience is to understand how activity in one region drives activity elsewhere, a process referred to as effective connectivity. Here we propose to model this causal interaction using integro-differential equations and causal kernels that allow for a rich analysis of effective connectivity. The approach combines the tractability and flexibility of autoregressive modeling with the biophysical interpretability of dynamic causal modeling. The causal kernels are learned nonparametrically using Gaussian process regression, yielding an efficient framework for causal inference. We construct a novel class of causal covariance functions that enforce the desired properties of the causal kernels, an approach which we call GP CaKe. By construction, the model and its hyperparameters have biophysical meaning and are therefore easily interpretable. We demonstrate the efficacy of GP CaKe on a number of simulations and give an example of a realistic application on magnetoencephalography (MEG) data.

### #92: Decoupling "when to update" from "how to update"
Not found on arxiv

### #93: Self-Normalizing Neural Networks
Deep Learning has revolutionized vision via convolutional neural networks (CNNs) and natural language processing via recurrent neural networks (RNNs). However, success stories of Deep Learning with standard feed-forward neural networks (FNNs) are rare. FNNs that perform well are typically shallow and, therefore cannot exploit many levels of abstract representations. We introduce self-normalizing neural networks (SNNs) to enable high-level abstract representations. While batch normalization requires explicit normalization, neuron activations of SNNs automatically converge towards zero mean and unit variance. The activation function of SNNs are "scaled exponential linear units" (SELUs), which induce self-normalizing properties. Using the Banach fixed-point theorem, we prove that activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance -- even under the presence of noise and perturbations. This convergence property of SNNs allows to (1) train deep networks with many layers, (2) employ strong regularization, and (3) to make learning highly robust. Furthermore, for activations not close to unit variance, we prove an upper and lower bound on the variance, thus, vanishing and exploding gradients are impossible. We compared SNNs on (a) 121 tasks from the UCI machine learning repository, on (b) drug discovery benchmarks, and on (c) astronomy tasks with standard FNNs and other machine learning methods such as random forests and support vector machines. SNNs significantly outperformed all competing FNN methods at 121 UCI tasks, outperformed all competing methods at the Tox21 dataset, and set a new record at an astronomy data set. The winning SNN architectures are often very deep. Implementations are available at: github.com/bioinf-jku/SNNs.

### #94: Learning to Pivot with Adversarial Networks
Several techniques for domain adaptation have been proposed to account for differences in the distribution of the data used for training and testing. The majority of this work focuses on a binary domain label. Similar problems occur in a scientific context where there may be a continuous family of plausible data generation processes associated to the presence of systematic uncertainties. Robust inference is possible if it is based on a pivot -- a quantity whose distribution does not depend on the unknown values of the nuisance parameters that parametrize this family of data generation processes. In this work, we introduce and derive theoretical results for a training procedure based on adversarial networks for enforcing the pivotal property (or, equivalently, fairness with respect to continuous attributes) on a predictive model. The method includes a hyperparameter to control the trade-off between accuracy and robustness. We demonstrate the effectiveness of this approach with a toy example and examples from particle physics.

### #95: MolecuLeNet: A continuous-filter convolutional neural network for modeling quantum interactions
Deep learning has the potential to revolutionize quantum chemistry as it is ideally suited to learn representations for structured data and speed up the exploration of chemical space. While convolutional neural networks have proven to be the first choice for images, audio and video data, the atoms in molecules are not restricted to a grid. Instead, their precise locations contain essential physical information, that would get lost if discretized. Thus, we propose to use continuous-filter convolutional layers to be able to model local correlations without requiring the data to lie on a grid. We apply those layers in SchNet: a novel deep learning architecture modeling quantum interactions in molecules. We obtain a joint model for the total energy and interatomic forces that follows fundamental quantum-chemical principles. This includes rotationally invariant energy predictions and a smooth, differentiable potential energy surface. Our architecture achieves state-of-the-art performance for benchmarks of equilibrium molecules and molecular dynamics trajectories. Finally, we introduce a more challenging benchmark with chemical and structural variations that suggests the path for further work.

### #96: Active Bias: Training a More Accurate Neural Network by Emphasizing High Variance Samples
Self-paced learning and hard example mining re-weight training instances to improve learning accuracy. This paper presents two improved alternatives based on lightweight estimates of sample uncertainty in stochastic gradient descent (SGD): the variance in predicted probability of the correct class across iterations of mini-batch SGD, and the proximity of the correct class probability to the decision threshold. Extensive experimental results on six datasets show that our methods reliably improve accuracy in various network architectures, including additional gains on top of other popular training techniques, such as residual learning, momentum, ADAM, batch normalization, dropout, and distillation.

### #97: Differentiable Learning of Submodular Functions
Not found on arxiv

### #98: Inductive Representation Learning on Large Graphs
Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during training of the embeddings; these previous approaches are inherently transductive and do not naturally generalize to unseen nodes. Here we present GraphSAGE, a general, inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, we learn a function that generates embeddings by sampling and aggregating features from a node's local neighborhood. Our algorithm outperforms strong baselines on three inductive node-classification benchmarks: we classify the category of unseen nodes in evolving information graphs based on citation and Reddit post data, and we show that our algorithm generalizes to completely unseen graphs using a multi-graph dataset of protein-protein interactions.

### #99: Subset Selection for Sequential Data
Not found on arxiv

### #100: Question Asking as Program Generation
Not found on arxiv

### #101: Revisiting Perceptron: Efficient and Label-Optimal Learning of Halfspaces
It has been a long-standing problem to efficiently learn a linear separator using as few labels as possible. In this work, we propose an efficient perceptron-based algorithm for actively learning homogeneous linear separators under uniform distribution. Under bounded noise, where each label is flipped with probability at most $\eta$, our algorithm achieves near-optimal $\tilde{O}\left(\frac{d}{(1-2\eta)^2}\log\frac{1}{\epsilon}\right)$ label complexity in time $\tilde{O}\left(\frac{d^2}{\epsilon(1-2\eta)^2}\right)$, and significantly improves over the best known result (Awasthi et al., 2016). Under adversarial noise, where at most $\nu$ fraction of labels can be flipped, our algorithm achieves near-optimal $\tilde{O}\left(d\log\frac{1}{\epsilon}\right)$ label complexity in time $\tilde{O}\left(\frac{d^2}{\epsilon}\right)$, which is better than the best known label complexity and time complexity in Awasthi et al. (2014).

### #102: Gradient Descent Can Take Exponential Time to Escape Saddle Points
Although gradient descent (GD) almost always escapes saddle points asymptotically [Lee et al., 2016], this paper shows that even with fairly natural random initialization schemes and non-pathological functions, GD can be significantly slowed down by saddle points, taking exponential time to escape. On the other hand, gradient descent with perturbations [Ge et al., 2015, Jin et al., 2017] is not slowed down by saddle points - it can find an approximate local minimizer in polynomial time. This result implies that GD is inherently slower than perturbed GD, and justifies the importance of adding perturbations for efficient non-convex optimization. While our focus is theoretical, we also present experiments that illustrate our theoretical findings.

### #103: Union of Intersections (UoI) for Interpretable Data Driven Discovery and Prediction
The increasing size and complexity of scientific data could dramatically enhance discovery and prediction for basic scientific applications. Realizing this potential, however, requires novel statistical analysis methods that are both interpretable and predictive. We introduce Union of Intersections (UoI), a flexible, modular, and scalable framework for enhanced model selection and estimation. Methods based on UoI perform model selection and model estimation through intersection and union operations, respectively. We show that UoI-based methods achieve low-variance and nearly unbiased estimation of a small number of interpretable features, while maintaining high-quality prediction accuracy. We perform extensive numerical investigation to evaluate a UoI algorithm ($UoI_{Lasso}$) on synthetic and real data. In doing so, we demonstrate the extraction of interpretable functional networks from human electrophysiology recordings as well as accurate prediction of phenotypes from genotype-phenotype data with reduced features. We also show (with the $UoI_{L1Logistic}$ and $UoI_{CUR}$ variants of the basic framework) improved prediction parsimony for classification and matrix factorization on several benchmark biomedical data sets. These results suggest that methods based on the UoI framework could improve interpretation and prediction in data-driven discovery across scientific fields.

### #104: One-Shot Imitation Learning
Imitation learning has been commonly applied to solve different tasks in isolation. This usually requires either careful feature engineering, or a significant number of samples. This is far from what we desire: ideally, robots should be able to learn from very few demonstrations of any given task, and instantly generalize to new situations of the same task, without requiring task-specific engineering. In this paper, we propose a meta-learning framework for achieving such capability, which we call one-shot imitation learning. Specifically, we consider the setting where there is a very large set of tasks, and each task has many instantiations. For example, a task could be to stack all blocks on a table into a single tower, another task could be to place all blocks on a table into two-block towers, etc. In each case, different instances of the task would consist of different sets of blocks with different initial states. At training time, our algorithm is presented with pairs of demonstrations for a subset of all tasks. A neural net is trained that takes as input one demonstration and the current state (which initially is the initial state of the other demonstration of the pair), and outputs an action with the goal that the resulting sequence of states and actions matches as closely as possible with the second demonstration. At test time, a demonstration of a single instance of a new task is presented, and the neural net is expected to perform well on new instances of this new task. The use of soft attention allows the model to generalize to conditions and tasks unseen in the training data. We anticipate that by training this model on a much greater variety of tasks and settings, we will obtain a general system that can turn any demonstrations into robust policies that can accomplish an overwhelming variety of tasks. Videos available at this https URL .

### #105: Learning the Morphology of Brain Signals Using Alpha-Stable Convolutional Sparse Coding
Neural time-series data contain a wide variety of prototypical signal waveforms (atoms) that are of significant importance in clinical and cognitive research. One of the goals for analyzing such data is hence to extract such 'shift-invariant' atoms. Even though some success has been reported with existing algorithms, they are limited in applicability due to their heuristic nature. Moreover, they are often vulnerable to artifacts and impulsive noise, which are typically present in raw neural recordings. In this study, we address these issues and propose a novel probabilistic convolutional sparse coding (CSC) model for learning shift-invariant atoms from raw neural signals containing potentially severe artifacts. In the core of our model, which we call $\alpha$CSC, lies a family of heavy-tailed distributions called $\alpha$-stable distributions. We develop a novel, computationally efficient Monte Carlo expectation-maximization algorithm for inference. The maximization step boils down to a weighted CSC problem, for which we develop a computationally efficient optimization algorithm. Our results show that the proposed algorithm achieves state-of-the-art convergence speeds. Besides, $\alpha$CSC is significantly more robust to artifacts when compared to three competing algorithms: it can extract spike bursts, oscillations, and even reveal more subtle phenomena such as cross-frequency coupling when applied to noisy neural time series.

### #106: Integration Methods and Optimization Algorithms
Not found on arxiv

### #107: Sharpness, Restart and Acceleration
The {\L}ojasievicz inequality shows that sharpness bounds on the minimum of convex optimization problems hold almost generically. Here, we show that sharpness directly controls the performance of restart schemes. The constants quantifying sharpness are of course unobservable, but we show that optimal restart strategies are fairly robust, and searching for the best scheme only increases the complexity by a logarithmic factor compared to the optimal bound. Overall then, restart schemes generically accelerate accelerated methods.

### #108: Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition
Not found on arxiv

### #109: Soft-to-Hard Vector Quantization for End-to-End Learning Compressible Representations
We present a new approach to learn compressible representations in deep architectures with an end-to-end training strategy. Our method is based on a soft (continuous) relaxation of quantization and entropy, which we anneal to their discrete counterparts throughout training. We showcase this method for two challenging applications: Image compression and neural network compression. While these tasks have typically been approached with different methods, our soft-to-hard quantization approach gives results competitive with the state-of-the-art for both.

### #110: Learning spatiotemporal piecewise-geodesic trajectories from longitudinal manifold-valued data
Not found on arxiv

### #111: Improving Regret Bounds for Combinatorial Semi-Bandits with Probabilistically Triggered Arms and Its Applications
Not found on arxiv

### #112: Predictive-State Decoders: Encoding the Future into Recurrent Networks
Not found on arxiv

### #113: Posterior sampling for reinforcement learning: worst-case regret bounds
Not found on arxiv

### #114: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
Not found on arxiv

### #115: Matching neural paths: transfer from recognition to correspondence search
Many machine learning tasks require finding per-part correspondences between objects. In this work we focus on low-level correspondences - a highly ambiguous matching problem. We propose to use a hierarchical semantic representation of the objects, coming from a convolutional neural network, to solve this ambiguity. Training it for low-level correspondence prediction directly might not be an option in some domains where the ground-truth correspondences are hard to obtain. We show how transfer from recognition can be used to avoid such training. Our idea is to mark parts as "matching" if their features are close to each other at all the levels of convolutional feature hierarchy (neural paths). Although the overall number of such paths is exponential in the number of layers, we propose a polynomial algorithm for aggregating all of them in a single backward pass. The empirical validation is done on the task of stereo correspondence and demonstrates that we achieve competitive results among the methods which do not use labeled target domain data.

### #116: Linearly constrained Gaussian processes
We consider a modification of the covariance function in Gaussian processes to correctly account for known linear constraints. By modelling the target function as a transformation of an underlying function, the constraints are explicitly incorporated in the model such that they are guaranteed to be fulfilled by any sample drawn or prediction made. We also propose a constructive procedure for designing the transformation operator and illustrate the result on both simulated and real-data examples.

### #117: Fixed-Rank Approximation of a Positive-Semidefinite Matrix from Streaming Data
Several important applications, such as streaming PCA and semidefinite programming, involve a large-scale positive-semidefinite (psd) matrix that is presented as a sequence of linear updates. Because of storage limitations, it may only be possible to retain a sketch of the psd matrix. This paper develops a new algorithm for fixed-rank psd approximation from a sketch. The approach combines the Nystrom approximation with a novel mechanism for rank truncation. Theoretical analysis establishes that the proposed method can achieve any prescribed relative error in the Schatten 1-norm and that it exploits the spectral decay of the input matrix. Computer experiments show that the proposed method dominates alternative techniques for fixed-rank psd matrix approximation across a wide range of examples.

### #118: Multi-Modal Imitation Learning from Unstructured Demonstrations using Generative Adversarial Nets
Imitation learning has traditionally been applied to learn a single task from demonstrations thereof. The requirement of structured and isolated demonstrations limits the scalability of imitation learning approaches as they are difficult to apply to real-world scenarios, where robots have to be able to execute a multitude of tasks. In this paper, we propose a multi-modal imitation learning framework that is able to segment and imitate skills from unlabelled and unstructured demonstrations by learning skill segmentation and imitation learning jointly. The extensive simulation results indicate that our method can efficiently separate the demonstrations into individual skills and learn to imitate them using a single multi-modal policy. The video of our experiments is available at this http URL

### #119: Learning to Inpaint for Image Compression
Not found on arxiv

### #120: Adaptive Bayesian Sampling with Monte Carlo EM
Not found on arxiv

### #121: No More Fixed Penalty  Parameter in ADMM: Faster Convergence with New Adaptive Penalization
Not found on arxiv

### #122: Shape and Material from Sound
Not found on arxiv

### #123: Flexible statistical inference for mechanistic models of neural dynamics
Not found on arxiv

### #124: Online Prediction with Selfish Experts
We consider the problem of binary prediction with expert advice in settings where experts have agency and seek to maximize their credibility. This paper makes three main contributions. First, it defines a model to reason formally about settings with selfish experts, and demonstrates that "incentive compatible" (IC) algorithms are closely related to the design of proper scoring rules. Designing a good IC algorithm is easy if the designer's loss function is quadratic, but for other loss functions, novel techniques are required. Second, we design IC algorithms with good performance guarantees for the absolute loss function. Third, we give a formal separation between the power of online prediction with selfish experts and online prediction with honest experts by proving lower bounds for both IC and non-IC algorithms. In particular, with selfish experts and the absolute loss function, there is no (randomized) algorithm for online prediction-IC or otherwise-with asymptotically vanishing regret.

### #125: Tensor Biclustering
Not found on arxiv

### #126: DPSCREEN: Dynamic Personalized Screening
Not found on arxiv

### #127: Learning Unknown Markov Decision Processes: A Thompson Sampling Approach
Not found on arxiv

### #128: Testing and Learning on Distributions with Symmetric Noise Invariance
Kernel embeddings of distributions and the Maximum Mean Discrepancy (MMD), the resulting distance between distributions, are useful tools for fully nonparametric two-sample testing and learning on distributions. However, it is rarely that all possible differences between samples are of interest -- discovered differences can be due to different types of measurement noise, data collection artefacts or other irrelevant sources of variability. We propose distances between distributions which encode invariance to additive symmetric noise, aimed at testing whether the assumed true underlying processes differ. Moreover, we construct invariant features of distributions, leading to learning algorithms robust to the impairment of the input distributions with symmetric additive noise. Such features lend themselves to a straightforward neural network implementation and can thus also be learned given a supervised signal.

### #129: A Dirichlet Mixture Model of Hawkes Processes for Event Sequence Clustering
We propose an effective method to solve the event sequence clustering problems based on a novel Dirichlet mixture model of a special but significant type of point processes --- Hawkes process. In this model, each event sequence belonging to a cluster is generated via the same Hawkes process with specific parameters, and different clusters correspond to different Hawkes processes. The prior distribution of the Hawkes processes is controlled via a Dirichlet distribution. We learn the model via a maximum likelihood estimator (MLE) and propose an effective variational Bayesian inference algorithm. We specifically analyze the resulting EM-type algorithm in the context of inner-outer iterations and discuss several inner iteration allocation strategies. The identifiability of our model, the convergence of our learning method, and its sample complexity are analyzed in both theoretical and empirical ways, which demonstrate the superiority of our method to other competitors. The proposed method learns the number of clusters automatically and is robust to model misspecification. Experiments on both synthetic and real-world data show that our method can learn diverse triggering patterns hidden in asynchronous event sequences and achieve encouraging performance on clustering purity and consistency.

### #130: Deanonymization in the Bitcoin P2P Network
Not found on arxiv

### #131: Accelerated consensus via Min-Sum Splitting
Not found on arxiv

### #132: Generalized Linear Model Regression under Distance-to-set Penalties
Not found on arxiv

### #133: Adaptive sampling for a population of neurons
Not found on arxiv

### #134: Nonbacktracking Bounds on the Influence in Independent Cascade Models
This paper develops upper and lower bounds on the influence measure in a network, more precisely, the expected number of nodes that a seed set can influence in the independent cascade model. In particular, our bounds exploit nonbacktracking walks, Fortuin-Kasteleyn-Ginibre (FKG) type inequalities, and are computed by message passing implementation. Nonbacktracking walks have recently allowed for headways in community detection, and this paper shows that their use can also impact the influence computation. Further, we provide a knob to control the trade-off between the efficiency and the accuracy of the bounds. Finally, the tightness of the bounds is illustrated with simulations on various network models.

### #135: Learning with Feature Evolvable Streams
Learning with streaming data has attracted much attention during the past few years. Though most studies consider data stream with fixed features, in real practice the features may be evolvable. For example, features of data gathered by limited-lifespan sensors will change when these sensors are substituted by new ones. In this paper, we propose a novel learning paradigm: Feature Evolvable Streaming Learning where old features would vanish and new features will occur. Rather than relying on only the current features, we attempt to recover the vanished features and exploit it to improve performance. Specifically, we learn two models from the recovered features and the current features, respectively. To benefit from the recovered features, we develop two ensemble methods. In the first method, we combine the predictions from two models and theoretically show that with assistance of old features, the performance on new features can be improved. In the second approach, we dynamically select the best single prediction and establish a better performance guarantee when the best model switches. Experiments on both synthetic and real data validate the effectiveness of our proposal.

### #136: Online Convex Optimization with Stochastic Constraints
This paper considers online convex optimization (OCO) with stochastic constraints, which generalizes Zinkevich's OCO over a known simple fixed set by introducing multiple stochastic functional constraints that are i.i.d. generated at each round and are disclosed to the decision maker only after the decision is made. This formulation arises naturally when decisions are restricted by stochastic environments or deterministic environments with noisy observations. It also includes many important problems as special cases, such as OCO with long term constraints, stochastic constrained convex optimization, and deterministic constrained convex optimization. To solve this problem, this paper proposes a new algorithm that achieves $O(\sqrt{T})$ expected regret and constraint violations and $O(\sqrt{T}\log(T))$ high probability regret and constraint violations. Experiments on a real-world data center scheduling problem further verify the performance of the new algorithm.

### #137: Max-Margin Invariant Features from Transformed Unlabelled Data
Not found on arxiv

### #138: Cognitive Impairment Prediction in Alzheimer’s Disease with Regularized Modal Regression
Not found on arxiv

### #139: Translation Synchronization via Truncated Least Squares
Not found on arxiv

### #140: From which world is your graph
Not found on arxiv

### #141: A New Alternating Direction Method for Linear Programming
Not found on arxiv

### #142: Regret Analysis for Continuous Dueling Bandit
Not found on arxiv

### #143: Best Response Regression
Not found on arxiv

### #144: TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning
High network communication cost for synchronizing gradients and parameters is the well-known bottleneck of distributed training. In this work, we propose TernGrad that uses ternary gradients to accelerate distributed deep learning in data parallelism. Our approach requires only three numerical levels {-1,0,1} which can aggressively reduce the communication time. We mathematically prove the convergence of TernGrad under the assumption of a bound on gradients. Guided by the bound, we propose layer-wise ternarizing and gradient clipping to improve its convergence. Our experiments show that applying TernGrad on AlexNet does not incur any accuracy loss and can even improve accuracy. The accuracy loss of GoogLeNet induced by TernGrad is less than 2% on average. Finally, a performance model is proposed to study the scalability of TernGrad. Experiments show significant speed gains for various deep neural networks.

### #145: Learning Affinity via Spatial Propagation Networks
Not found on arxiv

### #146: Linear regression without correspondence
This article considers algorithmic and statistical aspects of linear regression when the correspondence between the covariates and the responses is unknown. First, a fully polynomial-time approximation scheme is given for the natural least squares optimization problem in any constant dimension. Next, in an average-case and noise-free setting where the responses exactly correspond to a linear function of i.i.d. draws from a standard multivariate normal distribution, an efficient algorithm based on lattice basis reduction is shown to exactly recover the unknown linear function in arbitrary dimension. Finally, lower bounds on the signal-to-noise ratio are established for approximate recovery of the unknown linear function by any estimator.

### #147: NeuralFDR: Learning Discovery Thresholds from Hypothesis Features
Not found on arxiv

### #148: Cost efficient gradient boosting
### #148: Cost efficient gradient boosting
Not found on arxiv

### #149: Probabilistic Rule Realization and Selection
Abstraction and realization are bilateral processes that are key in deriving intelligence and creativity. In many domains, the two processes are approached through rules: high-level principles that reveal invariances within similar yet diverse examples. Under a probabilistic setting for discrete input spaces, we focus on the rule realization problem which generates input sample distributions that follow the given rules. More ambitiously, we go beyond a mechanical realization that takes whatever is given, but instead ask for proactively selecting reasonable rules to realize. This goal is demanding in practice, since the initial rule set may not always be consistent and thus intelligent compromises are needed. We formulate both rule realization and selection as two strongly connected components within a single and symmetric bi-convex problem, and derive an efficient algorithm that works at large scale. Taking music compositional rules as the main example throughout the paper, we demonstrate our model's efficiency in not only music realization (composition) but also music interpretation and understanding (analysis).

### #150: Nearest-Neighbor Sample Compression: Efficiency, Consistency, Infinite Dimensions
We examine the Bayes-consistency of a recently proposed 1-nearest-neighbor-based multiclass learning algorithm. This algorithm is derived from sample compression bounds and enjoys the statistical advantages of tight, fully empirical generalization bounds, as well as the algorithmic advantages of runtime and memory savings. We prove that this algorithm is strongly Bayes-consistent in metric spaces with finite doubling dimension --- the first consistency result for an efficient nearest-neighbor sample compression scheme. Rather surprisingly, we discover that this algorithm continues to be Bayes-consistent even in a certain infinite-dimensional setting, in which the basic measure-theoretic conditions on which classic consistency proofs hinge are violated. This is all the more surprising, since it is known that k-NN is not Bayes-consistent in this setting. We pose several challenging open problems for future research.

### #151: A Scale Free Algorithm for Stochastic Bandits with Bounded Kurtosis
Existing strategies for finite-armed stochastic bandits mostly depend on a parameter of scale that must be known in advance. Sometimes this is in the form of a bound on the payoffs, or the knowledge of a variance or subgaussian parameter. The notable exceptions are the analysis of Gaussian bandits with unknown mean and variance by Cowan and Katehakis [2015] and of uniform distributions with unknown support [Cowan and Katehakis, 2015]. The results derived in these specialised cases are generalised here to the non-parametric setup, where the learner knows only a bound on the kurtosis of the noise, which is a scale free measure of the extremity of outliers.

### #152: Learning Multiple Tasks with Deep Relationship Networks
Deep networks trained on large-scale data can learn transferable features to promote learning multiple tasks. As deep features eventually transition from general to specific along deep networks, a fundamental problem is how to exploit the relationship across different tasks and improve the feature transferability in the task-specific layers. In this paper, we propose Deep Relationship Networks (DRN) that discover the task relationship based on novel tensor normal priors over the parameter tensors of multiple task-specific layers in deep convolutional networks. By jointly learning transferable features and task relationships, DRN is able to alleviate the dilemma of negative-transfer in the feature layers and under-transfer in the classifier layer. Extensive experiments show that DRN yields state-of-the-art results on standard multi-task learning benchmarks.

### #153: Deep Hyperalignment
Not found on arxiv

### #154: Online to Offline Conversions and Adaptive Minibatch Sizes
We present an approach towards convex optimization that relies on a novel scheme which converts online adaptive algorithms into offline methods. In the offline optimization setting, our derived methods are shown to obtain favourable adaptive guarantees which depend on the harmonic sum of the queried gradients. We further show that our methods implicitly adapt to the objective's structure: in the smooth case fast convergence rates are ensured without any prior knowledge of the smoothness parameter, while still maintaining guarantees in the non-smooth setting. Our approach has a natural extension to the stochastic setting, resulting in a lazy version of SGD (stochastic GD), where minibathces are chosen \emph{adaptively} depending on the magnitude of the gradients. Thus providing a principled approach towards choosing minibatch sizes.

### #155: Stochastic Optimization with Variance Reduction for Infinite Datasets with Finite Sum Structure
Stochastic optimization algorithms with variance reduction have proven successful for minimizing large finite sums of functions. Unfortunately, these techniques are unable to deal with stochastic perturbations of input data, induced for example by data augmentation. In such cases, the objective is no longer a finite sum, and the main candidate for optimization is the stochastic gradient descent method (SGD). In this paper, we introduce a variance reduction approach for these settings when the objective is composite and strongly convex. The convergence rate outperforms SGD with a typically much smaller constant factor, which depends on the variance of gradient estimates only due to perturbations on a single example.

### #156: Deep Learning with Topological Signatures
Inferring topological and geometrical information from data can offer an alternative perspective on machine learning problems. Methods from topological data analysis, e.g., persistent homology, enable us to obtain such information, typically in the form of summary representations of topological features. However, such topological signatures often come with an unusual structure (e.g., multisets of intervals) that is highly impractical for most machine learning techniques. While many strategies have been proposed to map these topological signatures into machine learning compatible representations, they suffer from being agnostic to the target learning task. In contrast, we propose a technique that enables us to input topological signatures to deep neural networks and learn a task-optimal representation during training. Our approach is realized as a novel input layer with favorable theoretical properties. Classification experiments on 2D object shapes and social network graphs demonstrate the versatility of the approach and, in case of the latter, we even outperform the state-of-the-art by a large margin.

### #157: Predicting User Activity Level In Point Process Models With Mass Transport Equation
Not found on arxiv

### #158: Submultiplicative Glivenko-Cantelli and Uniform Convergence of Revenues
In this work we derive a variant of the classic Glivenko-Cantelli Theorem, which asserts uniform convergence of the empirical Cumulative Distribution Function (CDF) to the CDF of the underlying distribution. Our variant allows for tighter convergence bounds for extreme values of the CDF. We apply our bound in the context of revenue learning, which is a well-studied problem in economics and algorithmic game theory. We derive sample-complexity bounds on the uniform convergence rate of the empirical revenues to the true revenues, assuming a bound on the $k$th moment of the valuations, for any (possibly fractional) $k>1$. For uniform convergence in the limit, we give a complete characterization and a zero-one law: if the first moment of the valuations is finite, then uniform convergence almost surely occurs; conversely, if the first moment is infinite, then uniform convergence almost never occurs.

### #159: Deep Dynamic Poisson Factorization Model
Not found on arxiv

### #160: Positive-Unlabeled Learning with Non-Negative Risk Estimator
From only positive (P) and unlabeled (U) data, a binary classifier could be trained with PU learning. Unbiased PU learning that is based on unbiased risk estimators is now state of the art. However, if its model is very flexible, its empirical risk on training data will go negative, and we will suffer from overfitting seriously. In this paper, we propose a novel non-negative risk estimator for PU learning. When being minimized, it is more robust against overfitting, and thus we are able to train very flexible models given limited P data. Moreover, we analyze the bias, consistency and mean-squared-error reduction of the proposed risk estimator as well as the estimation error of the corresponding risk minimizer. Experiments show that the non-negative risk estimator outperforms unbiased counterparts when they disagree.

### #161: Optimal Sample Complexity of M-wise Data for Top-K Ranking
Not found on arxiv

### #162: What-If Reasoning using Counterfactual Gaussian Processes
Not found on arxiv

### #163: Communication-Efficient Stochastic Gradient Descent, with Applications to Neural Networks
Not found on arxiv

### #164: On the Convergence of Block Coordinate Descent in Training DNNs with Tikhonov Regularization
Not found on arxiv

### #165: Train longer, generalize better: closing the generalization gap in large batch training of neural networks
Background: Deep learning models are typically trained using stochastic gradient descent or one of its variants. These methods update the weights using their gradient, estimated from a small fraction of the training data. It has been observed that when using large batch sizes there is a persistent degradation in generalization performance - known as the "generalization gap" phenomena. Identifying the origin of this gap and closing it had remained an open problem. Contributions: We examine the initial high learning rate training phase. We find that the weight distance from its initialization grows logarithmically with the number of weight updates. We therefore propose a "random walk on random landscape" statistical model which is known to exhibit similar "ultra-slow" diffusion behavior. Following this hypothesis we conducted experiments to show empirically that the "generalization gap" stems from the relatively small number of updates rather than the batch size, and can be completely eliminated by adapting the training regime used. We further investigate different techniques to train models in the large-batch regime and present a novel algorithm named "Ghost Batch Normalization" which enables significant decrease in the generalization gap without increasing the number of updates. To validate our findings we conduct several additional experiments on MNIST, CIFAR-10, CIFAR-100 and ImageNet. Finally, we reassess common practices and beliefs concerning training of deep models and suggest they may not be optimal to achieve good generalization.

### #166: Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks
Not found on arxiv

### #167: Model evidence from nonequilibrium simulations
Not found on arxiv

### #168: Minimal Exploration in Structured Stochastic Bandits
Not found on arxiv

### #169: Learned D-AMP: Principled Neural-network-based Compressive Image Recovery
Not found on arxiv

### #170: Deliberation Networks: Sequence Generation Beyond One-Pass Decoding
Not found on arxiv

### #171: Adaptive Clustering through Semidefinite Programming
We analyze the clustering problem through a flexible probabilistic model that aims to identify an optimal partition on the sample X 1 , ..., X n. We perform exact clustering with high probability using a convex semidefinite estimator that interprets as a corrected, relaxed version of K-means. The estimator is analyzed through a non-asymptotic framework and showed to be optimal or near-optimal in recovering the partition. Furthermore, its performances are shown to be adaptive to the problem's effective dimension, as well as to K the unknown number of groups in this partition. We illustrate the method's performances in comparison to other classical clustering algorithms with numerical experiments on simulated data.

### #172: Log-normality and Skewness of Estimated State/Action Values in Reinforcement Learning
Not found on arxiv

### #173: Repeated Inverse Reinforcement Learning
How detailed should we make the goals we prescribe to AI agents acting on our behalf in complex environments? Detailed and low-level specification of goals can be tedious and expensive to create, and abstract and high-level goals could lead to negative surprises as the agent may find behaviors that we would not want it to do, i.e., lead to unsafe AI. One approach to addressing this dilemma is for the agent to infer human goals by observing human behavior. This is the Inverse Reinforcement Learning (IRL) problem. However, IRL is generally ill-posed for there are typically many reward functions for which the observed behavior is optimal. While the use of heuristics to select from among the set of feasible reward functions has led to successful applications of IRL to learning from demonstration, such heuristics do not address AI safety. In this paper we introduce a novel repeated IRL problem that captures an aspect of AI safety as follows. The agent has to act on behalf of a human in a sequence of tasks and wishes to minimize the number of tasks that it surprises the human. Each time the human is surprised the agent is provided a demonstration of the desired behavior by the human. We formalize this problem, including how the sequence of tasks is chosen, in a few different ways and provide some foundational results.

### #174: The Numerics of GANs
In this paper, we analyze the numerics of common algorithms for training Generative Adversarial Networks (GANs). Using the formalism of smooth two-player games we analyze the associated gradient vector field of GAN training objectives. Our findings suggest that the convergence of current algorithms suffers due to two factors: i) presence of eigenvalues of the Jacobian of the gradient vector field with zero real-part, and ii) eigenvalues with big imaginary part. Using these findings, we design a new algorithm that overcomes some of these limitations and has better convergence properties. Experimentally, we demonstrate its superiority on training common GAN architectures and show convergence on GAN architectures that are known to be notoriously hard to train.

### #175: Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search
Computational models in fields such as computational neuroscience are often evaluated via stochastic simulation or numerical approximation. Fitting these models implies a difficult optimization problem over complex, possibly noisy parameter landscapes. Bayesian optimization (BO) has been successfully applied to solving expensive black-box problems in engineering and machine learning. Here we explore whether BO can be applied as a general tool for model fitting. First, we present a novel BO algorithm, Bayesian adaptive direct search (BADS), that achieves competitive performance with an affordable computational overhead for the running time of typical models. We then perform an extensive benchmark of BADS vs. many common and state-of-the-art nonconvex, derivative-free optimizers on a set of model-fitting problems with real data and models from six studies in behavioral, cognitive, and computational neuroscience. With default settings, BADS consistently finds comparable or better solutions than other methods, showing great promise for BO, and BADS in particular, as a general model-fitting tool.

### #176: Learning Chordal Markov Networks via Branch and Bound
Not found on arxiv

### #177: Revenue Optimization with Approximate Bid Predictions
In the context of advertising auctions, finding good reserve prices is a notoriously challenging learning problem. This is due to the heterogeneity of ad opportunity types and the non-convexity of the objective function. In this work, we show how to reduce reserve price optimization to the standard setting of prediction under squared loss, a well understood problem in the learning community. We further bound the gap between the expected bid and revenue in terms of the average loss of the predictor. This is the first result that formally relates the revenue gained to the quality of a standard machine learned model.

### #178: Solving (Almost) all Systems of Random Quadratic Equations
Not found on arxiv

### #179: Unsupervised Learning of Disentangled Latent Representations from Sequential Data
Not found on arxiv

### #180: Lookahead  Bayesian Optimization with Inequality Constraints
Not found on arxiv

### #181: Hierarchical Methods of Moments
Not found on arxiv

### #182: Interpretable and Globally Optimal Prediction for Textual Grounding using Image Concepts
Not found on arxiv

### #183: Revisit Fuzzy Neural Network: Demystifying Batch Normalization and ReLU with Generalized Hamming Network
Not found on arxiv

### #184: Speeding Up Latent Variable Gaussian Graphical Model Estimation via Nonconvex Optimization
We study the estimation of the latent variable Gaussian graphical model (LVGGM), where the precision matrix is the superposition of a sparse matrix and a low-rank matrix. In order to speed up the estimation of the sparse plus low-rank components, we propose a sparsity constrained maximum likelihood estimator based on matrix factorization, and an efficient alternating gradient descent algorithm with hard thresholding to solve it. Our algorithm is orders of magnitude faster than the convex relaxation based methods for LVGGM. In addition, we prove that our algorithm is guaranteed to linearly converge to the unknown sparse and low-rank components up to the optimal statistical precision. Experiments on both synthetic and genomic data demonstrate the superiority of our algorithm over the state-of-the-art algorithms and corroborate our theory.

### #185: Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models
Batch Normalization is quite effective at accelerating and improving the training of deep models. However, its effectiveness diminishes when the training minibatches are small, or do not consist of independent samples. We hypothesize that this is due to the dependence of model layer inputs on all the examples in the minibatch, and different activations being produced between training and inference. We propose Batch Renormalization, a simple and effective extension to ensure that the training and inference models generate the same outputs that depend on individual examples rather than the entire minibatch. Models trained with Batch Renormalization perform substantially better than batchnorm when training with small or non-i.i.d. minibatches. At the same time, Batch Renormalization retains the benefits of batchnorm such as insensitivity to initialization and training efficiency.

### #186: Generating steganographic images via adversarial training
Not found on arxiv

### #187: Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration
Computing optimal transport distances such as the earth mover's distance is a fundamental problem in machine learning, statistics, and computer vision. Despite the recent introduction of several algorithms with good empirical performance, it is unknown whether general optimal transport distances can be approximated in near-linear time. This paper demonstrates that this ambitious goal is in fact achieved by Cuturi's Sinkhorn Distances, and provides guidance towards parameter tuning for this algorithm. This result relies on a new analysis of Sinkhorn iterations that also directly suggests a new algorithm Greenkhorn with the same theoretical guarantees. Numerical simulations illustrate that Greenkhorn significantly outperforms the classical Sinkhorn algorithm in practice.

### #188: PixelGAN Autoencoders
In this paper, we describe the "PixelGAN autoencoder", a generative autoencoder in which the generative path is a convolutional autoregressive neural network on pixels (PixelCNN) that is conditioned on a latent code, and the recognition path uses a generative adversarial network (GAN) to impose a prior distribution on the latent code. We show that different priors result in different decompositions of information between the latent code and the autoregressive decoder. For example, by imposing a Gaussian distribution as the prior, we can achieve a global vs. local decomposition, or by imposing a categorical distribution as the prior, we can disentangle the style and content information of images in an unsupervised fashion. We further show how the PixelGAN autoencoder with a categorical prior can be directly used in semi-supervised settings and achieve competitive semi-supervised classification results on the MNIST, SVHN and NORB datasets.

### #189: Consistent Multitask Learning with Nonlinear Output Relations
Key to multitask learning is exploiting relationships between different tasks to improve prediction performance. If the relations are linear, regularization approaches can be used successfully. However, in practice assuming the tasks to be linearly related might be restrictive, and allowing for nonlinear structures is a challenge. In this paper, we tackle this issue by casting the problem within the framework of structured prediction. Our main contribution is a novel algorithm for learning multiple tasks which are related by a system of nonlinear equations that their joint outputs need to satisfy. We show that the algorithm is consistent and can be efficiently implemented. Experimental results show the potential of the proposed method.

### #190: Fast Alternating Minimization Algorithms for Dictionary Learning
Not found on arxiv

### #191: Learning ReLUs via Gradient Descent
In this paper we study the problem of learning Rectified Linear Units (ReLUs) which are functions of the form $max(0,<w,x>)$ with $w$ denoting the weight vector. We study this problem in the high-dimensional regime where the number of observations are fewer than the dimension of the weight vector. We assume that the weight vector belongs to some closed set (convex or nonconvex) which captures known side-information about its structure. We focus on the realizable model where the inputs are chosen i.i.d.~from a Gaussian distribution and the labels are generated according to a planted weight vector. We show that projected gradient descent, when initialization at 0, converges at a linear rate to the planted model with a number of samples that is optimal up to numerical constants. Our results on the dynamics of convergence of these very shallow neural nets may provide some insights towards understanding the dynamics of deeper architectures.

### #192: Stabilizing Training of Generative Adversarial Networks through Regularization
Deep generative models based on Generative Adversarial Networks (GANs) have demonstrated impressive sample quality but in order to work they require a careful choice of architecture, parameter initialization, and selection of hyper-parameters. This fragility is in part due to a dimensional mismatch between the model distribution and the true distribution, causing their density ratio and the associated f-divergence to be undefined. We overcome this fundamental limitation and propose a new regularization approach with low computational cost that yields a stable GAN training procedure. We demonstrate the effectiveness of this approach on several datasets including common benchmark image generation tasks. Our approach turns GAN models into reliable building blocks for deep learning.

### #193: Expectation Propagation with Stochastic Kinetic Model in Complex Interaction Systems
Not found on arxiv

### #194: Data-Efficient Reinforcement Learning in Continuous State-Action Gaussian-POMDPs
Not found on arxiv

### #195: Compatible Reward Inverse Reinforcement Learning
Not found on arxiv

### #196: First-Order Adaptive Sample Size Methods to Reduce Complexity of Empirical Risk Minimization
This paper studies empirical risk minimization (ERM) problems for large-scale datasets and incorporates the idea of adaptive sample size methods to improve the guaranteed convergence bounds for first-order stochastic and deterministic methods. In contrast to traditional methods that attempt to solve the ERM problem corresponding to the full dataset directly, adaptive sample size schemes start with a small number of samples and solve the corresponding ERM problem to its statistical accuracy. The sample size is then grown geometrically -- e.g., scaling by a factor of two -- and use the solution of the previous ERM as a warm start for the new ERM. Theoretical analyses show that the use of adaptive sample size methods reduces the overall computational cost of achieving the statistical accuracy of the whole dataset for a broad range of deterministic and stochastic first-order methods. The gains are specific to the choice of method. When particularized to, e.g., accelerated gradient descent and stochastic variance reduce gradient, the computational cost advantage is a logarithm of the number of training samples. Numerical experiments on various datasets confirm theoretical claims and showcase the gains of using the proposed adaptive sample size scheme.

### #197: Hiding Images in Plain Sight: Deep Steganography
Not found on arxiv

### #198: Neural Program Meta-Induction
Not found on arxiv

### #199: Bayesian Dyadic Trees and Histograms for  Regression
Many machine learning tools for regression are based on recursive partitioning of the covariate space into smaller regions, where the regression function can be estimated locally. Among these, regression trees and their ensembles have demonstrated impressive empirical performance. In this work, we shed light on the machinery behind Bayesian variants of these methods. In particular, we study Bayesian regression histograms, such as Bayesian dyadic trees, in the simple regression case with just one predictor. We focus on the reconstruction of regression surfaces that are piecewise constant, where the number of jumps is unknown. We show that with suitably designed priors, posterior distributions concentrate around the true step regression function at the minimax rate (up to a log factor). These results do not require the knowledge of the true number of steps, nor the width of the true partitioning cells. Thus, Bayesian dyadic regression trees are fully adaptive and can recover the true piecewise regression function nearly as well as if we knew the exact number and location of jumps. Our results constitute the first step towards understanding why Bayesian trees and their ensembles have worked so well in practice. As an aside, we discuss prior distributions on balanced interval partitions and how they relate to a problem in geometric probability. Namely, we quantify the probability of covering the circumference of a circle with random arcs whose endpoints are confined to a grid, a new variant of the original problem.

### #200: A graph-theoretic approach to multitasking
Not found on arxiv

### #201: Consistent Robust Regression
Not found on arxiv

### #202:  Natural value approximators: learning when to trust past estimates
Not found on arxiv

### #203: Bandits Dueling on Partially Ordered Sets
Not found on arxiv

### #204: Elementary Symmetric Polynomials for Optimal Experimental Design
We revisit the classical problem of optimal experimental design (OED) under a new mathematical model grounded in a geometric motivation. Specifically, we introduce models based on elementary symmetric polynomials; these polynomials capture "partial volumes" and offer a graded interpolation between the widely used A-optimal design and D-optimal design models, obtaining each of them as special cases. We analyze properties of our models, and derive both greedy and convex-relaxation algorithms for computing the associated designs. Our analysis establishes approximation guarantees on these algorithms, while our empirical results substantiate our claims and demonstrate a curious phenomenon concerning our greedy method. Finally, as a byproduct, we obtain new results on the theory of elementary symmetric polynomials that may be of independent interest.

### #205: Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols
Learning to communicate through interaction, rather than relying on explicit supervision, is often considered a prerequisite for developing a general AI. We study a setting where two agents engage in playing a referential game and, from scratch, develop a communication protocol necessary to succeed in this game. Unlike previous work, we require that messages they exchange, both at train and test time, are in the form of a language (i.e. sequences of discrete symbols). We compare a reinforcement learning approach and one using a differentiable relaxation (straight-through Gumbel-softmax estimator) and observe that the latter is much faster to converge and it results in more effective protocols. Interestingly, we also observe that the protocol we induce by optimizing the communication success exhibits a degree of compositionality and variability (i.e. the same information can be phrased in different ways), both properties characteristic of natural languages. As the ultimate goal is to ensure that communication is accomplished in natural language, we also perform experiments where we inject prior information about natural language into our model and study properties of the resulting protocol.

### #206: Backprop without Learning Rates Through Coin Betting
Deep learning methods achieve state-of-the-art performance in many application scenarios. Yet, these methods require a significant amount of hyperparameters tuning in order to achieve the best results. In particular, tuning the learning rates in the stochastic optimization process is still one of the main bottlenecks. In this paper, we propose a new stochastic gradient descent procedure for deep networks that does not require any learning rate setting. Contrary to previous methods, we do not adapt the learning rates nor we make use of the assumed curvature of the objective function. Instead, we reduce the optimization process to a game of betting on a coin and propose a learning rate free optimal algorithm for this scenario. Theoretical convergence is proven for convex and quasi-convex functions and empirical evidence shows the advantage of our algorithm over popular stochastic gradient algorithms.

### #207: Pixels to Graphs by Associative Embedding
Graphs are a useful abstraction of image content. Not only can graphs represent details about individual objects in a scene but they can capture the interactions between pairs of objects. We present a method for training a convolutional neural network such that it takes in an input image and produces a full graph. This is done end-to-end in a single stage with the use of associative embeddings. The network learns to simultaneously identify all of the elements that make up a graph and piece them together. We benchmark on the Visual Genome dataset, and report a Recall@50 of 9.7% compared to the prior state-of-the-art at 3.4%, a nearly threefold improvement on the challenging task of scene graph generation.

### #208: Runtime Neural Pruning
Not found on arxiv

### #209: Compressing the Gram Matrix for Learning Neural Networks in Polynomial Time
Not found on arxiv

### #210: MMD GAN: Towards Deeper Understanding of Moment Matching Network
Generative moment matching network (GMMN) is a deep generative model that differs from Generative Adversarial Network (GAN) by replacing the discriminator in GAN with a two-sample test based on kernel maximum mean discrepancy (MMD). Although some theoretical guarantees of MMD have been studied, the empirical performance of GMMN is still not as competitive as that of GAN on challenging and large benchmark datasets. The computational efficiency of GMMN is also less desirable in comparison with GAN, partially due to its requirement for a rather large batch size during the training. In this paper, we propose to improve both the model expressiveness of GMMN and its computational efficiency by introducing adversarial kernel learning techniques, as the replacement of a fixed Gaussian kernel in the original GMMN. The new approach combines the key ideas in both GMMN and GAN, hence we name it MMD-GAN. The new distance measure in MMD-GAN is a meaningful loss that enjoys the advantage of weak topology and can be optimized via gradient descent with relatively small batch sizes. In our evaluation on multiple benchmark datasets, including MNIST, CIFAR- 10, CelebA and LSUN, the performance of MMD-GAN significantly outperforms GMMN, and is competitive with other representative GAN works.

### #211: The Reversible Residual Network: Backpropagation Without Storing Activations
Deep residual networks (ResNets) have significantly pushed forward the state-of-the-art on image classification, increasing in performance as networks grow both deeper and wider. However, memory consumption becomes a bottleneck, as one needs to store the activations in order to calculate gradients using backpropagation. We present the Reversible Residual Network (RevNet), a variant of ResNets where each layer's activations can be reconstructed exactly from the next layer's. Therefore, the activations for most layers need not be stored in memory during backpropagation. We demonstrate the effectiveness of RevNets on CIFAR-10, CIFAR-100, and ImageNet, establishing nearly identical classification accuracy to equally-sized ResNets, even though the activation storage requirements are independent of depth.

### #212: Fast Rates for Bandit Optimization with Upper-Confidence Frank-Wolfe
Not found on arxiv

### #213: Zap Q-Learning
Not found on arxiv

### #214: Expectation Propagation for t-Exponential Family Using Q-Algebra
Exponential family distributions are highly useful in machine learning since their calculation can be performed efficiently through natural parameters. The exponential family has recently been extended to the t-exponential family, which contains Student-t distributions as family members and thus allows us to handle noisy data well. However, since the t-exponential family is denied by the deformed exponential, we cannot derive an efficient learning algorithm for the t-exponential family such as expectation propagation (EP). In this paper, we borrow the mathematical tools of q-algebra from statistical physics and show that the pseudo additivity of distributions allows us to perform calculation of t-exponential family distributions through natural parameters. We then develop an expectation propagation (EP) algorithm for the t-exponential family, which provides a deterministic approximation to the posterior or predictive distribution with simple moment matching. We finally apply the proposed EP algorithm to the Bayes point machine and Student-t process classication, and demonstrate their performance numerically.

### #215: Few-Shot Learning Through an Information Retrieval  Lens
Few-shot learning refers to understanding new concepts from only a few examples. We propose an information retrieval-inspired approach for this problem that is motivated by the increased importance of maximally leveraging all the available information in this low-data regime. We define a training objective that aims to extract as much information as possible from each training batch by effectively optimizing over all relative orderings of the batch points simultaneously. In particular, we view each batch point as a `query' that ranks the remaining ones based on its predicted relevance to them and we define a model within the framework of structured prediction to optimize mean Average Precision over these rankings. Our method produces state-of-the-art results on standard few-shot learning benchmarks.

### #216: Formal Guarantees on the Robustness of a Classifier against Adversarial Manipulation
Recent work has shown that state-of-the-art classifiers are quite brittle, in the sense that a small adversarial change of an originally with high confidence correctly classified input leads to a wrong classification again with high confidence. This raises concerns that such classifiers are vulnerable to attacks and calls into question their usage in safety-critical systems. We show in this paper for the first time formal guarantees on the robustness of a classifier by giving instance-specific lower bounds on the norm of the input manipulation required to change the classifier decision. Based on this analysis we propose the Cross-Lipschitz regularization functional. We show that using this form of regularization in kernel methods resp. neural networks improves the robustness of the classifier without any loss in prediction performance.

### #217: Associative Embedding: End-to-End Learning for Joint Detection and Grouping
We introduce associative embedding, a novel method for supervising convolutional neural networks for the task of detection and grouping. A number of computer vision problems can be framed in this manner including multi-person pose estimation, instance segmentation, and multi-object tracking. Usually the grouping of detections is achieved with multi-stage pipelines, instead we propose an approach that teaches a network to simultaneously output detections and group assignments. This technique can be easily integrated into any state-of-the-art network architecture that produces pixel-wise predictions. We show how to apply this method to both multi-person pose estimation and instance segmentation and report state-of-the-art performance for multi-person pose on the MPII and MS-COCO datasets.

### #218: Practical Locally Private Heavy Hitters
We present new practical local differentially private heavy hitters algorithms achieving optimal or near-optimal worst-case error and running time -- TreeHist and Bitstogram. In both algorithms, server running time is $\tilde O(n)$ and user running time is $\tilde O(1)$, hence improving on the prior state-of-the-art result of Bassily and Smith [STOC 2015] requiring $O(n^{5/2})$ server time and $O(n^{3/2})$ user time. With a typically large number of participants in local algorithms ($n$ in the millions), this reduction in time complexity, in particular at the user side, is crucial for making locally private heavy hitters algorithms usable in practice. We implemented Algorithm TreeHist to verify our theoretical analysis and compared its performance with the performance of Google's RAPPOR code.

### #219: Large-Scale Quadratically Constrained Quadratic Program via Low-Discrepancy Sequences
Not found on arxiv

### #220: Inhomogoenous Hypergraph Clustering with Applications
Hypergraph partitioning is an important problem in machine learning, computer vision and network analytics. A widely used method for hypergraph partitioning relies on minimizing a normalized sum of the costs of partitioning hyperedges across clusters. Algorithmic solutions based on this approach assume that different partitions of a hyperedge incur the same cost. However, this assumption fails to leverage the fact that different subsets of vertices within the same hyperedge may have different structural importance. We hence propose a new hypergraph clustering technique, termed inhomogeneous hypergraph partitioning, which assigns different costs to different hyperedge cuts. We prove that inhomogeneous partitioning produces a quadratic approximation to the optimal solution if the inhomogeneous costs satisfy submodularity constraints. Moreover, we demonstrate that inhomogenous partitioning offers significant performance improvements in applications such as structure learning of rankings, subspace segmentation and motif clustering.

### #221: Differentiable Learning of Logical Rules for Knowledge Base Reasoning
We study the problem of learning probabilistic first-order logical rules for knowledge base reasoning. This learning problem is difficult because it requires learning the parameters in a continuous space as well as the structure in a discrete space. We propose a framework, Neural Logic Programming, that combines the parameter and structure learning of first-order logical rules in an end-to-end differentiable model. This approach is inspired by a recently-developed differentiable logic called TensorLog, where inference tasks can be compiled into sequences of differentiable operations. We design a neural controller system that learns to compose these operations. Empirically, our method obtains state-of-the-art results on multiple knowledge base benchmark datasets, including Freebase and WikiMovies.

### #222: Deep Multi-task Gaussian Processes for Survival Analysis with Competing Risks
Not found on arxiv

### #223: Masked Autoregressive Flow for Density Estimation
Autoregressive models are among the best performing neural density estimators. We describe an approach for increasing the flexibility of an autoregressive model, based on modelling the random numbers that the model uses internally when generating data. By constructing a stack of autoregressive models, each modelling the random numbers of the next model in the stack, we obtain a type of normalizing flow suitable for density estimation, which we call Masked Autoregressive Flow. This type of flow is closely related to Inverse Autoregressive Flow and is a generalization of Real NVP. Masked Autoregressive Flow achieves state-of-the-art performance in a range of general-purpose density estimation tasks.

### #224: Non-convex Finite-Sum Optimization Via SCSG Methods
Not found on arxiv

### #225: Beyond normality: Learning sparse probabilistic graphical models in the non-Gaussian setting
Not found on arxiv

### #226: Inner-loop free ADMM using Auxiliary Deep Neural Networks
Not found on arxiv

### #227: OnACID: Online Analysis of Calcium Imaging Data in Real Time
Not found on arxiv

### #228: Collaborative PAC Learning
Not found on arxiv

### #229: Fast Black-box Variational Inference through Stochastic Trust-Region Optimization
We introduce TrustVI, a fast second-order algorithm for black-box variational inference based on trust-region optimization and the reparameterization trick. At each iteration, TrustVI proposes and assesses a step based on minibatches of draws from the variational distribution. The algorithm provably converges to a stationary point. We implement TrustVI in the Stan framework and compare it to ADVI. TrustVI typically converges in tens of iterations to a solution at least as good as the one that ADVI reaches in thousands of iterations. TrustVI iterations can be more computationally expensive, but total computation is typically an order of magnitude less in our experiments.

### #230: Scalable Demand-Aware Recommendation
Not found on arxiv

### #231: SGD Learns the Conjugate Kernel Class of the Network
We show that the standard stochastic gradient decent (SGD) algorithm is guaranteed to learn, in polynomial time, a function that is competitive with the best function in the conjugate kernel space of the network, as defined in Daniely, Frostig and Singer. The result holds for log-depth networks from a rich family of architectures. To the best of our knowledge, it is the first polynomial-time guarantee for the standard neural network learning algorithm for networks of depth more that two. As corollaries, it follows that for neural networks of any depth between $2$ and $\log(n)$, SGD is guaranteed to learn, in polynomial time, constant degree polynomials with polynomially bounded coefficients. Likewise, it follows that SGD on large enough networks can learn any continuous function (not in polynomial time), complementing classical expressivity results.

### #232: Noise-Tolerant Interactive Learning Using Pairwise Comparisons
Not found on arxiv

### #233: Analyzing Hidden Representations in End-to-End Automatic Speech Recognition Systems
Not found on arxiv

### #234: Generative Local Metric Learning for Kernel Regression
Not found on arxiv

### #235: Information Theoretic Properties of Markov Random Fields, and their Algorithmic Applications
Markov random fields area popular model for high-dimensional probability distributions. Over the years, many mathematical, statistical and algorithmic problems on them have been studied. Until recently, the only known algorithms for provably learning them relied on exhaustive search, correlation decay or various incoherence assumptions. Bresler gave an algorithm for learning general Ising models on bounded degree graphs. His approach was based on a structural result about mutual information in Ising models. Here we take a more conceptual approach to proving lower bounds on the mutual information through setting up an appropriate zero-sum game. Our proof generalizes well beyond Ising models, to arbitrary Markov random fields with higher order interactions. As an application, we obtain algorithms for learning Markov random fields on bounded degree graphs on $n$ nodes with $r$-order interactions in $n^r$ time and $\log n$ sample complexity. The sample complexity is information theoretically optimal up to the dependence on the maximum degree. The running time is nearly optimal under standard conjectures about the hardness of learning parity with noise.

### #236: Fitting Low-Rank Tensors in Constant Time
Not found on arxiv

### #237: Deep supervised discrete hashing
Not found on arxiv

### #238: Using Options and Covariance Testing for Long Horizon Off-Policy Policy Evaluation
Not found on arxiv

### #239: How regularization affects the critical points in linear networks
Not found on arxiv

### #240: Fisher GAN
Generative Adversarial Networks (GANs) are powerful models for learning complex distributions. Stable training of GANs has been addressed in many recent works which explore different metrics between distributions. In this paper we introduce Fisher GAN which fits within the Integral Probability Metrics (IPM) framework for training GANs. Fisher GAN defines a critic with a data dependent constraint on its second order moments. We show in this paper that Fisher GAN allows for stable and time efficient training that does not compromise the capacity of the critic, and does not need data independent constraints such as weight clipping. We analyze our Fisher IPM theoretically and provide an algorithm based on Augmented Lagrangian for Fisher GAN. We validate our claims on both image sample generation and semi-supervised classification using Fisher GAN.

### #241: Information-theoretic analysis of generalization capability of learning algorithms
We derive upper bounds on the generalization error of a learning algorithm in terms of the mutual information between its input and output. The upper bounds provide theoretical guidelines for striking the right balance between data fit and generalization by controlling the input-output mutual information of a learning algorithm. The results can also be used to analyze the generalization capability of learning algorithms under adaptive composition, and the bias-accuracy tradeoffs in adaptive data analytics. Our work extends and leads to nontrivial improvements on the recent results of Russo and Zou.

### #242: Sparse Approximate Conic Hulls
Not found on arxiv

### #243: Rigorous Dynamics and Consistent Estimation in Arbitrarily Conditioned Linear Systems
The problem of estimating a random vector x from noisy linear measurements y = A x + w with unknown parameters on the distributions of x and w, which must also be learned, arises in a wide range of statistical learning and linear inverse problems. We show that a computationally simple iterative message-passing algorithm can provably obtain asymptotically consistent estimates in a certain high-dimensional large-system limit (LSL) under very general parameterizations. Previous message passing techniques have required i.i.d. sub-Gaussian A matrices and often fail when the matrix is ill-conditioned. The proposed algorithm, called adaptive vector approximate message passing (Adaptive VAMP) with auto-tuning, applies to all right-rotationally random A. Importantly, this class includes matrices with arbitrarily poor conditioning. We show that the parameter estimates and mean squared error (MSE) of x in each iteration converge to deterministic limits that can be precisely predicted by a simple set of state evolution (SE) equations. In addition, a simple testable condition is provided in which the MSE matches the Bayes-optimal value predicted by the replica method. The paper thus provides a computationally simple method with provable guarantees of optimality and consistency over a large class of linear inverse problems.

### #244: Toward Goal-Driven Neural Network Models for the Rodent Whisker-Trigeminal System
In large part, rodents see the world through their whiskers, a powerful tactile sense enabled by a series of brain areas that form the whisker-trigeminal system. Raw sensory data arrives in the form of mechanical input to the exquisitely sensitive, actively-controllable whisker array, and is processed through a sequence of neural circuits, eventually arriving in cortical regions that communicate with decision-making and memory areas. Although a long history of experimental studies has characterized many aspects of these processing stages, the computational operations of the whisker-trigeminal system remain largely unknown. In the present work, we take a goal-driven deep neural network (DNN) approach to modeling these computations. First, we construct a biophysically-realistic model of the rat whisker array. We then generate a large dataset of whisker sweeps across a wide variety of 3D objects in highly-varying poses, angles, and speeds. Next, we train DNNs from several distinct architectural families to solve a shape recognition task in this dataset. Each architectural family represents a structurally-distinct hypothesis for processing in the whisker-trigeminal system, corresponding to different ways in which spatial and temporal information can be integrated. We find that most networks perform poorly on the challenging shape recognition task, but that specific architectures from several families can achieve reasonable performance levels. Finally, we show that Representational Dissimilarity Matrices (RDMs), a tool for comparing population codes between neural systems, can separate these higher-performing networks with data of a type that could plausibly be collected in a neurophysiological or imaging experiment. Our results are a proof-of-concept that goal-driven DNN networks of the whisker-trigeminal system are potentially within reach.

### #245: Accuracy First: Selecting a Differential Privacy Level for Accuracy Constrained ERM
Traditional approaches to differential privacy assume a fixed privacy requirement $\epsilon$ for a computation, and attempt to maximize the accuracy of the computation subject to the privacy constraint. As differential privacy is increasingly deployed in practical settings, it may often be that there is instead a fixed accuracy requirement for a given computation and the data analyst would like to maximize the privacy of the computation subject to the accuracy constraint. This raises the question of how to find and run a maximally private empirical risk minimizer subject to a given accuracy requirement. We propose a general "noise reduction" framework that can apply to a variety of private empirical risk minimization (ERM) algorithms, using them to "search" the space of privacy levels to find the empirically strongest one that meets the accuracy constraint, incurring only logarithmic overhead in the number of privacy levels searched. The privacy analysis of our algorithm leads naturally to a version of differential privacy where the privacy parameters are dependent on the data, which we term ex-post privacy, and which is related to the recently introduced notion of privacy odometers. We also give an ex-post privacy analysis of the classical AboveThreshold privacy tool, modifying it to allow for queries chosen depending on the database. Finally, we apply our approach to two common objectives, regularized linear and logistic regression, and empirically compare our noise reduction methods to (i) inverting the theoretical utility guarantees of standard private ERM algorithms and (ii) a stronger, empirical baseline based on binary search.

### #246: EX2: Exploration with Exemplar Models for Deep Reinforcement Learning
Deep reinforcement learning algorithms have been shown to learn complex tasks using highly general policy classes. However, sparse reward problems remain a significant challenge. Exploration methods based on novelty detection have been particularly successful in such settings but typically require generative or predictive models of the observations, which can be difficult to train when the observations are very high-dimensional and complex, as in the case of raw images. We propose a novelty detection algorithm for exploration that is based entirely on discriminatively trained exemplar models, where classifiers are trained to discriminate each visited state against all others. Intuitively, novel states are easier to distinguish against other states seen during training. We show that this kind of discriminative modeling corresponds to implicit density estimation, and that it can be combined with count-based exploration to produce competitive results on a range of popular benchmark tasks, including state-of-the-art results on challenging egocentric observations in the vizDoom benchmark.

### #247: Multitask Spectral Learning of Weighted Automata
Not found on arxiv

### #248: Multi-way Interacting Regression via Factorization Machines
Not found on arxiv

### #249: Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network
Not found on arxiv

### #250: Practical Data-Dependent Metric Compression with Provable Guarantees
Not found on arxiv

### #251: REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models
Learning in models with discrete latent variables is challenging due to high variance gradient estimators. Generally, approaches have relied on control variates to reduce the variance of the REINFORCE estimator. Recent work (Jang et al. 2016; Maddison et al. 2016) has taken a different approach, introducing a continuous relaxation of discrete variables to produce low-variance, but biased, gradient estimates. In this work, we combine the two approaches through a novel control variate that produces low-variance, \emph{unbiased} gradient estimates. Then, we introduce a novel continuous relaxation and show that the tightness of the relaxation can be adapted online, removing it as a hyperparameter. We show state-of-the-art variance reduction on several benchmark generative modeling tasks, generally leading to faster convergence to a better final log likelihood.

### #252: Nonlinear random matrix theory for deep learning
Not found on arxiv

### #253: Parallel Streaming Wasserstein Barycenters
Efficiently aggregating data from different sources is a challenging problem, particularly when samples from each source are distributed differently. These differences can be inherent to the inference task or present for other reasons: sensors in a sensor network may be placed far apart, affecting their individual measurements. Conversely, it is computationally advantageous to split Bayesian inference tasks across subsets of data, but data need not be identically distributed across subsets. One principled way to fuse probability distributions is via the lens of optimal transport: the Wasserstein barycenter is a single distribution that summarizes a collection of input measures while respecting their geometry. However, computing the barycenter scales poorly and requires discretization of all input distributions and the barycenter itself. Improving on this situation, we present a scalable, communication-efficient, parallel algorithm for computing the Wasserstein barycenter of arbitrary distributions. Our algorithm can operate directly on continuous input distributions and is optimized for streaming data. Our method is even robust to nonstationary input distributions and produces a barycenter estimate that tracks the input measures over time. The algorithm is semi-discrete, needing to discretize only the barycenter estimate. To the best of our knowledge, we also provide the first bounds on the quality of the approximate barycenter as the discretization becomes finer. Finally, we demonstrate the practical effectiveness of our method, both in tracking moving distributions on a sphere, as well as in a large-scale Bayesian inference task.

### #254: ELF: An Extensive, Lightweight and Flexible Research Platform for Real-time Strategy Games
In this paper, we propose ELF, an Extensive, Lightweight and Flexible platform for fundamental reinforcement learning research. Using ELF, we implement a highly customizable real-time strategy (RTS) engine with three game environments (Mini-RTS, Capture the Flag and Tower Defense). Mini-RTS, as a miniature version of StarCraft, captures key game dynamics and runs at 40K frame-per-second (FPS) per core on a Macbook Pro notebook. When coupled with modern reinforcement learning methods, the system can train a full-game bot against built-in AIs end-to-end in one day with 6 CPUs and 1 GPU. In addition, our platform is flexible in terms of environment-agent communication topologies, choices of RL methods, changes in game parameters, and can host existing C/C++-based game environments like Arcade Learning Environment. Using ELF, we thoroughly explore training parameters and show that a network with Leaky ReLU and Batch Normalization coupled with long-horizon training and progressive curriculum beats the rule-based built-in AI more than $70\%$ of the time in the full game of Mini-RTS. Strong performance is also achieved on the other two games. In game replays, we show our agents learn interesting strategies. ELF, along with its RL platform, will be open-sourced.

### #255: Dual Discriminator Generative Adversarial Nets
Not found on arxiv

### #256: Dynamic Revenue Sharing
Not found on arxiv

### #257: Decomposition-Invariant Conditional Gradient for General Polytopes with Line Search
Not found on arxiv

### #258: Multi-agent Predictive Modeling with Attentional CommNets
Not found on arxiv

### #259: An Empirical Bayes Approach to Optimizing Machine Learning Algorithms
Not found on arxiv

### #260: Differentially Private Empirical Risk Minimization Revisited: Faster and More General
In this paper, we initiate a systematic investigation of differentially private algorithms for convex empirical risk minimization. Various instantiations of this problem have been studied before. We provide new algorithms and matching lower bounds for private ERM assuming only that each data point's contribution to the loss function is Lipschitz bounded and that the domain of optimization is bounded. We provide a separate set of algorithms and matching lower bounds for the setting in which the loss functions are known to also be strongly convex. Our algorithms run in polynomial time, and in some cases even match the optimal non-private running time (as measured by oracle complexity). We give separate algorithms (and lower bounds) for $(\epsilon,0)$- and $(\epsilon,\delta)$-differential privacy; perhaps surprisingly, the techniques used for designing optimal algorithms in the two cases are completely different. Our lower bounds apply even to very simple, smooth function families, such as linear and quadratic functions. This implies that algorithms from previous work can be used to obtain optimal error rates, under the additional assumption that the contributions of each data point to the loss function is smooth. We show that simple approaches to smoothing arbitrary loss functions (in order to apply previous techniques) do not yield optimal error rates. In particular, optimal algorithms were not previously known for problems such as training support vector machines and the high-dimensional median.

### #261: Variational Inference via $\chi$ Upper Bound Minimization
Not found on arxiv

### #262: On Quadratic Convergence of DC Proximal Newton Algorithm in Nonconvex Sparse Learning
We propose a DC proximal Newton algorithm for solving nonconvex regularized sparse learning problems in high dimensions. Our proposed algorithm integrates the proximal Newton algorithm with multi-stage convex relaxation based on difference of convex (DC) programming, and enjoys both strong computational and statistical guarantees. Specifically, by leveraging a sophisticated characterization of sparse modeling structures/assumptions (i.e., local restricted strong convexity and Hessian smoothness), we prove that within each stage of convex relaxation, our proposed algorithm achieves (local) quadratic convergence, and eventually obtains a sparse approximate local optimum with optimal statistical properties after only a few convex relaxations. Numerical experiments are provided to support our theory.

