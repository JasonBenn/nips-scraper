### #1: Wider and Deeper, Cheaper and Faster: Tensorized LSTMs for Sequence Learning

### #2: Concentration of Multilinear Functions of the Ising Model with Applications to Network Data

### #3: Deep Subspace Clustering Network
_Pan Ji,  Tong Zhang,  Hongdong Li,  Mathieu Salzmann,  Ian Reid_

We present a novel deep neural network architecture for unsupervised subspace clustering. This architecture is built upon deep auto-encoders, which non-linearly map the input data into a latent space. Our key idea is to introduce a novel self-expressive layer between the encoder and the decoder to mimic the "self-expressiveness" property that has proven effective in traditional subspace clustering. Being differentiable, our new self-expressive layer provides a simple but effective way to learn pairwise affinities between all data points through a standard back-propagation procedure. Being nonlinear, our neural-network based method is able to cluster data points having complex (often nonlinear) structures. We further propose pre-training and fine-tuning strategies that let us effectively learn the parameters of our subspace clustering networks. Our experiments show that the proposed method significantly outperforms the state-of-the-art unsupervised subspace clustering methods.

### #4: Attentional Pooling for Action Recognition

### #5: On the Consistency of Quick Shift

### #6: Rethinking Feature Discrimination and Polymerization for Large-scale Recognition
_Yu Liu,  Hongyang Li,  Xiaogang Wang_

Feature matters. How to train a deep network to acquire discriminative features across categories and polymerized features within classes has always been at the core of many computer vision tasks, specially for large-scale recognition systems where test identities are unseen during training and the number of classes could be at million scale. In this paper, we address this problem based on the simple intuition that the cosine distance of features in high-dimensional space should be close enough within one class and far away across categories. To this end, we proposed the congenerous cosine (COCO) algorithm to simultaneously optimize the cosine similarity among data. It inherits the softmax property to make inter-class features discriminative as well as shares the idea of class centroid in metric learning. Unlike previous work where the center is a temporal, statistical variable within one mini-batch during training, the formulated centroid is responsible for clustering inner-class features to enforce them polymerized around the network truncus. COCO is bundled with discriminative training and learned end-to-end with stable convergence. Experiments on five benchmarks have been extensively conducted to verify the effectiveness of our approach on both small-scale classification task and large-scale human recognition problem.

### #7: Breaking the Nonsmooth Barrier: A Scalable Parallel Method for Composite Optimization
_Fabian Pedregosa,  Rémi Leblond,  Simon Lacoste-Julien_

Due to their simplicity and excellent performance, parallel asynchronous variants of stochastic gradient descent have become popular methods to solve a wide range of large-scale optimization problems on multi-core architectures. Yet, despite their practical success, support for nonsmooth objectives is still lacking, making them unsuitable for many problems of interest in machine learning, such as the Lasso, group Lasso or empirical risk minimization with convex constraints. In this work, we propose and analyze ProxASAGA, a fully asynchronous sparse method inspired by SAGA, a variance reduced incremental gradient algorithm. The proposed method is easy to implement and significantly outperforms the state of the art on several nonsmooth, large-scale problems. We prove that our method achieves a theoretical linear speedup with respect to the sequential version under assumptions on the sparsity of gradients and block-separability of the proximal term. Empirical benchmarks on a multi-core architecture illustrate practical speedups of up to 12x on a 20-core machine.
[Abstract](https://arxiv.org/abs/1707.06468), [PDF](https://arxiv.org/pdf/1707.06468)


### #8: Dual-Agent GANs for Photorealistic and Identity Preserving Profile Face Synthesis

### #9: Dilated Recurrent Neural Networks
_Shiyu Chang,  Yang Zhang,  Wei Han,  Mo Yu,  Xiaoxiao Guo,  Wei Tan,  Xiaodong Cui,  Michael Witbrock,  Mark Hasegawa-Johnson,  Thomas Huang_

Notoriously, learning with recurrent neural networks (RNNs) on long sequences is a difficult task. There are three major challenges: 1) extracting complex dependencies, 2) vanishing and exploding gradients, and 3) efficient parallelization. In this paper, we introduce a simple yet effective RNN connection structure, the DILATEDRNN, which simultaneously tackles all these challenges. The proposed architecture is characterized by multi-resolution dilated recurrent skip connections and can be combined flexibly with different RNN cells. Moreover, the DILATEDRNN reduces the number of parameters and enhances training efficiency significantly, while matching state-of-the-art performance (even with Vanilla RNN cells) in tasks involving very long-term dependencies. To provide a theory-based quantification of the architecture's advantages, we introduce a memory capacity measure - the mean recurrent length, which is more suitable for RNNs with long skip connections than existing measures. We rigorously prove the advantages of the DILATEDRNN over other recurrent neural architectures.

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
_Haotian Pang,  Tuo Zhao,  Robert Vanderbei,  Han Liu_

High dimensional sparse learning has imposed a great computational challenge to large scale data analysis. In this paper, we are interested in a broad class of sparse learning approaches formulated as linear programs parametrized by a {\em regularization factor}, and solve them by the parametric simplex method (PSM). Our parametric simplex method offers significant advantages over other competing methods: (1) PSM naturally obtains the complete solution path for all values of the regularization parameter; (2) PSM provides a high precision dual certificate stopping criterion; (3) PSM yields sparse solutions through very few iterations, and the solution sparsity significantly reduces the computational cost per iteration. Particularly, we demonstrate the superiority of PSM over various sparse learning approaches, including Dantzig selector for sparse linear regression, LAD-Lasso for sparse robust linear regression, CLIME for sparse precision matrix estimation, sparse differential network estimation, and sparse Linear Programming Discriminant (LPD) analysis. We then provide sufficient conditions under which PSM always outputs sparse solutions such that its computational performance can be significantly boosted. Thorough numerical experiments are provided to demonstrate the outstanding performance of the PSM method.
[Abstract](https://arxiv.org/abs/1704.01079), [PDF](https://arxiv.org/pdf/1704.01079)


### #20: Group Sparse Additive Machine

### #21: Uprooting and Rerooting Higher-order Graphical Models

### #22: The Unreasonable Effectiveness of Structured Random Orthogonal Embeddings

### #23: From Parity to Preference: Learning with Cost-effective Notions of Fairness

### #24: Inferring Generative Model Structure with Static Analysis
_Paroma Varma,  Bryan He,  Payal Bajaj,  Imon Banerjee,  Nishith Khandwala,  Daniel L. Rubin,  Christopher Ré_

Obtaining enough labeled data to robustly train complex discriminative models is a major bottleneck in the machine learning pipeline. A popular solution is combining multiple sources of weak supervision using generative models. The structure of these models affects training label quality, but is difficult to learn without any ground truth labels. We instead rely on these weak supervision sources having some structure by virtue of being encoded programmatically. We present Coral, a paradigm that infers generative model structure by statically analyzing the code for these heuristics, thus reducing the data required to learn structure significantly. We prove that Coral's sample complexity scales quasilinearly with the number of heuristics and number of relations found, improving over the standard sample complexity, which is exponential in $n$ for identifying $n^{\textrm{th}}$ degree relations. Experimentally, Coral matches or outperforms traditional structure learning approaches by up to 3.81 F1 points. Using Coral to model dependencies instead of assuming independence results in better performance than a fully supervised model by 3.07 accuracy points when heuristics are used to label radiology data without ground truth labels.
[Abstract](https://arxiv.org/abs/1709.02477), [PDF](https://arxiv.org/pdf/1709.02477)


### #25: Structured Embedding Models for Grouped Data
_Maja Rudolph,  Francisco Ruiz,  Susan Athey,  David Blei_

Word embeddings are a powerful approach for analyzing language, and exponential family embeddings (EFE) extend them to other types of data. Here we develop structured exponential family embeddings (S-EFE), a method for discovering embeddings that vary across related groups of data. We study how the word usage of U.S. Congressional speeches varies across states and party affiliation, how words are used differently across sections of the ArXiv, and how the co-purchase patterns of groceries can vary across seasons. Key to the success of our method is that the groups share statistical information. We develop two sharing strategies: hierarchical modeling and amortization. We demonstrate the benefits of this approach in empirical studies of speeches, abstracts, and shopping baskets. We show how S-EFE enables group-specific interpretation of word usage, and outperforms EFE in predicting held-out data.

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
_Iku Ohama,  Issei Sato,  Takuya Kida,  Hiroki Arimura_

The edge partition model (EPM) is a fundamental Bayesian nonparametric model for extracting an overlapping structure from binary matrix. The EPM adopts a gamma process ($\Gamma$P) prior to automatically shrink the number of active atoms. However, we empirically found that the model shrinkage of the EPM does not typically work appropriately and leads to an overfitted solution. An analysis of the expectation of the EPM's intensity function suggested that the gamma priors for the EPM hyperparameters disturb the model shrinkage effect of the internal $\Gamma$P. In order to ensure that the model shrinkage effect of the EPM works in an appropriate manner, we proposed two novel generative constructions of the EPM: CEPM incorporating constrained gamma priors, and DEPM incorporating Dirichlet priors instead of the gamma priors. Furthermore, all DEPM's model parameters including the infinite atoms of the $\Gamma$P prior could be marginalized out, and thus it was possible to derive a truly infinite DEPM (IDEPM) that can be efficiently inferred using a collapsed Gibbs sampler. We experimentally confirmed that the model shrinkage of the proposed models works well and that the IDEPM indicated state-of-the-art performance in generalization ability, link prediction accuracy, mixing efficiency, and convergence speed.

### #40: Pose Guided Person Image Generation
_Liqian Ma,  Qianru Sun,  Xu Jia,  Bernt Schiele,  Tinne Tuytelaars,  Luc Van Gool_

This paper proposes the novel Pose Guided Person Generation Network (PG$^2$) that allows to synthesize person images in arbitrary poses, based on an image of that person and a novel pose. Our generation framework PG$^2$ utilizes the pose information explicitly and consists of two key stages: pose integration and image refinement. In the first stage the condition image and the target pose are fed into a U-Net-like network to generate an initial but coarse image of the person with the target pose. The second stage then refines the initial and blurry result by training a U-Net-like generator in an adversarial way. Extensive experimental results on both 128$\times$64 re-identification images and 256$\times$256 fashion photos show that our model generates high-quality person images with convincing details.
[Abstract](https://arxiv.org/abs/1705.09368), [PDF](https://arxiv.org/pdf/1705.09368)


### #41: Inference in Graphical Models via Semidefinite Programming Hierarchies 
_Murat A. Erdogdu,  Yash Deshpande,  Andrea Montanari_

Maximum A posteriori Probability (MAP) inference in graphical models amounts to solving a graph-structured combinatorial optimization problem. Popular inference algorithms such as belief propagation (BP) and generalized belief propagation (GBP) are intimately related to linear programming (LP) relaxation within the Sherali-Adams hierarchy. Despite the popularity of these algorithms, it is well understood that the Sum-of-Squares (SOS) hierarchy based on semidefinite programming (SDP) can provide superior guarantees. Unfortunately, SOS relaxations for a graph with $n$ vertices require solving an SDP with $n^{\Theta(d)}$ variables where $d$ is the degree in the hierarchy. In practice, for $d\ge 4$, this approach does not scale beyond a few tens of variables. In this paper, we propose binary SDP relaxations for MAP inference using the SOS hierarchy with two innovations focused on computational efficiency. Firstly, in analogy to BP and its variants, we only introduce decision variables corresponding to contiguous regions in the graphical model. Secondly, we solve the resulting SDP using a non-convex Burer-Monteiro style method, and develop a sequential rounding procedure. We demonstrate that the resulting algorithm can solve problems with tens of thousands of variables within minutes, and outperforms BP and GBP on practical problems such as image denoising and Ising spin glasses. Finally, for specific graph types, we establish a sufficient condition for the tightness of the proposed partial SOS relaxation.

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
_Yu-Chuan Su,  Kristen Grauman_

While 360{\deg} cameras offer tremendous new possibilities in vision, graphics, and augmented reality, the spherical images they produce make core feature extraction non-trivial. Convolutional neural networks (CNNs) trained on images from perspective cameras yield "flat" filters, yet 360{\deg} images cannot be projected to a single plane without significant distortion. A naive solution that repeatedly projects the viewing sphere to all tangent planes is accurate, but much too computationally intensive for real problems. We propose to learn a spherical convolutional network that translates a planar CNN to process 360{\deg} imagery directly in its equirectangular projection. Our approach learns to reproduce the flat filter outputs on 360{\deg} data, sensitive to the varying distortion effects across the viewing sphere. The key benefits are 1) efficient feature extraction for 360{\deg} images and video, and 2) the ability to leverage powerful pre-trained networks researchers have carefully honed (together with massive labeled image training sets) for perspective images. We validate our approach compared to several alternative methods in terms of both raw CNN output accuracy as well as applying a state-of-the-art "flat" object detector to 360{\deg} data. Our method yields the most accurate results while saving orders of magnitude in computation versus the existing exact reprojection solution.
[Abstract](https://arxiv.org/abs/1708.00919), [PDF](https://arxiv.org/pdf/1708.00919)


### #53: 3D Shape Reconstruction by Modeling 2.5D Sketch

### #54: Multimodal Learning and Reasoning for Visual Question Answering

### #55: Adversarial Surrogate Losses for Ordinal Regression

### #56: Hypothesis Transfer Learning via Transformation Functions
_Simon Shaolei Du,  Jayanth Koushik,  Aarti Singh,  Barnabas Poczos_

We consider the Hypothesis Transfer Learning (HTL) problem where one incorporates a hypothesis trained on the source domain into the learning procedure of the target domain. Existing theoretical analysis either only studies specific algorithms or only presents upper bounds on the generalization error but not on the excess risk. In this paper, we propose a unified algorithm-dependent framework for HTL through a novel notion of transformation function, which characterizes the relation between the source and the target domains. We conduct a general risk analysis of this framework and in particular, we show for the first time, if two domains are related, HTL enjoys faster convergence rates of excess risks for Kernel Smoothing and Kernel Ridge Regression than those of the classical non-transfer learning settings. Experiments on real world data demonstrate the effectiveness of our framework.
[Abstract](https://www.arxiv.org/abs/1612.01020?context=stat), [PDF](https://www.arxiv.org/pdf/1612.01020?context=stat)


### #57: Adversarial Invariant Feature Learning

### #58: Convergence Analysis of Two-layer Neural Networks with ReLU Activation
_Yuanzhi Li,  Yang Yuan_

In recent years, stochastic gradient descent (SGD) based techniques has become the standard tools for training neural networks. However, formal theoretical understanding of why SGD can train neural networks in practice is largely missing. In this paper, we make progress on understanding this mystery by providing a convergence analysis for SGD on a rich subset of two-layer feedforward networks with ReLU activations. This subset is characterized by a special structure called "identity mapping". We prove that, if input follows from Gaussian distribution, with standard $O(1/\sqrt{d})$ initialization of the weights, SGD converges to the global minimum in polynomial number of steps. Unlike normal vanilla networks, the "identity mapping" makes our network asymmetric and thus the global minimum is unique. To complement our theory, we are also able to show experimentally that multi-layer networks with this mapping have better performance compared with normal vanilla networks. Our convergence theorem differs from traditional non-convex optimization techniques. We show that SGD converges to optimal in "two phases": In phase I, the gradient points to the wrong direction, however, a potential function $g$ gradually decreases. Then in phase II, SGD enters a nice one point convex region and converges. We also show that the identity mapping is necessary for convergence, as it moves the initial point to a better place for optimization. Experiment verifies our claims.
[Abstract](https://arxiv.org/abs/1705.09886), [PDF](https://arxiv.org/pdf/1705.09886)


### #59: Doubly Accelerated Stochastic Variance Reduced Dual Averaging Method for Regularized Empirical Risk Minimization
_Tomoya Murata,  Taiji Suzuki_

In this paper, we develop a new accelerated stochastic gradient method for efficiently solving the convex regularized empirical risk minimization problem in mini-batch settings. The use of mini-batches is becoming a golden standard in the machine learning community, because mini-batch settings stabilize the gradient estimate and can easily make good use of parallel computing. The core of our proposed method is the incorporation of our new "double acceleration" technique and variance reduction technique. We theoretically analyze our proposed method and show that our method much improves the mini-batch efficiencies of previous accelerated stochastic methods, and essentially only needs size $\sqrt{n}$ mini-batches for achieving the optimal iteration complexities for both non-strongly and strongly convex objectives, where $n$ is the training set size. Further, we show that even in non-mini-batch settings, our method achieves the best known convergence rate for both non-strongly and strongly convex objectives.
[Abstract](https://arxiv.org/abs/1703.00439), [PDF](https://arxiv.org/pdf/1703.00439)


### #60: Langevin Dynamics with Continuous Tempering for Training Deep Neural Networks
_Nanyang Ye,  Zhanxing Zhu,  Rafal K. Mantiuk_

Minimizing non-convex and high-dimensional objective functions is challenging, especially when training modern deep neural networks. In this paper, a novel approach is proposed which divides the training process into two consecutive phases to obtain better generalization performance: Bayesian sampling and stochastic optimization. The first phase is to explore the energy landscape and to capture the "fat" modes; and the second one is to fine-tune the parameter learned from the first phase. In the Bayesian learning phase, we apply continuous tempering and stochastic approximation into the Langevin dynamics to create an efficient and effective sampler, in which the temperature is adjusted automatically according to the designed "temperature dynamics". These strategies can overcome the challenge of early trapping into bad local minima and have achieved remarkable improvements in various types of neural networks as shown in our theoretical analysis and empirical experiments.
[Abstract](https://arxiv.org/abs/1703.04379), [PDF](https://arxiv.org/pdf/1703.04379)


### #61: Efficient Online Linear Optimization with Approximation Algorithms
_Dan Garber_

We revisit the problem of \textit{online linear optimization} in case the set of feasible actions is accessible through an approximated linear optimization oracle with a factor $\alpha$ multiplicative approximation guarantee. This setting is in particular interesting since it captures natural online extensions of well-studied \textit{offline} linear optimization problems which are NP-hard, yet admit efficient approximation algorithms. The goal here is to minimize the $\alpha$\textit{-regret} which is the natural extension of the standard \textit{regret} in \textit{online learning} to this setting. We present new algorithms with significantly improved oracle complexity for both the full information and bandit variants of the problem. Mainly, for both variants, we present $\alpha$-regret bounds of $O(T^{-1/3})$, were $T$ is the number of prediction rounds, using only $O(\log{T})$ calls to the approximation oracle per iteration, on average. These are the first results to obtain both average oracle complexity of $O(\log{T})$ (or even poly-logarithmic in $T$) and $\alpha$-regret bound $O(T^{-c})$ for a constant $c>0$, for both variants.
[Abstract](https://arxiv.org/abs/1709.03093), [PDF](https://arxiv.org/pdf/1709.03093)


### #62: Geometric Descent Method for Convex Composite Minimization

### #63: Diffusion Approximations for Online Principal Component Estimation and Global Convergence

### #64:  Avoiding Discrimination through Causal Reasoning

### #65: Nonparametric Online Regression while Learning the Metric
_Ilja Kuzborskij,  Nicolò Cesa-Bianchi_

We study algorithms for online nonparametric regression that learn the directions along which the regression function is smoother. Our algorithm learns the Mahalanobis metric based on the gradient outer product matrix $\boldsymbol{G}$ of the regression function (automatically adapting to the effective rank of this matrix), while simultaneously bounding the regret ---on the same data sequence--- in terms of the spectrum of $\boldsymbol{G}$. As a preliminary step in our analysis, we generalize a nonparametric online learning algorithm by Hazan and Megiddo by enabling it to compete against functions whose Lipschitzness is measured with respect to an arbitrary Mahalanobis metric.
[Abstract](https://arxiv.org/abs/1705.07853), [PDF](https://arxiv.org/pdf/1705.07853)


### #66: Recycling for Fairness: Learning with Conditional Distribution Matching Constraints

### #67: Safe and Nested Subgame Solving for Imperfect-Information Games
_Noam Brown,  Tuomas Sandholm_

Unlike perfect-information games, imperfect-information games cannot be solved by decomposing the game into subgames that are solved independently. Instead, all decisions must consider the strategy of the game as a whole. While it is not possible to solve an imperfect-information game exactly through decomposition, it is possible to approximate solutions, or improve existing strategies, by solving disjoint subgames. This process is referred to as subgame solving. We introduce subgame solving techniques that outperform prior methods both in theory and practice. We also show how to adapt them, and past subgame solving techniques, to respond to opponent actions that are outside the original action abstraction; this significantly outperforms the prior state-of-the-art approach, action translation. Finally, we show that subgame solving can be repeated as the game progresses down the tree, leading to lower exploitability. Subgame solving is a key component of Libratus, the first AI to defeat top humans in heads-up no-limit Texas hold'em poker.
[Abstract](https://arxiv.org/abs/1705.02955), [PDF](https://arxiv.org/pdf/1705.02955)


### #68: Unsupervised Image-to-Image Translation Networks
_Ming-Yu Liu,  Thomas Breuel,  Jan Kautz_

Most of the existing image-to-image translation frameworks---mapping an image in one domain to a corresponding image in another---are based on supervised learning, i.e., pairs of corresponding images in two domains are required for learning the translation function. This largely limits their applications, because capturing corresponding images in two different domains is often a difficult task. To address the issue, we propose the UNsupervised Image-to-image Translation (UNIT) framework, which is based on variational autoencoders and generative adversarial networks. The proposed framework can learn the translation function without any corresponding images in two domains. We enable this learning capability by combining a weight-sharing constraint and an adversarial training objective. Through visualization results from various unsupervised image translation tasks, we verify the effectiveness of the proposed framework. An ablation study further reveals the critical design choices. Moreover, we apply the UNIT framework to the unsupervised domain adaptation task and achieve better results than competing algorithms do in benchmark datasets.
[Abstract](https://arxiv.org/abs/1703.00848), [PDF](https://arxiv.org/pdf/1703.00848)


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
_Wei Shen,  Kai Zhao,  Yilu Guo,  Alan Yuille_

Label distribution learning (LDL) is a general learning framework, which assigns to an instance a distribution over a set of labels rather than a single label or multiple labels. Current LDL methods have either restricted assumptions on the expression form of the label distribution or limitations in representation learning, e.g., to learn deep features in an end-to-end manner. This paper presents label distribution learning forests (LDLFs) - a novel label distribution learning algorithm based on differentiable decision trees, which have several advantages: 1) Decision trees have the potential to model any general form of label distributions by a mixture of leaf node predictions. 2) The learning of differentiable decision trees can be combined with representation learning. We define a distribution-based loss function for a forest, enabling all the trees to be learned jointly, and show that an update function for leaf node predictions, which guarantees a strict decrease of the loss function, can be derived by variational bounding. The effectiveness of the proposed LDLFs is verified on several LDL tasks and a computer vision application, showing significant improvements to the state-of-the-art LDL methods.
[Abstract](https://arxiv.org/abs/1702.06086), [PDF](https://arxiv.org/pdf/1702.06086)


### #82: Unsupervised object learning from dense equivariant image labelling
_James Thewlis,  Hakan Bilen,  Andrea Vedaldi_

One of the key challenges of visual perception is to extract abstract models of 3D objects and object categories from visual measurements, which are affected by complex nuisance factors such as viewpoint, occlusion, motion, and deformations. Starting from the recent idea of viewpoint factorization, we propose a new approach that, given a large number of images of an object and no other supervision, can extract a dense object-centric coordinate frame. This coordinate frame is invariant to deformations of the images and comes with a dense equivariant labelling neural network that can map image pixels to their corresponding object coordinates. We demonstrate the applicability of this method to simple articulated objects and deformable objects such as human faces, learning embeddings from random synthetic transformations or optical flow correspondences, all without any manual supervision.
[Abstract](https://arxiv.org/abs/1706.02932), [PDF](https://arxiv.org/pdf/1706.02932)


### #83: Compression-aware Training of Deep Neural Networks

### #84: Multiscale Semi-Markov Dynamics for Intracortical Brain-Computer Interfaces

### #85: PredRNN: Recurrent Neural Networks for Video Prediction using Spatiotemporal LSTMs

### #86: Detrended Partial Cross Correlation for Brain Connectivity Analysis

### #87: Contrastive Learning for Image Captioning
_Bo Dai,  Dahua Lin_

Image captioning, a popular topic in computer vision, has achieved substantial progress in recent years. However, the distinctiveness of natural descriptions is often overlooked in previous work. It is closely related to the quality of captions, as distinctive captions are more likely to describe images with their unique aspects. In this work, we propose a new learning method, Contrastive Learning (CL), for image captioning. Specifically, via two constraints formulated on top of a reference model, the proposed method can encourage distinctiveness, while maintaining the overall quality of the generated captions. We tested our method on two challenging datasets, where it improves the baseline model by significant margins. We also showed in our studies that the proposed method is generic and can be used for models with various structures.

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
_Arun Venkatraman,  Nicholas Rhinehart,  Wen Sun,  Lerrel Pinto,  Martial Hebert,  Byron Boots,  Kris M. Kitani,  J. Andrew Bagnell_

Recurrent neural networks (RNNs) are a vital modeling technique that rely on internal states learned indirectly by optimization of a supervised, unsupervised, or reinforcement training loss. RNNs are used to model dynamic processes that are characterized by underlying latent states whose form is often unknown, precluding its analytic representation inside an RNN. In the Predictive-State Representation (PSR) literature, latent state processes are modeled by an internal state representation that directly models the distribution of future observations, and most recent work in this area has relied on explicitly representing and targeting sufficient statistics of this probability distribution. We seek to combine the advantages of RNNs and PSRs by augmenting existing state-of-the-art recurrent neural networks with Predictive-State Decoders (PSDs), which add supervision to the network's internal state representation to target predicting future observations. Predictive-State Decoders are simple to implement and easily incorporated into existing training pipelines via additional loss regularization. We demonstrate the effectiveness of PSDs with experimental results in three different domains: probabilistic filtering, Imitation Learning, and Reinforcement Learning. In each, our method improves statistical performance of state-of-the-art recurrent baselines and does so with fewer iterations and less data.

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
_Mohammad Haris Baig,  Vladlen Koltun,  Lorenzo Torresani_

We study the design of deep architectures for lossy image compression. We present two architectural recipes in the context of multi-stage progressive encoders and empirically demonstrate their importance on compression performance. Specifically, we show that: (a) predicting the original image data from residuals in a multi-stage progressive architecture facilitates learning and leads to improved performance at approximating the original content and (b) learning to inpaint (from neighboring image pixels) before performing compression reduces the amount of information that must be stored to achieve a high-quality approximation. Incorporating these design choices in a baseline progressive encoder yields an average reduction of over $60\%$ in file size with similar quality compared to the original residual encoder.

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
_Wei Wen,  Cong Xu,  Feng Yan,  Chunpeng Wu,  Yandan Wang,  Yiran Chen,  Hai Li_

High network communication cost for synchronizing gradients and parameters is the well-known bottleneck of distributed training. In this work, we propose TernGrad that uses ternary gradients to accelerate distributed deep learning in data parallelism. Our approach requires only three numerical levels {-1,0,1} which can aggressively reduce the communication time. We mathematically prove the convergence of TernGrad under the assumption of a bound on gradients. Guided by the bound, we propose layer-wise ternarizing and gradient clipping to improve its convergence. Our experiments show that applying TernGrad on AlexNet does not incur any accuracy loss and can even improve accuracy. The accuracy loss of GoogLeNet induced by TernGrad is less than 2% on average. Finally, a performance model is proposed to study the scalability of TernGrad. Experiments show significant speed gains for various deep neural networks.
[Abstract](https://arxiv.org/abs/1705.07878), [PDF](https://arxiv.org/pdf/1705.07878)


### #146: Learning Affinity via Spatial Propagation Networks
_Sifei Liu,  Shalini De Mello,  Jinwei Gu,  Guangyu Zhong,  Ming-Hsuan Yang,  Jan Kautz_

In this paper, we propose spatial propagation networks for learning the affinity matrix for vision tasks. We show that by constructing a row/column linear propagation model, the spatially varying transformation matrix exactly constitutes an affinity matrix that models dense, global pairwise relationships of an image. Specifically, we develop a three-way connection for the linear propagation model, which (a) formulates a sparse transformation matrix, where all elements can be the output from a deep CNN, but (b) results in a dense affinity matrix that effectively models any task-specific pairwise similarity matrix. Instead of designing the similarity kernels according to image features of two points, we can directly output all the similarities in a purely data-driven manner. The spatial propagation network is a generic framework that can be applied to many affinity-related tasks, including but not limited to image matting, segmentation and colorization, to name a few. Essentially, the model can learn semantically-aware affinity values for high-level vision tasks due to the powerful learning capability of the deep neural network classifier. We validate the framework on the task of refinement for image segmentation boundaries. Experiments on the HELEN face parsing and PASCAL VOC-2012 semantic segmentation tasks show that the spatial propagation network provides a general, effective and efficient solution for generating high-quality segmentation results.

### #147: Linear regression without correspondence
_Daniel Hsu,  Kevin Shi,  Xiaorui Sun_

This article considers algorithmic and statistical aspects of linear regression when the correspondence between the covariates and the responses is unknown. First, a fully polynomial-time approximation scheme is given for the natural least squares optimization problem in any constant dimension. Next, in an average-case and noise-free setting where the responses exactly correspond to a linear function of i.i.d. draws from a standard multivariate normal distribution, an efficient algorithm based on lattice basis reduction is shown to exactly recover the unknown linear function in arbitrary dimension. Finally, lower bounds on the signal-to-noise ratio are established for approximate recovery of the unknown linear function by any estimator.
[Abstract](https://arxiv.org/abs/1705.07048), [PDF](https://arxiv.org/pdf/1705.07048)


### #148: NeuralFDR: Learning Discovery Thresholds from Hypothesis Features

### #149: Cost efficient gradient boosting

### #150: Probabilistic Rule Realization and Selection
_Haizi Yu,  Tianxi Li,  Lav R. Varshney_

Abstraction and realization are bilateral processes that are key in deriving intelligence and creativity. In many domains, the two processes are approached through rules: high-level principles that reveal invariances within similar yet diverse examples. Under a probabilistic setting for discrete input spaces, we focus on the rule realization problem which generates input sample distributions that follow the given rules. More ambitiously, we go beyond a mechanical realization that takes whatever is given, but instead ask for proactively selecting reasonable rules to realize. This goal is demanding in practice, since the initial rule set may not always be consistent and thus intelligent compromises are needed. We formulate both rule realization and selection as two strongly connected components within a single and symmetric bi-convex problem, and derive an efficient algorithm that works at large scale. Taking music compositional rules as the main example throughout the paper, we demonstrate our model's efficiency in not only music realization (composition) but also music interpretation and understanding (analysis).
[Abstract](https://arxiv.org/abs/1709.01674), [PDF](https://arxiv.org/pdf/1709.01674)


### #151: Nearest-Neighbor Sample Compression: Efficiency, Consistency, Infinite Dimensions
_Aryeh Kontorovich,  Sivan Sabato,  Roi Weiss_

We examine the Bayes-consistency of a recently proposed 1-nearest-neighbor-based multiclass learning algorithm. This algorithm is derived from sample compression bounds and enjoys the statistical advantages of tight, fully empirical generalization bounds, as well as the algorithmic advantages of runtime and memory savings. We prove that this algorithm is strongly Bayes-consistent in metric spaces with finite doubling dimension --- the first consistency result for an efficient nearest-neighbor sample compression scheme. Rather surprisingly, we discover that this algorithm continues to be Bayes-consistent even in a certain infinite-dimensional setting, in which the basic measure-theoretic conditions on which classic consistency proofs hinge are violated. This is all the more surprising, since it is known that k-NN is not Bayes-consistent in this setting. We pose several challenging open problems for future research.
[Abstract](https://arxiv.org/abs/1705.08184), [PDF](https://arxiv.org/pdf/1705.08184)


### #152: A Scale Free Algorithm for Stochastic Bandits with Bounded Kurtosis
_Tor Lattimore_

Existing strategies for finite-armed stochastic bandits mostly depend on a parameter of scale that must be known in advance. Sometimes this is in the form of a bound on the payoffs, or the knowledge of a variance or subgaussian parameter. The notable exceptions are the analysis of Gaussian bandits with unknown mean and variance by Cowan and Katehakis [2015] and of uniform distributions with unknown support [Cowan and Katehakis, 2015]. The results derived in these specialised cases are generalised here to the non-parametric setup, where the learner knows only a bound on the kurtosis of the noise, which is a scale free measure of the extremity of outliers.
[Abstract](https://arxiv.org/abs/1703.08937), [PDF](https://arxiv.org/pdf/1703.08937)


### #153: Learning Multiple Tasks with Deep Relationship Networks
_Mingsheng Long,  Jianmin Wang,  Philip S. Yu_

Deep networks trained on large-scale data can learn transferable features to promote learning multiple tasks. As deep features eventually transition from general to specific along deep networks, a fundamental problem is how to exploit the relationship across different tasks and improve the feature transferability in the task-specific layers. In this paper, we propose Deep Relationship Networks (DRN) that discover the task relationship based on novel tensor normal priors over the parameter tensors of multiple task-specific layers in deep convolutional networks. By jointly learning transferable features and task relationships, DRN is able to alleviate the dilemma of negative-transfer in the feature layers and under-transfer in the classifier layer. Extensive experiments show that DRN yields state-of-the-art results on standard multi-task learning benchmarks.
[Abstract](https://arxiv.org/abs/1506.02117), [PDF](https://arxiv.org/pdf/1506.02117)


### #154: Deep Hyperalignment

### #155: Online to Offline Conversions and Adaptive Minibatch Sizes
_Kfir Y. Levy_

We present an approach towards convex optimization that relies on a novel scheme which converts online adaptive algorithms into offline methods. In the offline optimization setting, our derived methods are shown to obtain favourable adaptive guarantees which depend on the harmonic sum of the queried gradients. We further show that our methods implicitly adapt to the objective's structure: in the smooth case fast convergence rates are ensured without any prior knowledge of the smoothness parameter, while still maintaining guarantees in the non-smooth setting. Our approach has a natural extension to the stochastic setting, resulting in a lazy version of SGD (stochastic GD), where minibathces are chosen \emph{adaptively} depending on the magnitude of the gradients. Thus providing a principled approach towards choosing minibatch sizes.
[Abstract](https://arxiv.org/abs/1705.10499), [PDF](https://arxiv.org/pdf/1705.10499)


### #156: Stochastic Optimization with Variance Reduction for Infinite Datasets with Finite Sum Structure
_Alberto Bietti (Thoth, MSR - INRIA),  Julien Mairal (Thoth)_

Stochastic optimization algorithms with variance reduction have proven successful for minimizing large finite sums of functions. Unfortunately, these techniques are unable to deal with stochastic perturbations of input data, induced for example by data augmentation. In such cases, the objective is no longer a finite sum, and the main candidate for optimization is the stochastic gradient descent method (SGD). In this paper, we introduce a variance reduction approach for these settings when the objective is composite and strongly convex. The convergence rate outperforms SGD with a typically much smaller constant factor, which depends on the variance of gradient estimates only due to perturbations on a single example.
[Abstract](https://arxiv.org/abs/1610.00970), [PDF](https://arxiv.org/pdf/1610.00970)


### #157: Deep Learning with Topological Signatures
_Christoph Hofer,  Roland Kwitt,  Marc Niethammer,  Andreas Uhl_

Inferring topological and geometrical information from data can offer an alternative perspective on machine learning problems. Methods from topological data analysis, e.g., persistent homology, enable us to obtain such information, typically in the form of summary representations of topological features. However, such topological signatures often come with an unusual structure (e.g., multisets of intervals) that is highly impractical for most machine learning techniques. While many strategies have been proposed to map these topological signatures into machine learning compatible representations, they suffer from being agnostic to the target learning task. In contrast, we propose a technique that enables us to input topological signatures to deep neural networks and learn a task-optimal representation during training. Our approach is realized as a novel input layer with favorable theoretical properties. Classification experiments on 2D object shapes and social network graphs demonstrate the versatility of the approach and, in case of the latter, we even outperform the state-of-the-art by a large margin.
[Abstract](https://arxiv.org/abs/1707.04041), [PDF](https://arxiv.org/pdf/1707.04041)


### #158: Predicting User Activity Level In Point Process Models With Mass Transport Equation

### #159: Submultiplicative Glivenko-Cantelli and Uniform Convergence of Revenues
_Noga Alon,  Moshe Babaioff,  Yannai A. Gonczarowski,  Yishay Mansour,  Shay Moran,  Amir Yehudayoff_

In this work we derive a variant of the classic Glivenko-Cantelli Theorem, which asserts uniform convergence of the empirical Cumulative Distribution Function (CDF) to the CDF of the underlying distribution. Our variant allows for tighter convergence bounds for extreme values of the CDF. We apply our bound in the context of revenue learning, which is a well-studied problem in economics and algorithmic game theory. We derive sample-complexity bounds on the uniform convergence rate of the empirical revenues to the true revenues, assuming a bound on the $k$th moment of the valuations, for any (possibly fractional) $k>1$. For uniform convergence in the limit, we give a complete characterization and a zero-one law: if the first moment of the valuations is finite, then uniform convergence almost surely occurs; conversely, if the first moment is infinite, then uniform convergence almost never occurs.
[Abstract](https://arxiv.org/abs/1705.08430), [PDF](https://arxiv.org/pdf/1705.08430)


### #160: Deep Dynamic Poisson Factorization Model

### #161: Positive-Unlabeled Learning with Non-Negative Risk Estimator
_Ryuichi Kiryo,  Gang Niu,  Marthinus C. du Plessis,  Masashi Sugiyama_

From only positive (P) and unlabeled (U) data, a binary classifier could be trained with PU learning. Unbiased PU learning that is based on unbiased risk estimators is now state of the art. However, if its model is very flexible, its empirical risk on training data will go negative, and we will suffer from overfitting seriously. In this paper, we propose a novel non-negative risk estimator for PU learning. When being minimized, it is more robust against overfitting, and thus we are able to train very flexible models given limited P data. Moreover, we analyze the bias, consistency and mean-squared-error reduction of the proposed risk estimator as well as the estimation error of the corresponding risk minimizer. Experiments show that the non-negative risk estimator outperforms unbiased counterparts when they disagree.
[Abstract](https://arxiv.org/abs/1703.00593), [PDF](https://arxiv.org/pdf/1703.00593)


### #162: Optimal Sample Complexity of M-wise Data for Top-K Ranking

### #163: What-If Reasoning using Counterfactual Gaussian Processes

### #164: Communication-Efficient Stochastic Gradient Descent, with Applications to Neural Networks

### #165: On the Convergence of Block Coordinate Descent in Training DNNs with Tikhonov Regularization

### #166: Train longer, generalize better: closing the generalization gap in large batch training of neural networks
_Elad Hoffer,  Itay Hubara,  Daniel Soudry_

Background: Deep learning models are typically trained using stochastic gradient descent or one of its variants. These methods update the weights using their gradient, estimated from a small fraction of the training data. It has been observed that when using large batch sizes there is a persistent degradation in generalization performance - known as the "generalization gap" phenomena. Identifying the origin of this gap and closing it had remained an open problem. Contributions: We examine the initial high learning rate training phase. We find that the weight distance from its initialization grows logarithmically with the number of weight updates. We therefore propose a "random walk on random landscape" statistical model which is known to exhibit similar "ultra-slow" diffusion behavior. Following this hypothesis we conducted experiments to show empirically that the "generalization gap" stems from the relatively small number of updates rather than the batch size, and can be completely eliminated by adapting the training regime used. We further investigate different techniques to train models in the large-batch regime and present a novel algorithm named "Ghost Batch Normalization" which enables significant decrease in the generalization gap without increasing the number of updates. To validate our findings we conduct several additional experiments on MNIST, CIFAR-10, CIFAR-100 and ImageNet. Finally, we reassess common practices and beliefs concerning training of deep models and suggest they may not be optimal to achieve good generalization.
[Abstract](https://arxiv.org/abs/1705.08741), [PDF](https://arxiv.org/pdf/1705.08741)


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
_Lars Mescheder,  Sebastian Nowozin,  Andreas Geiger_

In this paper, we analyze the numerics of common algorithms for training Generative Adversarial Networks (GANs). Using the formalism of smooth two-player games we analyze the associated gradient vector field of GAN training objectives. Our findings suggest that the convergence of current algorithms suffers due to two factors: i) presence of eigenvalues of the Jacobian of the gradient vector field with zero real-part, and ii) eigenvalues with big imaginary part. Using these findings, we design a new algorithm that overcomes some of these limitations and has better convergence properties. Experimentally, we demonstrate its superiority on training common GAN architectures and show convergence on GAN architectures that are known to be notoriously hard to train.

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
_Rowan McAllister,  Carl Edward Rasmussen_

We present a data-efficient reinforcement learning algorithm resistant to observation noise. Our method extends the highly data-efficient PILCO algorithm (Deisenroth & Rasmussen, 2011) into partially observed Markov decision processes (POMDPs) by considering the filtering process during policy evaluation. PILCO conducts policy search, evaluating each policy by first predicting an analytic distribution of possible system trajectories. We additionally predict trajectories w.r.t. a filtering process, achieving significantly higher performance than combining a filter with a policy optimised by the original (unfiltered) framework. Our test setup is the cartpole swing-up task with sensor noise, which involves nonlinear dynamics and requires nonlinear control.

### #196: Compatible Reward Inverse Reinforcement Learning

### #197: First-Order Adaptive Sample Size Methods to Reduce Complexity of Empirical Risk Minimization

### #198: Hiding Images in Plain Sight: Deep Steganography

### #199: Neural Program Meta-Induction
_Jacob Devlin,  Rudy Bunel,  Rishabh Singh,  Matthew Hausknecht,  Pushmeet Kohli_

Most recently proposed methods for Neural Program Induction work under the assumption of having a large set of input/output (I/O) examples for learning any underlying input-output mapping. This paper aims to address the problem of data and computation efficiency of program induction by leveraging information from related tasks. Specifically, we propose two approaches for cross-task knowledge transfer to improve program induction in limited-data scenarios. In our first proposal, portfolio adaptation, a set of induction models is pretrained on a set of related tasks, and the best model is adapted towards the new task using transfer learning. In our second approach, meta program induction, a $k$-shot learning approach is used to make a model generalize to new tasks without additional training. To test the efficacy of our methods, we constructed a new benchmark of programs written in the Karel programming language. Using an extensive experimental evaluation on the Karel benchmark, we demonstrate that our proposals dramatically outperform the baseline induction method that does not use knowledge transfer. We also analyze the relative performance of the two approaches and study conditions in which they perform best. In particular, meta induction outperforms all existing approaches under extreme data sparsity (when a very small number of examples are available), i.e., fewer than ten. As the number of available I/O examples increase (i.e. a thousand or more), portfolio adapted program induction becomes the best approach. For intermediate data sizes, we demonstrate that the combined method of adapted meta program induction has the strongest performance.

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
_Kinjal Basu,  Ankan Saha,  Shaunak Chatterjee_

We consider the problem of solving a large-scale Quadratically Constrained Quadratic Program. Such problems occur naturally in many scientific and web applications. Although there are efficient methods which tackle this problem, they are mostly not scalable. In this paper, we develop a method that transforms the quadratic constraint into a linear form by sampling a set of low-discrepancy points. The transformed problem can then be solved by applying any state-of-the-art large-scale quadratic programming solvers. We show the convergence of our approximate solution to the true solution as well as some finite sample error bounds. Experimental results are also shown to prove scalability as well as improved quality of approximation in practice.

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
_Jeffrey Regier,  Michael I. Jordan,  Jon McAuliffe_

We introduce TrustVI, a fast second-order algorithm for black-box variational inference based on trust-region optimization and the reparameterization trick. At each iteration, TrustVI proposes and assesses a step based on minibatches of draws from the variational distribution. The algorithm provably converges to a stationary point. We implement TrustVI in the Stan framework and compare it to ADVI. TrustVI typically converges in tens of iterations to a solution at least as good as the one that ADVI reaches in thousands of iterations. TrustVI iterations can be more computationally expensive, but total computation is typically an order of magnitude less in our experiments.
[Abstract](https://arxiv.org/abs/1706.02375), [PDF](https://arxiv.org/pdf/1706.02375)


### #231: Scalable Demand-Aware Recommendation

### #232: SGD Learns the Conjugate Kernel Class of the Network
_Amit Daniely_

We show that the standard stochastic gradient decent (SGD) algorithm is guaranteed to learn, in polynomial time, a function that is competitive with the best function in the conjugate kernel space of the network, as defined in Daniely, Frostig and Singer. The result holds for log-depth networks from a rich family of architectures. To the best of our knowledge, it is the first polynomial-time guarantee for the standard neural network learning algorithm for networks of depth more that two. As corollaries, it follows that for neural networks of any depth between $2$ and $\log(n)$, SGD is guaranteed to learn, in polynomial time, constant degree polynomials with polynomially bounded coefficients. Likewise, it follows that SGD on large enough networks can learn any continuous function (not in polynomial time), complementing classical expressivity results.
[Abstract](https://arxiv.org/abs/1702.08503), [PDF](https://arxiv.org/pdf/1702.08503)


### #233: Noise-Tolerant Interactive Learning Using Pairwise Comparisons

### #234: Analyzing Hidden Representations in End-to-End Automatic Speech Recognition Systems
_Yonatan Belinkov,  James Glass_

Neural models have become ubiquitous in automatic speech recognition systems. While neural networks are typically used as acoustic models in more complex systems, recent studies have explored end-to-end speech recognition systems based on neural networks, which can be trained to directly predict text from input acoustic features. Although such systems are conceptually elegant and simpler than traditional systems, it is less obvious how to interpret the trained models. In this work, we analyze the speech representations learned by a deep end-to-end model that is based on convolutional and recurrent layers, and trained with a connectionist temporal classification (CTC) loss. We use a pre-trained model to generate frame-level features which are given to a classifier that is trained on frame classification into phones. We evaluate representations from different layers of the deep model and compare their quality for predicting phone labels. Our experiments shed light on important aspects of the end-to-end model such as layer depth, model complexity, and other design choices.
[Abstract](https://arxiv.org/abs/1709.04482), [PDF](https://arxiv.org/pdf/1709.04482)


### #235: Generative Local Metric Learning for Kernel Regression

### #236: Information Theoretic Properties of Markov Random Fields, and their Algorithmic Applications
_Linus Hamilton,  Frederic Koehler,  Ankur Moitra_

Markov random fields area popular model for high-dimensional probability distributions. Over the years, many mathematical, statistical and algorithmic problems on them have been studied. Until recently, the only known algorithms for provably learning them relied on exhaustive search, correlation decay or various incoherence assumptions. Bresler gave an algorithm for learning general Ising models on bounded degree graphs. His approach was based on a structural result about mutual information in Ising models. Here we take a more conceptual approach to proving lower bounds on the mutual information through setting up an appropriate zero-sum game. Our proof generalizes well beyond Ising models, to arbitrary Markov random fields with higher order interactions. As an application, we obtain algorithms for learning Markov random fields on bounded degree graphs on $n$ nodes with $r$-order interactions in $n^r$ time and $\log n$ sample complexity. The sample complexity is information theoretically optimal up to the dependence on the maximum degree. The running time is nearly optimal under standard conjectures about the hardness of learning parity with noise.
[Abstract](https://arxiv.org/abs/1705.11107), [PDF](https://arxiv.org/pdf/1705.11107)


### #237: Fitting Low-Rank Tensors in Constant Time

### #238: Deep supervised discrete hashing

### #239: Using Options and Covariance Testing for Long Horizon Off-Policy Policy Evaluation

### #240: How regularization affects the critical points in linear networks
_Amirhossein Taghvaei,  Jin W. Kim,  Prashant G. Mehta_

This paper is concerned with the problem of representing and learning a linear transformation using a linear neural network. In recent years, there has been a growing interest in the study of such networks in part due to the successes of deep learning. The main question of this body of research and also of this paper pertains to the existence and optimality properties of the critical points of the mean-squared loss function. The primary concern here is the robustness of the critical points with regularization of the loss function. An optimal control model is introduced for this purpose and a learning algorithm (regularized form of backprop) derived for the same using the Hamilton's formulation of optimal control. The formulation is used to provide a complete characterization of the critical points in terms of the solutions of a nonlinear matrix-valued equation, referred to as the characteristic equation. Analytical and numerical tools from bifurcation theory are used to compute the critical points via the solutions of the characteristic equation. The main conclusion is that the critical point diagram can be fundamentally different even with arbitrary small amounts of regularization.
[Abstract](http://arxiv.org/abs/1709.09625), [PDF](http://arxiv.org/pdf/1709.09625)


### #241: Fisher GAN
_Youssef Mroueh,  Tom Sercu_

Generative Adversarial Networks (GANs) are powerful models for learning complex distributions. Stable training of GANs has been addressed in many recent works which explore different metrics between distributions. In this paper we introduce Fisher GAN which fits within the Integral Probability Metrics (IPM) framework for training GANs. Fisher GAN defines a critic with a data dependent constraint on its second order moments. We show in this paper that Fisher GAN allows for stable and time efficient training that does not compromise the capacity of the critic, and does not need data independent constraints such as weight clipping. We analyze our Fisher IPM theoretically and provide an algorithm based on Augmented Lagrangian for Fisher GAN. We validate our claims on both image sample generation and semi-supervised classification using Fisher GAN.
[Abstract](https://arxiv.org/abs/1705.09675), [PDF](https://arxiv.org/pdf/1705.09675)


### #242: Information-theoretic analysis of generalization capability of learning algorithms
_Aolin Xu,  Maxim Raginsky_

We derive upper bounds on the generalization error of a learning algorithm in terms of the mutual information between its input and output. The upper bounds provide theoretical guidelines for striking the right balance between data fit and generalization by controlling the input-output mutual information of a learning algorithm. The results can also be used to analyze the generalization capability of learning algorithms under adaptive composition, and the bias-accuracy tradeoffs in adaptive data analytics. Our work extends and leads to nontrivial improvements on the recent results of Russo and Zou.
[Abstract](https://arxiv.org/abs/1705.07809), [PDF](https://arxiv.org/pdf/1705.07809)


### #243: Sparse Approximate Conic Hulls

### #244: Rigorous Dynamics and Consistent Estimation in Arbitrarily Conditioned Linear Systems 
_Alyson K. Fletcher,  Mojtaba Sahraee-Ardakan,  Philip Schniter,  Sundeep Rangan_

The problem of estimating a random vector x from noisy linear measurements y = A x + w with unknown parameters on the distributions of x and w, which must also be learned, arises in a wide range of statistical learning and linear inverse problems. We show that a computationally simple iterative message-passing algorithm can provably obtain asymptotically consistent estimates in a certain high-dimensional large-system limit (LSL) under very general parameterizations. Previous message passing techniques have required i.i.d. sub-Gaussian A matrices and often fail when the matrix is ill-conditioned. The proposed algorithm, called adaptive vector approximate message passing (Adaptive VAMP) with auto-tuning, applies to all right-rotationally random A. Importantly, this class includes matrices with arbitrarily poor conditioning. We show that the parameter estimates and mean squared error (MSE) of x in each iteration converge to deterministic limits that can be precisely predicted by a simple set of state evolution (SE) equations. In addition, a simple testable condition is provided in which the MSE matches the Bayes-optimal value predicted by the replica method. The paper thus provides a computationally simple method with provable guarantees of optimality and consistency over a large class of linear inverse problems.
[Abstract](https://arxiv.org/abs/1706.06054), [PDF](https://arxiv.org/pdf/1706.06054)


### #245: Toward Goal-Driven Neural Network Models for the Rodent Whisker-Trigeminal System
_Chengxu Zhuang,  Jonas Kubilius,  Mitra Hartmann,  Daniel Yamins_

In large part, rodents see the world through their whiskers, a powerful tactile sense enabled by a series of brain areas that form the whisker-trigeminal system. Raw sensory data arrives in the form of mechanical input to the exquisitely sensitive, actively-controllable whisker array, and is processed through a sequence of neural circuits, eventually arriving in cortical regions that communicate with decision-making and memory areas. Although a long history of experimental studies has characterized many aspects of these processing stages, the computational operations of the whisker-trigeminal system remain largely unknown. In the present work, we take a goal-driven deep neural network (DNN) approach to modeling these computations. First, we construct a biophysically-realistic model of the rat whisker array. We then generate a large dataset of whisker sweeps across a wide variety of 3D objects in highly-varying poses, angles, and speeds. Next, we train DNNs from several distinct architectural families to solve a shape recognition task in this dataset. Each architectural family represents a structurally-distinct hypothesis for processing in the whisker-trigeminal system, corresponding to different ways in which spatial and temporal information can be integrated. We find that most networks perform poorly on the challenging shape recognition task, but that specific architectures from several families can achieve reasonable performance levels. Finally, we show that Representational Dissimilarity Matrices (RDMs), a tool for comparing population codes between neural systems, can separate these higher-performing networks with data of a type that could plausibly be collected in a neurophysiological or imaging experiment. Our results are a proof-of-concept that goal-driven DNN networks of the whisker-trigeminal system are potentially within reach.
[Abstract](https://arxiv.org/abs/1706.07555), [PDF](https://arxiv.org/pdf/1706.07555)


### #246: Accuracy First: Selecting a Differential Privacy Level for Accuracy Constrained ERM
_Katrina Ligett,  Seth Neel,  Aaron Roth,  Bo Waggoner,  Z. Steven Wu_

Traditional approaches to differential privacy assume a fixed privacy requirement $\epsilon$ for a computation, and attempt to maximize the accuracy of the computation subject to the privacy constraint. As differential privacy is increasingly deployed in practical settings, it may often be that there is instead a fixed accuracy requirement for a given computation and the data analyst would like to maximize the privacy of the computation subject to the accuracy constraint. This raises the question of how to find and run a maximally private empirical risk minimizer subject to a given accuracy requirement. We propose a general "noise reduction" framework that can apply to a variety of private empirical risk minimization (ERM) algorithms, using them to "search" the space of privacy levels to find the empirically strongest one that meets the accuracy constraint, incurring only logarithmic overhead in the number of privacy levels searched. The privacy analysis of our algorithm leads naturally to a version of differential privacy where the privacy parameters are dependent on the data, which we term ex-post privacy, and which is related to the recently introduced notion of privacy odometers. We also give an ex-post privacy analysis of the classical AboveThreshold privacy tool, modifying it to allow for queries chosen depending on the database. Finally, we apply our approach to two common objectives, regularized linear and logistic regression, and empirically compare our noise reduction methods to (i) inverting the theoretical utility guarantees of standard private ERM algorithms and (ii) a stronger, empirical baseline based on binary search.
[Abstract](https://arxiv.org/abs/1705.10829), [PDF](https://arxiv.org/pdf/1705.10829)


### #247: EX2: Exploration with Exemplar Models for Deep Reinforcement Learning
_Justin Fu,  John D. Co-Reyes,  Sergey Levine_

Deep reinforcement learning algorithms have been shown to learn complex tasks using highly general policy classes. However, sparse reward problems remain a significant challenge. Exploration methods based on novelty detection have been particularly successful in such settings but typically require generative or predictive models of the observations, which can be difficult to train when the observations are very high-dimensional and complex, as in the case of raw images. We propose a novelty detection algorithm for exploration that is based entirely on discriminatively trained exemplar models, where classifiers are trained to discriminate each visited state against all others. Intuitively, novel states are easier to distinguish against other states seen during training. We show that this kind of discriminative modeling corresponds to implicit density estimation, and that it can be combined with count-based exploration to produce competitive results on a range of popular benchmark tasks, including state-of-the-art results on challenging egocentric observations in the vizDoom benchmark.
[Abstract](https://arxiv.org/abs/1703.01260), [PDF](https://arxiv.org/pdf/1703.01260)


### #248: Multitask Spectral Learning of Weighted Automata

### #249: Multi-way Interacting Regression via Factorization Machines
_Mikhail Yurochkin,  XuanLong Nguyen,  Nikolaos Vasiloglou_

We propose a Bayesian regression method that accounts for multi-way interactions of arbitrary orders among the predictor variables. Our model makes use of a factorization mechanism for representing the regression coefficients of interactions among the predictors, while the interaction selection is guided by a prior distribution on random hypergraphs, a construction which generalizes the Finite Feature Model. We present a posterior inference algorithm based on Gibbs sampling, and establish posterior consistency of our regression model. Our method is evaluated with extensive experiments on simulated data and demonstrated to be able to identify meaningful interactions in applications in genetics and retail demand forecasting.
[Abstract](https://arxiv.org/abs/1709.09301), [PDF](https://arxiv.org/pdf/1709.09301)


### #250: Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network
_Wengong Jin,  Connor W. Coley,  Regina Barzilay,  Tommi Jaakkola_

The prediction of organic reaction outcomes is a fundamental problem in computational chemistry. Since a reaction may involve hundreds of atoms, fully exploring the space of possible transformations is intractable. The current solution utilizes reaction templates to limit the space, but it suffers from coverage and efficiency issues. In this paper, we propose a template-free approach to efficiently explore the space of product molecules by first pinpointing the reaction center -- the set of nodes and edges where graph edits occur. Since only a small number of atoms contribute to reaction center, we can directly enumerate candidate products. The generated candidates are scored by a Weisfeiler-Lehman Difference Network that models high-order interactions between changes occurring at nodes across the molecule. Our framework outperforms the top-performing template-based approach with a 10\% margin, while running orders of magnitude faster. Finally, we demonstrate that the model accuracy rivals the performance of domain experts.
[Abstract](https://arxiv.org/abs/1709.04555), [PDF](https://arxiv.org/pdf/1709.04555)


### #251: Practical Data-Dependent Metric Compression with Provable Guarantees

### #252: REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models
_George Tucker,  Andriy Mnih,  Chris J. Maddison,  Dieterich Lawson,  Jascha Sohl-Dickstein_

Learning in models with discrete latent variables is challenging due to high variance gradient estimators. Generally, approaches have relied on control variates to reduce the variance of the REINFORCE estimator. Recent work (Jang et al. 2016; Maddison et al. 2016) has taken a different approach, introducing a continuous relaxation of discrete variables to produce low-variance, but biased, gradient estimates. In this work, we combine the two approaches through a novel control variate that produces low-variance, \emph{unbiased} gradient estimates. Then, we introduce a novel continuous relaxation and show that the tightness of the relaxation can be adapted online, removing it as a hyperparameter. We show state-of-the-art variance reduction on several benchmark generative modeling tasks, generally leading to faster convergence to a better final log likelihood.
[Abstract](https://arxiv.org/abs/1703.07370), [PDF](https://arxiv.org/pdf/1703.07370)


### #253: Nonlinear random matrix theory for deep learning

### #254: Parallel Streaming Wasserstein Barycenters
_Matthew Staib,  Sebastian Claici,  Justin Solomon,  Stefanie Jegelka_

Efficiently aggregating data from different sources is a challenging problem, particularly when samples from each source are distributed differently. These differences can be inherent to the inference task or present for other reasons: sensors in a sensor network may be placed far apart, affecting their individual measurements. Conversely, it is computationally advantageous to split Bayesian inference tasks across subsets of data, but data need not be identically distributed across subsets. One principled way to fuse probability distributions is via the lens of optimal transport: the Wasserstein barycenter is a single distribution that summarizes a collection of input measures while respecting their geometry. However, computing the barycenter scales poorly and requires discretization of all input distributions and the barycenter itself. Improving on this situation, we present a scalable, communication-efficient, parallel algorithm for computing the Wasserstein barycenter of arbitrary distributions. Our algorithm can operate directly on continuous input distributions and is optimized for streaming data. Our method is even robust to nonstationary input distributions and produces a barycenter estimate that tracks the input measures over time. The algorithm is semi-discrete, needing to discretize only the barycenter estimate. To the best of our knowledge, we also provide the first bounds on the quality of the approximate barycenter as the discretization becomes finer. Finally, we demonstrate the practical effectiveness of our method, both in tracking moving distributions on a sphere, as well as in a large-scale Bayesian inference task.
[Abstract](https://arxiv.org/abs/1705.07443), [PDF](https://arxiv.org/pdf/1705.07443)


### #255: ELF: An Extensive, Lightweight and Flexible Research Platform for Real-time Strategy Games
_Yuandong Tian,  Qucheng Gong,  Wenling Shang,  Yuxin Wu,  Larry Zitnick_

In this paper, we propose ELF, an Extensive, Lightweight and Flexible platform for fundamental reinforcement learning research. Using ELF, we implement a highly customizable real-time strategy (RTS) engine with three game environments (Mini-RTS, Capture the Flag and Tower Defense). Mini-RTS, as a miniature version of StarCraft, captures key game dynamics and runs at 40K frame-per-second (FPS) per core on a Macbook Pro notebook. When coupled with modern reinforcement learning methods, the system can train a full-game bot against built-in AIs end-to-end in one day with 6 CPUs and 1 GPU. In addition, our platform is flexible in terms of environment-agent communication topologies, choices of RL methods, changes in game parameters, and can host existing C/C++-based game environments like Arcade Learning Environment. Using ELF, we thoroughly explore training parameters and show that a network with Leaky ReLU and Batch Normalization coupled with long-horizon training and progressive curriculum beats the rule-based built-in AI more than $70\%$ of the time in the full game of Mini-RTS. Strong performance is also achieved on the other two games. In game replays, we show our agents learn interesting strategies. ELF, along with its RL platform, will be open-sourced.
[Abstract](https://arxiv.org/abs/1707.01067), [PDF](https://arxiv.org/pdf/1707.01067)


### #256: Dual Discriminator Generative Adversarial Nets
_Tu Dinh Nguyen,  Trung Le,  Hung Vu,  Dinh Phung_

We propose in this paper a novel approach to tackle the problem of mode collapse encountered in generative adversarial network (GAN). Our idea is intuitive but proven to be very effective, especially in addressing some key limitations of GAN. In essence, it combines the Kullback-Leibler (KL) and reverse KL divergences into a unified objective function, thus it exploits the complementary statistical properties from these divergences to effectively diversify the estimated density in capturing multi-modes. We term our method dual discriminator generative adversarial nets (D2GAN) which, unlike GAN, has two discriminators; and together with a generator, it also has the analogy of a minimax game, wherein a discriminator rewards high scores for samples from data distribution whilst another discriminator, conversely, favoring data from the generator, and the generator produces data to fool both two discriminators. We develop theoretical analysis to show that, given the maximal discriminators, optimizing the generator of D2GAN reduces to minimizing both KL and reverse KL divergences between data distribution and the distribution induced from the data generated by the generator, hence effectively avoiding the mode collapsing problem. We conduct extensive experiments on synthetic and real-world large-scale datasets (MNIST, CIFAR-10, STL-10, ImageNet), where we have made our best effort to compare our D2GAN with the latest state-of-the-art GAN's variants in comprehensive qualitative and quantitative evaluations. The experimental results demonstrate the competitive and superior performance of our approach in generating good quality and diverse samples over baselines, and the capability of our method to scale up to ImageNet database.
[Abstract](https://arxiv.org/abs/1709.03831), [PDF](https://arxiv.org/pdf/1709.03831)


### #257: Dynamic Revenue Sharing

### #258: Decomposition-Invariant Conditional Gradient for General Polytopes with Line Search

### #259: Multi-agent Predictive Modeling with Attentional CommNets

### #260: An Empirical Bayes Approach to Optimizing Machine Learning Algorithms

### #261: Differentially Private Empirical Risk Minimization Revisited: Faster and More General
_Raef Bassily,  Adam Smith,  Abhradeep Thakurta_

In this paper, we initiate a systematic investigation of differentially private algorithms for convex empirical risk minimization. Various instantiations of this problem have been studied before. We provide new algorithms and matching lower bounds for private ERM assuming only that each data point's contribution to the loss function is Lipschitz bounded and that the domain of optimization is bounded. We provide a separate set of algorithms and matching lower bounds for the setting in which the loss functions are known to also be strongly convex. Our algorithms run in polynomial time, and in some cases even match the optimal non-private running time (as measured by oracle complexity). We give separate algorithms (and lower bounds) for $(\epsilon,0)$- and $(\epsilon,\delta)$-differential privacy; perhaps surprisingly, the techniques used for designing optimal algorithms in the two cases are completely different. Our lower bounds apply even to very simple, smooth function families, such as linear and quadratic functions. This implies that algorithms from previous work can be used to obtain optimal error rates, under the additional assumption that the contributions of each data point to the loss function is smooth. We show that simple approaches to smoothing arbitrary loss functions (in order to apply previous techniques) do not yield optimal error rates. In particular, optimal algorithms were not previously known for problems such as training support vector machines and the high-dimensional median.
[Abstract](https://arxiv.org/abs/1405.7085), [PDF](https://arxiv.org/pdf/1405.7085)


### #262: Variational Inference via $\chi$ Upper Bound Minimization

### #263: On Quadratic Convergence of DC Proximal Newton Algorithm in Nonconvex Sparse Learning
_Xingguo Li,  Lin F. Yang,  Jason Ge,  Jarvis Haupt,  Tong Zhang,  Tuo Zhao_

We propose a DC proximal Newton algorithm for solving nonconvex regularized sparse learning problems in high dimensions. Our proposed algorithm integrates the proximal Newton algorithm with multi-stage convex relaxation based on difference of convex (DC) programming, and enjoys both strong computational and statistical guarantees. Specifically, by leveraging a sophisticated characterization of sparse modeling structures/assumptions (i.e., local restricted strong convexity and Hessian smoothness), we prove that within each stage of convex relaxation, our proposed algorithm achieves (local) quadratic convergence, and eventually obtains a sparse approximate local optimum with optimal statistical properties after only a few convex relaxations. Numerical experiments are provided to support our theory.
[Abstract](https://arxiv.org/abs/1706.06066), [PDF](https://arxiv.org/pdf/1706.06066)


### #264: #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning

### #265: An Empirical Study on The Properties of Random Bases for Kernel Methods

### #266: Bridging the Gap Between Value and Policy Based Reinforcement Learning
_Ofir Nachum,  Mohammad Norouzi,  Kelvin Xu,  Dale Schuurmans_

We establish a new connection between value and policy based reinforcement learning (RL) based on a relationship between softmax temporal value consistency and policy optimality under entropy regularization. Specifically, we show that softmax consistent action values satisfy a strong consistency property with optimal entropy regularized policy probabilities along any action sequence, regardless of provenance. From this observation, we develop a new RL algorithm, Path Consistency Learning (PCL), that minimizes inconsistency measured along multi-step action sequences extracted from both on- and off-policy traces. We subsequently deepen the relationship by showing how a single model can be used to represent both a policy and its softmax action values. Beyond eliminating the need for a separate critic, the unification demonstrates how policy gradients can be stabilized via self-bootstrapping from both on- and off-policy data. An experimental evaluation demonstrates that both algorithms can significantly outperform strong actor-critic and Q-learning baselines across several benchmark tasks.
[Abstract](https://arxiv.org/abs/1702.08892), [PDF](https://arxiv.org/pdf/1702.08892)


### #267: Premise Selection for Theorem Proving by Deep Graph Embedding
_Mingzhe Wang,  Yihe Tang,  Jian Wang,  Jia Deng_

We propose a deep learning-based approach to the problem of premise selection: selecting mathematical statements relevant for proving a given conjecture. We represent a higher-order logic formula as a graph that is invariant to variable renaming but still fully preserves syntactic and semantic information. We then embed the graph into a vector via a novel embedding method that preserves the information of edge ordering. Our approach achieves state-of-the-art results on the HolStep dataset, improving the classification accuracy from 83% to 90.3%.

### #268: A Bayesian Data Augmentation Approach for Learning Deep Models

### #269: Principles of Riemannian Geometry  in Neural Networks

### #270: Cold-Start Reinforcement Learning with Softmax Policy Gradients
_Nan Ding,  Radu Soricut_

Policy-gradient approaches to reinforcement learning have two common and undesirable overhead procedures, namely warm-start training and sample variance reduction. In this paper, we describe a reinforcement learning method based on a softmax policy that requires neither of these procedures. Our method combines the advantages of policy-gradient methods with the efficiency and simplicity of maximum-likelihood approaches. We apply this new cold-start reinforcement learning method in training sequence generation models for structured output prediction problems. Empirical evidence validates this method on automatic summarization and image captioning tasks.
[Abstract](http://arxiv.org/abs/1709.09346), [PDF](http://arxiv.org/pdf/1709.09346)


### #271: Online Dynamic Programming
_Holakou Rahmanian,  S.V.N. Vishwanathan,  Manfred K. Warmuth_

We consider the problem of repeatedly solving a variant of the same dynamic programming problem in successive trials. An instance of the type of problems we consider is to find the optimal binary search tree. At the beginning of each trial, the learner probabilistically chooses a tree with the n keys at the internal nodes and the n + 1 gaps between keys at the leaves. It is then told the frequencies of the keys and gaps and is charged by the average search cost for the chosen tree. The problem is online because the frequencies can change between trials. The goal is to develop algorithms with the property that their total average search cost (loss) in all trials is close to the total loss of the best tree chosen in hind sight for all trials. The challenge, of course, is that the algorithm has to deal with exponential number of trees. We develop a methodology for tackling such problems for a wide class of dynamic programming algorithms. Our framework allows us to extend online learning algorithms like Hedge and Component Hedge to a significantly wider class of combinatorial objects than was possible before.
[Abstract](https://arxiv.org/abs/1706.00834), [PDF](https://arxiv.org/pdf/1706.00834)


### #272: Alternating Estimation for Structured High-Dimensional Multi-Response Models
_Sheng Chen,  Arindam Banerjee_

We consider learning high-dimensional multi-response linear models with structured parameters. By exploiting the noise correlations among responses, we propose an alternating estimation (AltEst) procedure to estimate the model parameters based on the generalized Dantzig selector. Under suitable sample size and resampling assumptions, we show that the error of the estimates generated by AltEst, with high probability, converges linearly to certain minimum achievable level, which can be tersely expressed by a few geometric measures, such as Gaussian width of sets related to the parameter structure. To the best of our knowledge, this is the first non-asymptotic statistical guarantee for such AltEst-type algorithm applied to estimation problem with general structures.
[Abstract](https://arxiv.org/abs/1606.08957), [PDF](https://arxiv.org/pdf/1606.08957)


### #273: Convolutional Gaussian Processes
_Mark van der Wilk,  Carl Edward Rasmussen,  James Hensman_

We present a practical way of introducing convolutional structure into Gaussian processes, making them more suited to high-dimensional inputs like images. The main contribution of our work is the construction of an inter-domain inducing point approximation that is well-tailored to the convolutional kernel. This allows us to gain the generalisation benefit of a convolutional kernel, together with fast but accurate posterior inference. We investigate several variations of the convolutional kernel, and apply it to MNIST and CIFAR-10, which have both been known to be challenging for Gaussian processes. We also show how the marginal likelihood can be used to find an optimal weighting between convolutional and RBF kernels to further improve performance. We hope that this illustration of the usefulness of a marginal likelihood will help automate discovering architectures in larger models.
[Abstract](https://arxiv.org/abs/1709.01894), [PDF](https://arxiv.org/pdf/1709.01894)


### #274: Estimation of the covariance structure of heavy-tailed distributions
_Stanislav Minsker,  Xiaohan Wei_

We propose and analyze a new estimator of the covariance matrix that admits strong theoretical guarantees under weak assumptions on the underlying distribution, such as existence of moments of only low order. While estimation of covariance matrices corresponding to sub-Gaussian distributions is well-understood, much less in known in the case of heavy-tailed data. As K. Balasubramanian and M. Yuan write, "data from real-world experiments oftentimes tend to be corrupted with outliers and/or exhibit heavy tails. In such cases, it is not clear that those covariance matrix estimators .. remain optimal" and "..what are the other possible strategies to deal with heavy tailed distributions warrant further studies." We make a step towards answering this question and prove tight deviation inequalities for the proposed estimator that depend only on the parameters controlling the "intrinsic dimension" associated to the covariance matrix (as opposed to the dimension of the ambient space); in particular, our results are applicable in the case of high-dimensional observations.
[Abstract](https://arxiv.org/abs/1708.00502), [PDF](https://arxiv.org/pdf/1708.00502)


### #275: Mean Field Residual Networks: On the Edge of Chaos

### #276: Decomposable Submodular Function Minimization: Discrete and Continuous
_Alina Ene,  Huy L. Nguyen,  László A. Végh_

This paper investigates connections between discrete and continuous approaches for decomposable submodular function minimization. We provide improved running time estimates for the state-of-the-art continuous algorithms for the problem using combinatorial arguments. We also provide a systematic experimental comparison of the two types of methods, based on a clear distinction between level-0 and level-1 algorithms.
[Abstract](https://arxiv.org/abs/1703.01830), [PDF](https://arxiv.org/pdf/1703.01830)


### #277: Gauging Variational Inference

### #278: Deep Recurrent Neural Network-Based Identification of Precursor microRNAs

### #279: Robust Estimation of Neural Signals in Calcium Imaging

### #280: State Aware Imitation Learning

### #281: Beyond Parity: Fairness Objectives for Collaborative Filtering
_Sirui Yao,  Bert Huang_

We study fairness in collaborative-filtering recommender systems, which are sensitive to discrimination that exists in historical data. Biased data can lead collaborative-filtering methods to make unfair predictions for users from minority groups. We identify the insufficiency of existing fairness metrics and propose four new metrics that address different forms of unfairness. These fairness metrics can be optimized by adding fairness terms to the learning objective. Experiments on synthetic and real data show that our new metrics can better measure fairness than the baseline, and that the fairness objectives effectively help reduce unfairness.
[Abstract](https://arxiv.org/abs/1705.08804), [PDF](https://arxiv.org/pdf/1705.08804)


### #282: A PAC-Bayesian Analysis of Randomized Learning with Application to Stochastic Gradient Descent
_Ben London_

We analyze the generalization error of randomized learning algorithms -- focusing on stochastic gradient descent (SGD) -- using a novel combination of PAC-Bayes and algorithmic stability. Importantly, our risk bounds hold for all posterior distributions on the algorithm's random hyperparameters, including distributions that depend on the training data. This inspires an adaptive sampling algorithm for SGD that optimizes the posterior at runtime. We analyze this algorithm in the context of our risk bounds and evaluate it empirically on a benchmark dataset.
[Abstract](https://arxiv.org/abs/1709.06617), [PDF](https://arxiv.org/pdf/1709.06617)


### #283: Fully Decentralized Policies for Multi-Agent Systems: An Information Theoretic Approach
_Roel Dobbe,  David Fridovich-Keil,  Claire Tomlin_

Learning cooperative policies for multi-agent systems is often challenged by partial observability and a lack of coordination. In some settings, the structure of a problem allows a distributed solution with limited communication. Here, we consider a scenario where no communication is available, and instead we learn local policies for all agents that collectively mimic the solution to a centralized multi-agent static optimization problem. Our main contribution is an information theoretic framework based on rate distortion theory which facilitates analysis of how well the resulting fully decentralized policies are able to reconstruct the optimal solution. Moreover, this framework provides a natural extension that addresses which nodes an agent should communicate with to improve the performance of its individual policy.
[Abstract](https://arxiv.org/abs/1707.06334), [PDF](https://arxiv.org/pdf/1707.06334)


### #284: Model-Powered Conditional Independence Test
_Rajat Sen,  Ananda Theertha Suresh,  Karthikeyan Shanmugam,  Alexandros G. Dimakis,  Sanjay Shakkottai_

We consider the problem of non-parametric Conditional Independence testing (CI testing) for continuous random variables. Given i.i.d samples from the joint distribution $f(x,y,z)$ of continuous random vectors $X,Y$ and $Z,$ we determine whether $X \perp Y | Z$. We approach this by converting the conditional independence test into a classification problem. This allows us to harness very powerful classifiers like gradient-boosted trees and deep neural networks. These models can handle complex probability distributions and allow us to perform significantly better compared to the prior state of the art, for high-dimensional CI testing. The main technical challenge in the classification problem is the need for samples from the conditional product distribution $f^{CI}(x,y,z) = f(x|z)f(y|z)f(z)$ -- the joint distribution if and only if $X \perp Y | Z.$ -- when given access only to i.i.d. samples from the true joint distribution $f(x,y,z)$. To tackle this problem we propose a novel nearest neighbor bootstrap procedure and theoretically show that our generated samples are indeed close to $f^{CI}$ in terms of total variational distance. We then develop theoretical results regarding the generalization bounds for classification for our problem, which translate into error bounds for CI testing. We provide a novel analysis of Rademacher type classification bounds in the presence of non-i.i.d near-independent samples. We empirically validate the performance of our algorithm on simulated and real datasets and show performance gains over previous methods.
[Abstract](https://arxiv.org/abs/1709.06138), [PDF](https://arxiv.org/pdf/1709.06138)


### #285: Deep Voice 2: Multi-Speaker Neural Text-to-Speech
_Sercan Arik,  Gregory Diamos,  Andrew Gibiansky,  John Miller,  Kainan Peng,  Wei Ping,  Jonathan Raiman,  Yanqi Zhou_

We introduce a technique for augmenting neural text-to-speech (TTS) with lowdimensional trainable speaker embeddings to generate different voices from a single model. As a starting point, we show improvements over the two state-ofthe-art approaches for single-speaker neural TTS: Deep Voice 1 and Tacotron. We introduce Deep Voice 2, which is based on a similar pipeline with Deep Voice 1, but constructed with higher performance building blocks and demonstrates a significant audio quality improvement over Deep Voice 1. We improve Tacotron by introducing a post-processing neural vocoder, and demonstrate a significant audio quality improvement. We then demonstrate our technique for multi-speaker speech synthesis for both Deep Voice 2 and Tacotron on two multi-speaker TTS datasets. We show that a single neural TTS system can learn hundreds of unique voices from less than half an hour of data per speaker, while achieving high audio quality synthesis and preserving the speaker identities almost perfectly.
[Abstract](https://arxiv.org/abs/1705.08947), [PDF](https://arxiv.org/pdf/1705.08947)


### #286: Variance-based Regularization with Convex Objectives

### #287: Deep Lattice Networks and Partial Monotonic Functions
_Seungil You,  David Ding,  Kevin Canini,  Jan Pfeifer,  Maya Gupta_

We propose learning deep models that are monotonic with respect to a user-specified set of inputs by alternating layers of linear embeddings, ensembles of lattices, and calibrators (piecewise linear functions), with appropriate constraints for monotonicity, and jointly training the resulting network. We implement the layers and projections with new computational graph nodes in TensorFlow and use the ADAM optimizer and batched stochastic gradients. Experiments on benchmark and real-world datasets show that six-layer monotonic deep lattice networks achieve state-of-the art performance for classification and regression with monotonicity guarantees.
[Abstract](https://arxiv.org/abs/1709.06680), [PDF](https://arxiv.org/pdf/1709.06680)


### #288: Continual Learning with Deep Generative Replay
_Hanul Shin,  Jung Kwon Lee,  Jaehong Kim,  Jiwon Kim_

Attempts to train a comprehensive artificial intelligence capable of solving multiple tasks have been impeded by a chronic problem called catastrophic forgetting. Although simply replaying all previous data alleviates the problem, it requires large memory and even worse, often infeasible in real world applications where the access to past data is limited. Inspired by the generative nature of hippocampus as a short-term memory system in primate brain, we propose the Deep Generative Replay, a novel framework with a cooperative dual model architecture consisting of a deep generative model ("generator") and a task solving model ("solver"). With only these two models, training data for previous tasks can easily be sampled and interleaved with those for a new task. We test our methods in several sequential learning settings involving image classification tasks.
[Abstract](https://arxiv.org/abs/1705.08690), [PDF](https://arxiv.org/pdf/1705.08690)


### #289: AIDE: An algorithm for measuring the accuracy of probabilistic inference algorithms
_Marco F. Cusumano-Towner,  Vikash K. Mansinghka_

Approximate probabilistic inference algorithms are central to many fields. Examples include sequential Monte Carlo inference in robotics, variational inference in machine learning, and Markov chain Monte Carlo inference in statistics. A key problem faced by practitioners is measuring the accuracy of an approximate inference algorithm on a specific dataset. This paper introduces the auxiliary inference divergence estimator (AIDE), an algorithm for measuring the accuracy of approximate inference algorithms. AIDE is based on the observation that inference algorithms can be treated as probabilistic models and the random variables used within the inference algorithm can be viewed as auxiliary variables. This view leads to a new estimator for the symmetric KL divergence between the output distributions of two inference algorithms. The paper illustrates application of AIDE to algorithms for inference in regression, hidden Markov, and Dirichlet process mixture models. The experiments show that AIDE captures the qualitative behavior of a broad class of inference algorithms and can detect failure modes of inference algorithms that are missed by standard heuristics.
[Abstract](https://arxiv.org/abs/1705.07224), [PDF](https://arxiv.org/pdf/1705.07224)


### #290: Learning Causal Structures Using Regression Invariance
_AmirEmad Ghassami,  Saber Salehkaleybar,  Negar Kiyavash,  Kun Zhang_

We study causal inference in a multi-environment setting, in which the functional relations for producing the variables from their direct causes remain the same across environments, while the distribution of exogenous noises may vary. We introduce the idea of using the invariance of the functional relations of the variables to their causes across a set of environments. We define a notion of completeness for a causal inference algorithm in this setting and prove the existence of such algorithm by proposing the baseline algorithm. Additionally, we present an alternate algorithm that has significantly improved computational and sample complexity compared to the baseline algorithm. The experiment results show that the proposed algorithm outperforms the other existing algorithms.
[Abstract](https://arxiv.org/abs/1705.09644), [PDF](https://arxiv.org/pdf/1705.09644)


### #291: Online Influence Maximization under Independent Cascade Model with Semi-Bandit Feedback

### #292: Minimax Optimal Players for the Finite-Time 3-Expert Prediction Problem

### #293: Reinforcement Learning under Model Mismatch
_Aurko Roy,  Huan Xu,  Sebastian Pokutta_

We study reinforcement learning under model misspecification, where we do not have access to the true environment but only to a reasonably close approximation to it. We address this problem by extending the framework of robust MDPs to the model-free Reinforcement Learning setting, where we do not have access to the model parameters, but can only sample states from it. We define robust versions of Q-learning, SARSA, and TD-learning and prove convergence to an approximately optimal robust policy and approximate value function respectively. We scale up the robust algorithms to large MDPs via function approximation and prove convergence under two different settings. We prove convergence of robust approximate policy iteration and robust approximate value iteration for linear architectures (under mild assumptions). We also define a robust loss function, the mean squared robust projected Bellman error and give stochastic gradient descent algorithms that are guaranteed to converge to a local minimum.
[Abstract](https://arxiv.org/abs/1706.04711), [PDF](https://arxiv.org/pdf/1706.04711)


### #294: Hierarchical Attentive Recurrent Tracking
_Adam R. Kosiorek,  Alex Bewley,  Ingmar Posner_

Class-agnostic object tracking is particularly difficult in cluttered environments as target specific discriminative models cannot be learned a priori. Inspired by how the human visual cortex employs spatial attention and separate "where" and "what" processing pathways to actively suppress irrelevant visual features, this work develops a hierarchical attentive recurrent model for single object tracking in videos. The first layer of attention discards the majority of background by selecting a region containing the object of interest, while the subsequent layers tune in on visual features particular to the tracked object. This framework is fully differentiable and can be trained in a purely data driven fashion by gradient methods. To improve training convergence, we augment the loss function with terms for a number of auxiliary tasks relevant for tracking. Evaluation of the proposed model is performed on two datasets: pedestrian tracking on the KTH activity recognition dataset and the more difficult KITTI object tracking dataset.
[Abstract](https://arxiv.org/abs/1706.09262), [PDF](https://arxiv.org/pdf/1706.09262)


### #295: Tomography of the London Underground: a Scalable Model for Origin-Destination Data

### #296: Rotting Bandits
_Nir Levine,  Koby Crammer,  Shie Mannor_

The Multi-Armed Bandits (MAB) framework highlights the tension between acquiring new knowledge (Exploration) and leveraging available knowledge (Exploitation). In the classical MAB problem, a decision maker must choose an arm at each time step, upon which she receives a reward. The decision maker's objective is to maximize her cumulative expected reward over the time horizon. The MAB problem has been studied extensively, specifically under the assumption of the arms' rewards distributions being stationary, or quasi-stationary, over time. We consider a variant of the MAB framework, which we termed Rotting Bandits, where each arm's expected reward decays as a function of the number of times it has been pulled. We are motivated by many real-world scenarios such as online advertising, content recommendation, crowdsourcing, and more. We present algorithms, accompanied by simulations, and derive theoretical guarantees.
[Abstract](https://arxiv.org/abs/1702.07274), [PDF](https://arxiv.org/pdf/1702.07274)


### #297: Unbiased estimates for linear regression via volume sampling
_Michal Derezinski,  Manfred K. Warmuth_

Given a full rank matrix $X$ with more columns than rows consider the task of estimating the pseudo inverse $X^+$ based on the pseudo inverse of a sampled subset of columns (of size at least the number of rows). We show that this is possible if the subset of columns is chosen proportional to the squared volume spanned by the rows of the chosen submatrix (ie, volume sampling). The resulting estimator is unbiased and surprisingly the covariance of the estimator also has a closed form: It equals a specific factor times $X^+X^{+\top}$. Pseudo inverse plays an important part in solving the linear least squares problem, where we try to predict a label for each column of $X$. We assume labels are expensive and we are only given the labels for the small subset of columns we sample from $X$. Using our methods we show that the weight vector of the solution for the sub problem is an unbiased estimator of the optimal solution for the whole problem based on all column labels. We believe that these new formulas establish a fundamental connection between linear least squares and volume sampling. We use our methods to obtain an algorithm for volume sampling that is faster than state-of-the-art and for obtaining bounds for the total loss of the estimated least-squares solution on all labeled columns.
[Abstract](https://arxiv.org/abs/1705.06908), [PDF](https://arxiv.org/pdf/1705.06908)


### #298: An Applied Algorithmic Foundation for Hierarchical Clustering

### #299: Adaptive Accelerated Gradient Converging Method under H\"{o}lderian Error Bound Condition

### #300: Stein Variational Gradient Descent as Gradient Flow
_Qiang Liu_

Stein variational gradient descent (SVGD) is a deterministic sampling algorithm that iteratively transports a set of particles to approximate given distributions, based on an efficient gradient-based update that guarantees to optimally decrease the KL divergence within a function space. This paper develops the first theoretical analysis on SVGD, discussing its weak convergence properties and showing that its asymptotic behavior is captured by a gradient flow of the KL divergence functional under a new metric structure induced by Stein operator. We also provide a number of results on Stein operator and Stein's identity using the notion of weak derivative, including a new proof of the distinguishability of Stein discrepancy under weak conditions.
[Abstract](https://arxiv.org/abs/1704.07520), [PDF](https://arxiv.org/pdf/1704.07520)


### #301: Partial Hard Thresholding: A Towards Unified Analysis of Support Recovery

### #302: Shallow Updates for Deep Reinforcement Learning
_Nir Levine,  Tom Zahavy,  Daniel J. Mankowitz,  Aviv Tamar,  Shie Mannor_

Deep reinforcement learning (DRL) methods such as the Deep Q-Network (DQN) have achieved state-of-the-art results in a variety of challenging, high-dimensional domains. This success is mainly attributed to the power of deep neural networks to learn rich domain representations for approximating the value function or policy. Batch reinforcement learning methods with linear representations, on the other hand, are more stable and require less hyper parameter tuning. Yet, substantial feature engineering is necessary to achieve good results. In this work we propose a hybrid approach -- the Least Squares Deep Q-Network (LS-DQN), which combines rich feature representations learned by a DRL algorithm with the stability of a linear least squares method. We do this by periodically re-training the last hidden layer of a DRL network with a batch least squares update. Key to our approach is a Bayesian regularization term for the least squares update, which prevents over-fitting to the more recent data. We tested LS-DQN on five Atari games and demonstrate significant improvement over vanilla DQN and Double-DQN. We also investigated the reasons for the superior performance of our method. Interestingly, we found that the performance improvement can be attributed to the large batch size used by the LS method when optimizing the last layer.
[Abstract](https://arxiv.org/abs/1705.07461), [PDF](https://arxiv.org/pdf/1705.07461)


### #303: A Highly Efficient Gradient Boosting Decision Tree

### #304: Adversarial Ranking for Language Generation
_Kevin Lin,  Dianqi Li,  Xiaodong He,  Zhengyou Zhang,  Ming-Ting Sun_

Generative adversarial networks (GANs) have great successes on synthesizing data. However, the existing GANs restrict the discriminator to be a binary classifier, and thus limit their learning capacity for tasks that need to synthesize output with rich structures such as natural language descriptions. In this paper, we propose a novel generative adversarial network, RankGAN, for generating high-quality language descriptions. Rather than train the discriminator to learn and assign absolute binary predicate for individual data sample, the proposed RankGAN is able to analyze and rank a collection of human-written and machine-written sentences by giving a reference group. By viewing a set of data samples collectively and evaluating their quality through relative ranking scores, the discriminator is able to make better assessment which in turn helps to learn a better generator. The proposed RankGAN is optimized through the policy gradient technique. Experimental results on multiple public datasets clearly demonstrate the effectiveness of the proposed approach.
[Abstract](https://arxiv.org/abs/1705.11001), [PDF](https://arxiv.org/pdf/1705.11001)


### #305: Regret Minimization in MDPs with Options without Prior Knowledge

### #306: Net-Trim: Convex Pruning of Deep Neural Networks with Performance Guarantee

### #307: Graph Matching via Multiplicative Update Algorithm

### #308: Dynamic Importance Sampling for Anytime Bounds of the Partition Function

### #309: Is the Bellman residual a bad proxy?

### #310: Generalization Properties of Learning with Random Features
_Alessandro Rudi,  Lorenzo Rosasco_

We study the generalization properties of ridge regression with random features in the statistical learning framework. We show for the first time that $O(1/\sqrt{n})$ learning bounds can be achieved with only $O(\sqrt{n}\log n)$ random features rather than $O({n})$ as suggested by previous results. Further, we prove faster learning rates and show that they might require more random features, unless they are sampled according to a possibly problem dependent distribution. Our results shed light on the statistical computational trade-offs in large scale kernelized learning, showing the potential effectiveness of random features in reducing the computational complexity while keeping optimal generalization properties.
[Abstract](https://arxiv.org/abs/1602.04474), [PDF](https://arxiv.org/pdf/1602.04474)


### #311: Differentially private Bayesian learning on distributed data

### #312: Learning to Compose Domain-Specific Transformations for Data Augmentation
_Alexander J. Ratner,  Henry R. Ehrenberg,  Zeshan Hussain,  Jared Dunnmon,  Christopher Ré_

Data augmentation is a ubiquitous technique for increasing the size of labeled training sets by leveraging task-specific data transformations that preserve class labels. While it is often easy for domain experts to specify individual transformations, constructing and tuning the more sophisticated compositions typically needed to achieve state-of-the-art results is a time-consuming manual task in practice. We propose a method for automating this process by learning a generative sequence model over user-specified transformation functions using a generative adversarial approach. Our method can make use of arbitrary, non-deterministic transformation functions, is robust to misspecified user input, and is trained on unlabeled data. The learned transformation model can then be used to perform data augmentation for any end discriminative model. In our experiments, we show the efficacy of our approach on both image and text datasets, achieving improvements of 4.0 accuracy points on CIFAR-10, 1.4 F1 points on the ACE relation extraction task, and 3.4 accuracy points when using domain-specific transformation operations on a medical imaging dataset as compared to standard heuristic augmentation approaches.
[Abstract](https://arxiv.org/abs/1709.01643), [PDF](https://arxiv.org/pdf/1709.01643)


### #313: Wasserstein Learning of Deep Generative Point Process Models
_Shuai Xiao,  Mehrdad Farajtabar,  Xiaojing Ye,  Junchi Yan,  Le Song,  Hongyuan Zha_

Point processes are becoming very popular in modeling asynchronous sequential data due to their sound mathematical foundation and strength in modeling a variety of real-world phenomena. Currently, they are often characterized via intensity function which limits model's expressiveness due to unrealistic assumptions on its parametric form used in practice. Furthermore, they are learned via maximum likelihood approach which is prone to failure in multi-modal distributions of sequences. In this paper, we propose an intensity-free approach for point processes modeling that transforms nuisance processes to a target one. Furthermore, we train the model using a likelihood-free leveraging Wasserstein distance between point processes. Experiments on various synthetic and real-world data substantiate the superiority of the proposed point process model over conventional ones.
[Abstract](https://arxiv.org/abs/1705.08051), [PDF](https://arxiv.org/pdf/1705.08051)


### #314: Ensemble Sampling
_Xiuyuan Lu,  Benjamin Van Roy_

Thompson sampling has emerged as an effective heuristic for a broad range of online decision problems. In its basic form, the algorithm requires computing and sampling from a posterior distribution over models, which is tractable only for simple special cases. This paper develops ensemble sampling, which aims to approximate Thompson sampling while maintaining tractability even in the face of complex models such as neural networks. Ensemble sampling dramatically expands on the range of applications for which Thompson sampling is viable. We establish a theoretical basis that supports the approach and present computational results that offer further insight.
[Abstract](https://arxiv.org/abs/1705.07347), [PDF](https://arxiv.org/pdf/1705.07347)


### #315: Language modeling with recurrent highway hypernetworks

### #316: Searching in the Dark: Practical SVRG Methods under Error Bound Conditions with  Guarantee

### #317: Bayesian Compression for Deep Learning
_Christos Louizos,  Karen Ullrich,  Max Welling_

Compression and computational efficiency in deep learning have become a problem of great significance. In this work, we argue that the most principled and effective way to attack this problem is by taking a Bayesian point of view, where through sparsity inducing priors we prune large parts of the network. We introduce two novelties in this paper: 1) we use hierarchical priors to prune nodes instead of individual weights, and 2) we use the posterior uncertainties to determine the optimal fixed point precision to encode the weights. Both factors significantly contribute to achieving the state of the art in terms of compression rates, while still staying competitive with methods designed to optimize for speed or energy efficiency.
[Abstract](https://arxiv.org/abs/1705.08665), [PDF](https://arxiv.org/pdf/1705.08665)


### #318: Streaming Sparse Gaussian Process Approximations

### #319: VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning
_Akash Srivastava,  Lazar Valkov,  Chris Russell,  Michael Gutmann,  Charles Sutton_

Deep generative models provide powerful tools for distributions over complicated manifolds, such as those of natural images. But many of these methods, including generative adversarial networks (GANs), can be difficult to train, in part because they are prone to mode collapse, which means that they characterize only a few modes of the true distribution. To address this, we introduce VEEGAN, which features a reconstructor network, reversing the action of the generator by mapping from data to noise. Our training objective retains the original asymptotic consistency guarantee of GANs, and can be interpreted as a novel autoencoder loss over the noise. In sharp contrast to a traditional autoencoder over data points, VEEGAN does not require specifying a loss function over the data, but rather only over the representations, which are standard normal by assumption. On an extensive set of synthetic and real world image datasets, VEEGAN indeed resists mode collapsing to a far greater extent than other recent GAN variants, and produces more realistic samples.
[Abstract](https://arxiv.org/abs/1705.07761), [PDF](https://arxiv.org/pdf/1705.07761)


### #320: Sparse k-Means Embedding

### #321: Utile Context Tree Weighting

### #322: A Regularized Framework for Sparse and Structured Neural Attention
_Vlad Niculae,  Mathieu Blondel_

Modern neural networks are often augmented with an attention mechanism, which tells the network where to focus within the input. We propose in this paper a new framework for sparse and structured attention, building upon a max operator regularized with a strongly convex function. We show that this operator is differentiable and that its gradient defines a mapping from real values to probabilities, suitable as an attention mechanism. Our framework includes softmax and a slight generalization of the recently-proposed sparsemax as special cases. However, we also show how our framework can incorporate modern structured penalties, resulting in new attention mechanisms that focus on entire segments or groups of an input, encouraging parsimony and interpretability. We derive efficient algorithms to compute the forward and backward passes of these attention mechanisms, enabling their use in a neural network trained with backpropagation. To showcase their potential as a drop-in replacement for existing attention mechanisms, we evaluate them on three large-scale tasks: textual entailment, machine translation, and sentence summarization. Our attention mechanisms improve interpretability without sacrificing performance; notably, on textual entailment and summarization, we outperform the existing attention mechanisms based on softmax and sparsemax.
[Abstract](https://arxiv.org/abs/1705.07704), [PDF](https://arxiv.org/pdf/1705.07704)


### #323: Multi-output Polynomial Networks and Factorization Machines
_Mathieu Blondel,  Vlad Niculae,  Takuma Otsuka,  Naonori Ueda_

Factorization machines and polynomial networks are supervised polynomial models based on an efficient low-rank decomposition. We extend these models to the multi-output setting, i.e., for learning vector-valued functions, with application to multi-class or multi-task problems. We cast this as the problem of learning a 3-way tensor whose slices share a common decomposition and propose a convex formulation of that problem. We then develop an efficient conditional gradient algorithm and prove its global convergence, despite the fact that it involves a non-convex hidden unit selection step. On classification tasks, we show that our algorithm achieves excellent accuracy with much sparser models than existing methods. On recommendation system tasks, we show how to combine our algorithm with a reduction from ordinal regression to multi-output classification and show that the resulting algorithm outperforms existing baselines in terms of ranking accuracy.
[Abstract](https://arxiv.org/abs/1705.07603), [PDF](https://arxiv.org/pdf/1705.07603)


### #324: Clustering Billions of Reads for DNA Data Storage

### #325: Multi-Objective Non-parametric Sequential Prediction
_Guy Uziel,  Ran El-Yaniv_

Online-learning research has mainly been focusing on minimizing one objective function. In many real-world applications, however, several objective functions have to be considered simultaneously. Recently, an algorithm for dealing with several objective functions in the i.i.d. case has been presented. In this paper, we extend the multi-objective framework to the case of stationary and ergodic processes, thus allowing dependencies among observations. We first identify an asymptomatic lower bound for any prediction strategy and then present an algorithm whose predictions achieve the optimal solution while fulfilling any continuous and convex constraining criterion.
[Abstract](https://arxiv.org/abs/1703.01680), [PDF](https://arxiv.org/pdf/1703.01680)


### #326: A Universal Analysis of Large-Scale Regularized Least Squares Solutions

### #327: Deep Sets
_Manzil Zaheer,  Satwik Kottur,  Siamak Ravanbakhsh,  Barnabas Poczos,  Ruslan Salakhutdinov,  Alexander Smola_

In this paper, we study the problem of designing objective functions for machine learning problems defined on finite \emph{sets}. In contrast to traditional objective functions defined for machine learning problems operating on finite dimensional vectors, the new objective functions we propose are operating on finite sets and are invariant to permutations. Such problems are widespread, ranging from estimation of population statistics \citep{poczos13aistats}, via anomaly detection in piezometer data of embankment dams \citep{Jung15Exploration}, to cosmology \citep{Ntampaka16Dynamical,Ravanbakhsh16ICML1}. Our main theorem characterizes the permutation invariant objective functions and provides a family of functions to which any permutation invariant objective function must belong. This family of functions has a special structure which enables us to design a deep network architecture that can operate on sets and which can be deployed on a variety of scenarios including both unsupervised and supervised learning tasks. We demonstrate the applicability of our method on population statistic estimation, point cloud classification, set expansion, and image tagging.
[Abstract](https://arxiv.org/abs/1703.06114), [PDF](https://arxiv.org/pdf/1703.06114)


### #328: ExtremeWeather: A large-scale climate dataset for semi-supervised detection, localization, and understanding of extreme weather events

### #329: Process-constrained batch Bayesian optimisation

### #330: Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes
_Ahmed M. Alaa,  Mihaela van der Schaar_

Predicated on the increasing abundance of electronic health records, we investi- gate the problem of inferring individualized treatment effects using observational data. Stemming from the potential outcomes model, we propose a novel multi- task learning framework in which factual and counterfactual outcomes are mod- eled as the outputs of a function in a vector-valued reproducing kernel Hilbert space (vvRKHS). We develop a nonparametric Bayesian method for learning the treatment effects using a multi-task Gaussian process (GP) with a linear coregion- alization kernel as a prior over the vvRKHS. The Bayesian approach allows us to compute individualized measures of confidence in our estimates via pointwise credible intervals, which are crucial for realizing the full potential of precision medicine. The impact of selection bias is alleviated via a risk-based empirical Bayes method for adapting the multi-task GP prior, which jointly minimizes the empirical error in factual outcomes and the uncertainty in (unobserved) counter- factual outcomes. We conduct experiments on observational datasets for an inter- ventional social program applied to premature infants, and a left ventricular assist device applied to cardiac patients wait-listed for a heart transplant. In both experi- ments, we show that our method significantly outperforms the state-of-the-art.
[Abstract](https://arxiv.org/abs/1704.02801), [PDF](https://arxiv.org/pdf/1704.02801)


### #331: Spherical convolutions and their application in molecular modelling

### #332: Efficient Optimization for Linear Dynamical Systems with Applications to Clustering and Sparse Coding

### #333: On Optimal Generalizability in Parametric Learning

### #334: Near Optimal Sketching of Low-Rank Tensor Regression
_Jarvis Haupt,  Xingguo Li,  David P. Woodruff_

We study the least squares regression problem \begin{align*} \min_{\Theta \in \mathcal{S}_{\odot D,R}} \|A\Theta-b\|_2, \end{align*} where $\mathcal{S}_{\odot D,R}$ is the set of $\Theta$ for which $\Theta = \sum_{r=1}^{R} \theta_1^{(r)} \circ \cdots \circ \theta_D^{(r)}$ for vectors $\theta_d^{(r)} \in \mathbb{R}^{p_d}$ for all $r \in [R]$ and $d \in [D]$, and $\circ$ denotes the outer product of vectors. That is, $\Theta$ is a low-dimensional, low-rank tensor. This is motivated by the fact that the number of parameters in $\Theta$ is only $R \cdot \sum_{d=1}^D p_d$, which is significantly smaller than the $\prod_{d=1}^{D} p_d$ number of parameters in ordinary least squares regression. We consider the above CP decomposition model of tensors $\Theta$, as well as the Tucker decomposition. For both models we show how to apply data dimensionality reduction techniques based on {\it sparse} random projections $\Phi \in \mathbb{R}^{m \times n}$, with $m \ll n$, to reduce the problem to a much smaller problem $\min_{\Theta} \|\Phi A \Theta - \Phi b\|_2$, for which if $\Theta'$ is a near-optimum to the smaller problem, then it is also a near optimum to the original problem. We obtain significantly smaller dimension and sparsity in $\Phi$ than is possible for ordinary least squares regression, and we also provide a number of numerical simulations supporting our theory.
[Abstract](https://arxiv.org/abs/1709.07093), [PDF](https://arxiv.org/pdf/1709.07093)


### #335: Tractability in Structured Probability Spaces

### #336: Model-based Bayesian inference of neural activity and connectivity from all-optical interrogation of a neural circuit

### #337: Gaussian process based nonlinear latent structure discovery in multivariate spike train data 

### #338: Neural system identification for large populations separating "what" and "where"

### #339: Certified Defenses for Data Poisoning Attacks
_Jacob Steinhardt,  Pang Wei Koh,  Percy Liang_

Machine learning systems trained on user-provided data are susceptible to data poisoning attacks, whereby malicious users inject false training data with the aim of corrupting the learned model. While recent work has proposed a number of attacks and defenses, little is understood about the worst-case loss of a defense in the face of a determined attacker. We address this by constructing approximate upper bounds on the loss across a broad family of attacks, for defenders that first perform outlier removal followed by empirical risk minimization. Our bound comes paired with a candidate attack that nearly realizes the bound, giving us a powerful tool for quickly assessing defenses on a given dataset. Empirically, we find that even under a simple defense, the MNIST-1-7 and Dogfish datasets are resilient to attack, while in contrast the IMDB sentiment dataset can be driven from 12% to 23% test error by adding only 3% poisoned data.
[Abstract](https://arxiv.org/abs/1706.03691), [PDF](https://arxiv.org/pdf/1706.03691)


### #340: Eigen-Distortions of Hierarchical Representations
_Alexander Berardino,  Johannes Ballé,  Valero Laparra,  Eero P. Simoncelli_

We develop a method for comparing hierarchical image representations in terms of their ability to explain perceptual sensitivity in humans. Specifically, we utilize Fisher information to establish a model-derived prediction of sensitivity to local perturbations around a given natural image. For a given image, we compute the eigenvectors of the Fisher information matrix with largest and smallest eigenvalues, corresponding to the model-predicted most- and least-noticeable image distortions, respectively. For human subjects, we then measure the amount of each distortion that can be reliably detected when added to the image, and compare these thresholds to the predictions of the corresponding model. We use this method to test the ability of a variety of representations to mimic human perceptual sensitivity. We find that the early layers of VGG16, a deep neural network optimized for object recognition, provide a better match to human perception than later layers, and a better match than a 4-stage convolutional neural network (CNN) trained on a database of human ratings of distorted image quality. On the other hand, we find that simple models of early visual processing, incorporating one or more stages of local gain control, trained on the same database of distortion ratings, provide substantially better predictions of human sensitivity than both the CNN and all layers of VGG16.

### #341: Limitations on Variance-Reduction and Acceleration Schemes for Finite Sums Optimization
_Yossi Arjevani_

We study the conditions under which one is able to efficiently apply variance-reduction and acceleration schemes on finite sum optimization problems. First, we show that, perhaps surprisingly, the finite sum structure by itself, is not sufficient for obtaining a complexity bound of $\tilde{\cO}((n+L/\mu)\ln(1/\epsilon))$ for $L$-smooth and $\mu$-strongly convex individual functions - one must also know which individual function is being referred to by the oracle at each iteration. Next, we show that for a broad class of first-order and coordinate-descent finite sum algorithms (including, e.g., SDCA, SVRG, SAG), it is not possible to get an `accelerated' complexity bound of $\tilde{\cO}((n+\sqrt{n L/\mu})\ln(1/\epsilon))$, unless the strong convexity parameter is given explicitly. Lastly, we show that when this class of algorithms is used for minimizing $L$-smooth and convex finite sums, the optimal complexity bound is $\tilde{\cO}(n+L/\epsilon)$, assuming that (on average) the same update rule is used in every iteration, and $\tilde{\cO}(n+\sqrt{nL/\epsilon})$, otherwise.
[Abstract](https://arxiv.org/abs/1706.01686), [PDF](https://arxiv.org/pdf/1706.01686)


### #342: Unsupervised Sequence Classification using Sequential Output Statistics

### #343: Subset Selection under Noise

### #344: Collecting Telemetry Data Privately

### #345: Concrete Dropout
_Yarin Gal,  Jiri Hron,  Alex Kendall_

Dropout is used as a practical tool to obtain uncertainty estimates in large vision models and reinforcement learning (RL) tasks. But to obtain well-calibrated uncertainty estimates, a grid-search over the dropout probabilities is necessary - a prohibitive operation with large models, and an impossible one with RL. We propose a new dropout variant which gives improved performance and better calibrated uncertainties. Relying on recent developments in Bayesian deep learning, we use a continuous relaxation of dropout's discrete masks. Together with a principled optimisation objective, this allows for automatic tuning of the dropout probability in large models, and as a result faster experimentation cycles. In RL this allows the agent to adapt its uncertainty dynamically as more data is observed. We analyse the proposed variant extensively on a range of tasks, and give insights into common practice in the field where larger dropout probabilities are often used in deeper model layers.
[Abstract](https://arxiv.org/abs/1705.07832), [PDF](https://arxiv.org/pdf/1705.07832)


### #346: Adaptive Batch Size for Safe Policy Gradients

### #347: A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning

### #348: PASS-GLM: polynomial approximate sufficient statistics for scalable Bayesian GLM inference
_Jonathan H. Huggins,  Ryan P. Adams,  Tamara Broderick_

Generalized linear models (GLMs) -- such as logistic regression, Poisson regression, and robust regression -- provide interpretable models for diverse data types. Probabilistic approaches, particularly Bayesian ones, allow coherent estimates of uncertainty, incorporation of prior information, and sharing of power across experiments via hierarchical models. In practice, however, the approximate Bayesian methods necessary for inference have either failed to scale to large data sets or failed to provide theoretical guarantees on the quality of inference. We propose a new approach based on constructing polynomial approximate sufficient statistics for GLMs (PASS-GLM). We demonstrate that our method admits a simple algorithm as well as trivial streaming and distributed extensions that do not compound error across computations. We provide theoretical guarantees on the quality of point (MAP) estimates, the approximate posterior, and posterior mean and uncertainty estimates. We validate our approach empirically in the case of logistic regression using a quadratic approximation and show competitive performance with stochastic gradient descent, MCMC, and the Laplace approximation in terms of speed and multiple measures of accuracy -- including on an advertising data set with 40 million data points and 20,000 covariates.
[Abstract](https://arxiv.org/abs/1709.09216), [PDF](https://arxiv.org/pdf/1709.09216)


### #349: Bayesian GANs

### #350: Off-policy evaluation for slate recommendation
_Adith Swaminathan,  Akshay Krishnamurthy,  Alekh Agarwal,  Miroslav Dudík,  John Langford,  Damien Jose,  Imed Zitouni_

This paper studies the evaluation of policies that recommend an ordered set of items (e.g., a ranking) based on some context---a common scenario in web search, ads and recommender systems. We develop the first practical technique for evaluating page-level metrics of such policies offline using logged past data, alleviating the need for online A/B tests. Our method models the observed quality of the recommended set (e.g., time to success in web search) as an additive decomposition across items. Crucially, the per-item quality is not directly observed or easily modeled from the item's features. A thorough empirical evaluation reveals that this model fits many realistic measures of quality and theoretical analysis shows exponential savings in the amount of required data compared with prior off-policy evaluation approaches.
[Abstract](https://arxiv.org/abs/1605.04812), [PDF](https://arxiv.org/pdf/1605.04812)


### #351: A multi-agent reinforcement learning model of common-pool resource appropriation
_Julien Perolat,  Joel Z. Leibo,  Vinicius Zambaldi,  Charles Beattie,  Karl Tuyls,  Thore Graepel_

Humanity faces numerous problems of common-pool resource appropriation. This class of multi-agent social dilemma includes the problems of ensuring sustainable use of fresh water, common fisheries, grazing pastures, and irrigation systems. Abstract models of common-pool resource appropriation based on non-cooperative game theory predict that self-interested agents will generally fail to find socially positive equilibria---a phenomenon called the tragedy of the commons. However, in reality, human societies are sometimes able to discover and implement stable cooperative solutions. Decades of behavioral game theory research have sought to uncover aspects of human behavior that make this possible. Most of that work was based on laboratory experiments where participants only make a single choice: how much to appropriate. Recognizing the importance of spatial and temporal resource dynamics, a recent trend has been toward experiments in more complex real-time video game-like environments. However, standard methods of non-cooperative game theory can no longer be used to generate predictions for this case. Here we show that deep reinforcement learning can be used instead. To that end, we study the emergent behavior of groups of independently learning agents in a partially observed Markov game modeling common-pool resource appropriation. Our experiments highlight the importance of trial-and-error learning in common-pool resource appropriation and shed light on the relationship between exclusion, sustainability, and inequality.
[Abstract](https://arxiv.org/abs/1707.06600), [PDF](https://arxiv.org/pdf/1707.06600)


### #352: On the Optimization Landscape of Tensor Decompositions
_Rong Ge,  Tengyu Ma_

Non-convex optimization with local search heuristics has been widely used in machine learning, achieving many state-of-art results. It becomes increasingly important to understand why they can work for these NP-hard problems on typical data. The landscape of many objective functions in learning has been conjectured to have the geometric property that "all local optima are (approximately) global optima", and thus they can be solved efficiently by local search algorithms. However, establishing such property can be very difficult. In this paper, we analyze the optimization landscape of the random over-complete tensor decomposition problem, which has many applications in unsupervised learning, especially in learning latent variable models. In practice, it can be efficiently solved by gradient ascent on a non-convex objective. We show that for any small constant $\epsilon > 0$, among the set of points with function values $(1+\epsilon)$-factor larger than the expectation of the function, all the local maxima are approximate global maxima. Previously, the best-known result only characterizes the geometry in small neighborhoods around the true components. Our result implies that even with an initialization that is barely better than the random guess, the gradient ascent algorithm is guaranteed to solve this problem. Our main technique uses Kac-Rice formula and random matrix theory. To our best knowledge, this is the first time when Kac-Rice formula is successfully applied to counting the number of local minima of a highly-structured random polynomial with dependent coefficients.
[Abstract](https://arxiv.org/abs/1706.05598), [PDF](https://arxiv.org/pdf/1706.05598)


### #353: High-Order Attention Models for Visual Question Answering

### #354: Sparse convolutional coding for neuronal assembly detection

### #355: Quantifying how much sensory information in a neural code is relevant for behavior

### #356: Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks
_Federico Monti,  Michael M. Bronstein,  Xavier Bresson_

Matrix completion models are among the most common formulations of recommender systems. Recent works have showed a boost of performance of these techniques when introducing the pairwise relationships between users/items in the form of graphs, and imposing smoothness priors on these graphs. However, such techniques do not fully exploit the local stationarity structures of user/item graphs, and the number of parameters to learn is linear w.r.t. the number of users and items. We propose a novel approach to overcome these limitations by using geometric deep learning on graphs. Our matrix completion architecture combines graph convolutional neural networks and recurrent neural networks to learn meaningful statistical graph-structured patterns and the non-linear diffusion process that generates the known ratings. This neural network system requires a constant number of parameters independent of the matrix size. We apply our method on both synthetic and real datasets, showing that it outperforms state-of-the-art techniques.
[Abstract](https://arxiv.org/abs/1704.06803), [PDF](https://arxiv.org/pdf/1704.06803)


### #357: Reducing Reparameterization Gradient Variance
_Andrew C. Miller,  Nicholas J. Foti,  Alexander D'Amour,  Ryan P. Adams_

Optimization with noisy gradients has become ubiquitous in statistics and machine learning. Reparameterization gradients, or gradient estimates computed via the "reparameterization trick," represent a class of noisy gradients often used in Monte Carlo variational inference (MCVI). However, when these gradient estimators are too noisy, the optimization procedure can be slow or fail to converge. One way to reduce noise is to use more samples for the gradient estimate, but this can be computationally expensive. Instead, we view the noisy gradient as a random variable, and form an inexpensive approximation of the generating procedure for the gradient sample. This approximation has high correlation with the noisy gradient by construction, making it a useful control variate for variance reduction. We demonstrate our approach on non-conjugate multi-level hierarchical models and a Bayesian neural net where we observed gradient variance reductions of multiple orders of magnitude (20-2,000x).
[Abstract](https://arxiv.org/abs/1705.07880), [PDF](https://arxiv.org/pdf/1705.07880)


### #358: Visual Reference Resolution using Attention Memory for Visual Dialog
_Paul Hongsuck Seo,  Andreas Lehrmann,  Bohyung Han,  Leonid Sigal_

Visual dialog is a task of answering a series of inter-dependent questions given an input image, and often requires to resolve visual references among the questions. This problem is different from visual question answering (VQA), which relies on spatial attention (a.k.a. visual grounding) estimated from an image and question pair. We propose a novel attention mechanism that exploits visual attentions in the past to resolve the current reference in the visual dialog scenario. The proposed model is equipped with an associative attention memory storing a sequence of previous (attention, key) pairs. From this memory, the model retrieves previous attention, taking into account recency, that is most relevant for the current question, in order to resolve potentially ambiguous reference(s). The model then merges the retrieved attention with the tentative one to obtain the final attention for the current question; specifically, we use dynamic parameter prediction to combine the two attentions conditioned on the question. Through extensive experiments on a new synthetic visual dialog dataset, we show that our model significantly outperforms the state-of-the-art (by ~16 % points) in the situation where the visual reference resolution plays an important role. Moreover, the proposed model presents superior performance (~2 % points improvement) in the Visual Dialog dataset, despite having significantly fewer parameters than the baselines.
[Abstract](https://arxiv.org/abs/1709.07992), [PDF](https://arxiv.org/pdf/1709.07992)


### #359: Joint distribution optimal transportation for domain adaptation

### #360: Multiresolution Kernel Approximation for Gaussian Process Regression
_Yi Ding,  Risi Kondor,  Jonathan Eskreis-Winkler_

Gaussian process regression generally does not scale to beyond a few thousands data points without applying some sort of kernel approximation method. Most approximations focus on the high eigenvalue part of the spectrum of the kernel matrix, $K$, which leads to bad performance when the length scale of the kernel is small. In this paper we introduce Multiresolution Kernel Approximation (MKA), the first true broad bandwidth kernel approximation algorithm. Important points about MKA are that it is memory efficient, and it is a direct method, which means that it also makes it easy to approximate $K^{-1}$ and $\mathop{\textrm{det}}(K)$.
[Abstract](https://arxiv.org/abs/1708.02183), [PDF](https://arxiv.org/pdf/1708.02183)


### #361: Collapsed variational Bayes for Markov jump processes

### #362: Universal consistency and minimax rates for online Mondrian Forest

### #363: Efficiency Guarantees from Data

### #364: Diving into the shallows: a computational perspective on large-scale shallow learning
_Siyuan Ma,  Mikhail Belkin_

In this paper we first identify a basic limitation in gradient descent-based optimization methods when used in conjunctions with smooth kernels. An analysis based on the spectral properties of the kernel demonstrates that only a vanishingly small portion of the function space is reachable after a polynomial number of gradient descent iterations. This lack of approximating power drastically limits gradient descent for a fixed computational budget leading to serious over-regularization/underfitting. The issue is purely algorithmic, persisting even in the limit of infinite data. To address this shortcoming in practice, we introduce EigenPro iteration, based on a preconditioning scheme using a small number of approximately computed eigenvectors. It can also be viewed as learning a new kernel optimized for gradient descent. It turns out that injecting this small (computationally inexpensive and SGD-compatible) amount of approximate second-order information leads to major improvements in convergence. For large data, this translates into significant performance boost over the standard kernel methods. In particular, we are able to consistently match or improve the state-of-the-art results recently reported in the literature with a small fraction of their computational budget. Finally, we feel that these results show a need for a broader computational perspective on modern large-scale learning to complement more traditional statistical and convergence analyses. In particular, many phenomena of large-scale high-dimensional inference are best understood in terms of optimization on infinite dimensional Hilbert spaces, where standard algorithms can sometimes have properties at odds with finite-dimensional intuition. A systematic analysis concentrating on the approximation power of such algorithms within a budget of computation may lead to progress both in theory and practice.
[Abstract](https://arxiv.org/abs/1703.10622), [PDF](https://arxiv.org/pdf/1703.10622)


### #365: End-to-end Differentiable Proving
_Tim Rocktäschel,  Sebastian Riedel_

We introduce neural networks for end-to-end differentiable theorem proving that operate on dense vector representations of symbols. These neural networks are constructed recursively by taking inspiration from the backward chaining algorithm as used in Prolog. Specifically, we replace symbolic unification with a differentiable computation on vector representations of symbols using a radial basis function kernel, thereby combining symbolic reasoning with learning subsymbolic vector representations. By using gradient descent, the resulting neural network can be trained to infer facts from a given incomplete knowledge base. It learns to (i) place representations of similar symbols in close proximity in a vector space, (ii) make use of such similarities to prove facts, (iii) induce logical rules, and (iv) use provided and induced logical rules for complex multi-hop reasoning. We demonstrate that this architecture outperforms ComplEx, a state-of-the-art neural link prediction model, on four benchmark knowledge bases while at the same time inducing interpretable function-free first-order logic rules.
[Abstract](https://arxiv.org/abs/1705.11040), [PDF](https://arxiv.org/pdf/1705.11040)


### #366: Influence Maximization with $\varepsilon$-Almost Submodular Threshold Function

### #367: Inferring The Latent Structure of Human Decision-Making from Raw Visual Inputs
_Yunzhu Li,  Jiaming Song,  Stefano Ermon_

The goal of imitation learning is to match example expert behavior, without access to a reinforcement signal. Expert demonstrations provided by humans, however, often show significant variability due to latent factors that are not explicitly modeled. We introduce an extension to the Generative Adversarial Imitation Learning method that can infer the latent structure of human decision-making in an unsupervised way. Our method can not only imitate complex behaviors, but also learn interpretable and meaningful representations. We demonstrate that the approach is applicable to high-dimensional environments including raw visual inputs. In the highway driving domain, we show that a model learned from demonstrations is able to both produce different styles of human-like driving behaviors and accurately anticipate human actions. Our method surpasses various baselines in terms of performance and functionality.
[Abstract](https://arxiv.org/abs/1703.08840), [PDF](https://arxiv.org/pdf/1703.08840)


### #368: Variational Laws of Visual Attention for Dynamic Scenes

### #369: Recursive Sampling for the Nystrom Method

### #370: Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning
_Shixiang Gu,  Timothy Lillicrap,  Zoubin Ghahramani,  Richard E. Turner,  Bernhard Schölkopf,  Sergey Levine_

Off-policy model-free deep reinforcement learning methods using previously collected data can improve sample efficiency over on-policy policy gradient techniques. On the other hand, on-policy algorithms are often more stable and easier to use. This paper examines, both theoretically and empirically, approaches to merging on- and off-policy updates for deep reinforcement learning. Theoretical results show that off-policy updates with a value function estimator can be interpolated with on-policy policy gradient updates whilst still satisfying performance bounds. Our analysis uses control variate methods to produce a family of policy gradient algorithms, with several recently proposed algorithms being special cases of this family. We then provide an empirical comparison of these techniques with the remaining algorithmic details fixed, and show how different mixing of off-policy gradient estimates with on-policy samples contribute to improvements in empirical performance. The final algorithm provides a generalization and unification of existing deep policy gradient techniques, has theoretical guarantees on the bias introduced by off-policy updates, and improves on the state-of-the-art model-free deep RL methods on a number of OpenAI Gym continuous control benchmarks.
[Abstract](https://arxiv.org/abs/1706.00387), [PDF](https://arxiv.org/pdf/1706.00387)


### #371: Dynamic Routing Between Capsules

### #372: Incorporating Side Information by Adaptive Convolution

### #373: Conic Scan Coverage algorithm for nonparametric topic modeling

### #374: FALKON: An Optimal Large Scale Kernel Method
_Alessandro Rudi,  Luigi Carratino,  Lorenzo Rosasco_

Kernel methods provide a principled way to perform non linear, nonparametric learning. They rely on solid functional analytic foundations and enjoy optimal statistical properties. However, at least in their basic form, they have limited applicability in large scale scenarios because of stringent computational requirements in terms of time and especially memory. In this paper, we take a substantial step in scaling up kernel methods, proposing FALKON, a novel algorithm that allows to efficiently process millions of points. FALKON is derived combining several algorithmic principles, namely stochastic projections, iterative solvers and preconditioning. Our theoretical analysis shows that optimal statistical accuracy is achieved requiring essentially $O(n)$ memory and $O(n\sqrt{n})$ time. Extensive experiments show that state of the art results on available large scale datasets can be achieved even on a single machine.
[Abstract](https://arxiv.org/abs/1705.10958), [PDF](https://arxiv.org/pdf/1705.10958)


### #375: Structured Generative Adversarial Networks

### #376: Conservative Contextual Linear Bandits
_Abbas Kazerouni,  Mohammad Ghavamzadeh,  Yasin Abbasi-Yadkori,  Benjamin Van Roy_

Safety is a desirable property that can immensely increase the applicability of learning algorithms in real-world decision-making problems. It is much easier for a company to deploy an algorithm that is safe, i.e., guaranteed to perform at least as well as a baseline. In this paper, we study the issue of safety in contextual linear bandits that have application in many different fields including personalized ad recommendation in online marketing. We formulate a notion of safety for this class of algorithms. We develop a safe contextual linear bandit algorithm, called conservative linear UCB (CLUCB), that simultaneously minimizes its regret and satisfies the safety constraint, i.e., maintains its performance above a fixed percentage of the performance of a baseline strategy, uniformly over time. We prove an upper-bound on the regret of CLUCB and show that it can be decomposed into two terms: 1) an upper-bound for the regret of the standard linear UCB algorithm that grows with the time horizon and 2) a constant (does not grow with the time horizon) term that accounts for the loss of being conservative in order to satisfy the safety constraint. We empirically show that our algorithm is safe and validate our theoretical analysis.
[Abstract](https://arxiv.org/abs/1611.06426), [PDF](https://arxiv.org/pdf/1611.06426)


### #377: Variational Memory Addressing in Generative Models
_Jörg Bornschein,  Andriy Mnih,  Daniel Zoran,  Danilo J. Rezende_

Aiming to augment generative models with external memory, we interpret the output of a memory module with stochastic addressing as a conditional mixture distribution, where a read operation corresponds to sampling a discrete memory address and retrieving the corresponding content from memory. This perspective allows us to apply variational inference to memory addressing, which enables effective training of the memory module by using the target information to guide memory lookups. Stochastic addressing is particularly well-suited for generative models as it naturally encourages multimodality which is a prominent aspect of most high-dimensional datasets. Treating the chosen address as a latent variable also allows us to quantify the amount of information gained with a memory lookup and measure the contribution of the memory module to the generative process. To illustrate the advantages of this approach we incorporate it into a variational autoencoder and apply the resulting model to the task of generative few-shot learning. The intuition behind this architecture is that the memory module can pick a relevant template from memory and the continuous part of the model can concentrate on modeling remaining variations. We demonstrate empirically that our model is able to identify and access the relevant memory contents even with hundreds of unseen Omniglot characters in memory
[Abstract](https://arxiv.org/abs/1709.07116), [PDF](https://arxiv.org/pdf/1709.07116)


### #378: On Tensor Train Rank Minimization : Statistical Efficiency and Scalable Algorithm

### #379: Scalable Levy Process Priors for Spectral Kernel Learning

### #380: Deep Hyperspherical Learning

### #381: Learning Deep Structured Multi-Scale Features using Attention-Gated CRFs for Contour Prediction

### #382: On-the-fly Operation Batching in Dynamic Computation Graphs
_Graham Neubig,  Yoav Goldberg,  Chris Dyer_

Dynamic neural network toolkits such as PyTorch, DyNet, and Chainer offer more flexibility for implementing models that cope with data of varying dimensions and structure, relative to toolkits that operate on statically declared computations (e.g., TensorFlow, CNTK, and Theano). However, existing toolkits - both static and dynamic - require that the developer organize the computations into the batches necessary for exploiting high-performance algorithms and hardware. This batching task is generally difficult, but it becomes a major hurdle as architectures become complex. In this paper, we present an algorithm, and its implementation in the DyNet toolkit, for automatically batching operations. Developers simply write minibatch computations as aggregations of single instance computations, and the batching algorithm seamlessly executes them, on the fly, using computationally efficient batched operations. On a variety of tasks, we obtain throughput similar to that obtained with manual batches, as well as comparable speedups over single-instance learning on architectures that are impractical to batch manually.
[Abstract](https://arxiv.org/abs/1705.07860), [PDF](https://arxiv.org/pdf/1705.07860)


### #383: Nonlinear Acceleration of Stochastic Algorithms
_Damien Scieur,  Alexandre d'Aspremont,  Francis Bach_

Extrapolation methods use the last few iterates of an optimization algorithm to produce a better estimate of the optimum. They were shown to achieve optimal convergence rates in a deterministic setting using simple gradient iterates. Here, we study extrapolation methods in a stochastic setting, where the iterates are produced by either a simple or an accelerated stochastic gradient algorithm. We first derive convergence bounds for arbitrary, potentially biased perturbations, then produce asymptotic bounds using the ratio between the variance of the noise and the accuracy of the current point. Finally, we apply this acceleration technique to stochastic algorithms such as SGD, SAGA, SVRG and Katyusha in different settings, and show significant performance gains.
[Abstract](https://arxiv.org/abs/1706.07270), [PDF](https://arxiv.org/pdf/1706.07270)


### #384: Optimized Pre-Processing for Discrimination Prevention

### #385: YASS: Yet Another Spike Sorter

### #386: Independence clustering (without a matrix)
_Daniil Ryabko_

The independence clustering problem is considered in the following formulation: given a set $S$ of random variables, it is required to find the finest partitioning $\{U_1,\dots,U_k\}$ of $S$ into clusters such that the clusters $U_1,\dots,U_k$ are mutually independent. Since mutual independence is the target, pairwise similarity measurements are of no use, and thus traditional clustering algorithms are inapplicable. The distribution of the random variables in $S$ is, in general, unknown, but a sample is available. Thus, the problem is cast in terms of time series. Two forms of sampling are considered: i.i.d.\ and stationary time series, with the main emphasis being on the latter, more general, case. A consistent, computationally tractable algorithm for each of the settings is proposed, and a number of open directions for further research are outlined.
[Abstract](https://arxiv.org/abs/1703.06700), [PDF](https://arxiv.org/pdf/1703.06700)


### #387: Fast amortized inference of neural activity from calcium imaging data with variational autoencoders

### #388: Adaptive Active Hypothesis Testing under Limited Information

### #389: Streaming Weak Submodularity: Interpreting Neural Networks on the Fly
_Ethan R. Elenberg,  Alexandros G. Dimakis,  Moran Feldman,  Amin Karbasi_

In many machine learning applications, it is important to explain the predictions of a black-box classifier. For example, why does a deep neural network assign an image to a particular class? We cast interpretability of black-box classifiers as a combinatorial maximization problem and propose an efficient streaming algorithm to solve it subject to cardinality constraints. By extending ideas from Badanidiyuru et al. [2014], we provide a constant factor approximation guarantee for our algorithm in the case of random stream order and a weakly submodular objective function. This is the first such theoretical guarantee for this general class of functions, and we also show that no such algorithm exists for a worst case stream order. Our algorithm obtains similar explanations of Inception V3 predictions $10$ times faster than the state-of-the-art LIME framework of Ribeiro et al. [2016].
[Abstract](https://arxiv.org/abs/1703.02647), [PDF](https://arxiv.org/pdf/1703.02647)


### #390: Successor Features for Transfer in Reinforcement Learning
_André Barreto,  Rémi Munos,  Tom Schaul,  David Silver_

Transfer in reinforcement learning refers to the notion that generalization should occur not only within a task but also across tasks. Our focus is on transfer where the reward functions vary across tasks while the environment's dynamics remain the same. The method we propose rests on two key ideas: "successor features," a value function representation that decouples the dynamics of the environment from the rewards, and "generalized policy improvement," a generalization of dynamic programming's policy improvement step that considers a set of policies rather than a single one. Put together, the two ideas lead to an approach that integrates seamlessly within the reinforcement learning framework and allows transfer to take place between tasks without any restriction. The proposed method also provides performance guarantees for the transferred policy even before any learning has taken place. We derive two theorems that set our approach in firm theoretical ground and present experiments that show that it successfully promotes transfer in practice.
[Abstract](https://arxiv.org/abs/1606.05312), [PDF](https://arxiv.org/pdf/1606.05312)


### #391: Counterfactual Fairness
_Matt J. Kusner,  Joshua R. Loftus,  Chris Russell,  Ricardo Silva_

Machine learning can impact people with legal or ethical consequences when it is used to automate decisions in areas such as insurance, lending, hiring, and predictive policing. In many of these scenarios, previous decisions have been made that are unfairly biased against certain subpopulations, for example those of a particular race, gender, or sexual orientation. Since this past data may be biased, machine learning predictors must account for this to avoid perpetuating or creating discriminatory practices. In this paper, we develop a framework for modeling fairness using tools from causal inference. Our definition of counterfactual fairness captures the intuition that a decision is fair towards an individual if it the same in (a) the actual world and (b) a counterfactual world where the individual belonged to a different demographic group. We demonstrate our framework on a real-world problem of fair prediction of success in law school.
[Abstract](https://arxiv.org/abs/1703.06856), [PDF](https://arxiv.org/pdf/1703.06856)


### #392: Prototypical Networks for Few-shot Learning
_Jake Snell,  Kevin Swersky,  Richard S. Zemel_

We propose prototypical networks for the problem of few-shot classification, where a classifier must generalize to new classes not seen in the training set, given only a small number of examples of each new class. Prototypical networks learn a metric space in which classification can be performed by computing distances to prototype representations of each class. Compared to recent approaches for few-shot learning, they reflect a simpler inductive bias that is beneficial in this limited-data regime, and achieve excellent results. We provide an analysis showing that some simple design decisions can yield substantial improvements over recent approaches involving complicated architectural choices and meta-learning. We further extend prototypical networks to zero-shot learning and achieve state-of-the-art results on the CU-Birds dataset.
[Abstract](https://arxiv.org/abs/1703.05175), [PDF](https://arxiv.org/pdf/1703.05175)


### #393: Triple Generative Adversarial Nets
_Chongxuan Li,  Kun Xu,  Jun Zhu,  Bo Zhang_

Generative Adversarial Nets (GANs) have shown promise in image generation and semi-supervised learning (SSL). However, existing GANs in SSL have two problems: (1) the generator and discriminator may compete in learning; and (2) the generator cannot generate images in a specific class. The problems essentially arise from the two-player formulation, where a single discriminator shares incompatible roles of identifying fake samples and predicting labels and it only estimates the data without considering labels. We address the problems by presenting triple generative adversarial net (Triple-GAN), a flexible game-theoretical framework for classification and class-conditional generation in SSL. Triple-GAN consists of three players---a generator, a discriminator and a classifier, where the generator and classifier characterize the conditional distributions between images and labels, and the discriminator solely focuses on identifying fake image-label pairs. We design compatible utilities to ensure that the distributions characterized by the classifier and generator both concentrate to the data distribution. Our results on various datasets demonstrate that Triple-GAN as a unified model can simultaneously (1) achieve state-of-the-art classification results among deep generative models, and (2) disentangle the classes and styles and transfer smoothly on the data level via interpolation in the latent space class-conditionally.
[Abstract](https://arxiv.org/abs/1703.02291), [PDF](https://arxiv.org/pdf/1703.02291)


### #394: Efficient Sublinear-Regret Algorithms for Online Sparse Linear Regression

### #395: Mapping distinct timescales of functional interactions among brain networks

### #396: Multi-Armed Bandits with Metric Movement Costs

### #397: Learning A Structured Optimal Bipartite Graph for Co-Clustering

### #398: Learning Low-Dimensional Metrics
_Lalit Jain,  Blake Mason,  Robert Nowak_

This paper investigates the theoretical foundations of metric learning, focused on three key questions that are not fully addressed in prior work: 1) we consider learning general low-dimensional (low-rank) metrics as well as sparse metrics; 2) we develop upper and lower (minimax)bounds on the generalization error; 3) we quantify the sample complexity of metric learning in terms of the dimension of the feature space and the dimension/rank of the underlying metric;4) we also bound the accuracy of the learned metric relative to the underlying true generative metric. All the results involve novel mathematical approaches to the metric learning problem, and lso shed new light on the special case of ordinal embedding (aka non-metric multidimensional scaling).
[Abstract](https://arxiv.org/abs/1709.06171), [PDF](https://arxiv.org/pdf/1709.06171)


### #399: The Marginal Value of Adaptive Gradient Methods in Machine Learning
_Ashia C. Wilson,  Rebecca Roelofs,  Mitchell Stern,  Nathan Srebro,  Benjamin Recht_

Adaptive optimization methods, which perform local optimization with a metric constructed from the history of iterates, are becoming increasingly popular for training deep neural networks. Examples include AdaGrad, RMSProp, and Adam. We show that for simple overparameterized problems, adaptive methods often find drastically different solutions than gradient descent (GD) or stochastic gradient descent (SGD). We construct an illustrative binary classification problem where the data is linearly separable, GD and SGD achieve zero test error, and AdaGrad, Adam, and RMSProp attain test errors arbitrarily close to half. We additionally study the empirical generalization capability of adaptive methods on several state-of-the-art deep learning models. We observe that the solutions found by adaptive methods generalize worse (often significantly worse) than SGD, even when these solutions have better training performance. These results suggest that practitioners should reconsider the use of adaptive methods to train neural networks.
[Abstract](https://arxiv.org/abs/1705.08292), [PDF](https://arxiv.org/pdf/1705.08292)


### #400: Aggressive Sampling for Multi-class to Binary Reduction with Applications to Text Classification
_Bikash Joshi,  Massih-Reza Amini,  Ioannis Partalas,  Franck Iutzeler,  Yury Maximov_

We address the problem of multi-class classification in the case where the number of classes is very large. We propose a double sampling strategy on top of a multi-class to binary reduction strategy, which transforms the original multi-class problem into a binary classification problem over pairs of examples. The aim of the sampling strategy is to overcome the curse of long-tailed class distributions exhibited in majority of large-scale multi-class classification problems and to reduce the number of pairs of examples in the expanded data. We show that this strategy does not alter the consistency of the empirical risk minimization principle defined over the double sample reduction. Experiments are carried out on DMOZ and Wikipedia collections with 10,000 to 100,000 classes where we show the efficiency of the proposed approach in terms of training and prediction time, memory consumption, and predictive performance with respect to state-of-the-art approaches.
[Abstract](https://arxiv.org/abs/1701.06511), [PDF](https://arxiv.org/pdf/1701.06511)


### #401: Deconvolutional Paragraph Representation Learning
_Yizhe Zhang,  Dinghan Shen,  Guoyin Wang,  Zhe Gan,  Ricardo Henao,  Lawrence Carin_

Learning latent representations from long text sequences is an important first step in many natural language processing applications. Recurrent Neural Networks (RNNs) have become a cornerstone for this challenging task. However, the quality of sentences during RNN-based decoding (reconstruction) decreases with the length of the text. We propose a sequence-to-sequence, purely convolutional and deconvolutional autoencoding framework that is free of the above issue, while also being computationally efficient. The proposed method is simple, easy to implement and can be leveraged as a building block for many applications. We show empirically that compared to RNNs, our framework is better at reconstructing and correcting long paragraphs. Quantitative evaluation on semi-supervised text classification and summarization tasks demonstrate the potential for better utilization of long unlabeled text data.
[Abstract](https://arxiv.org/abs/1708.04729), [PDF](https://arxiv.org/pdf/1708.04729)


### #402: Random Permutation Online Isotonic Regression

### #403: A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning

### #404: Inverse Filtering for Hidden Markov Models

### #405: Non-parametric Neural Networks

### #406: Learning Active Learning from Data

### #407: VAE Learning via Stein Variational Gradient Descent

### #408: Deep adversarial neural decoding
_Yağmur Güçlütürk,  Umut Güçlü,  Katja Seeliger,  Sander Bosch,  Rob van Lier,  Marcel van Gerven_

Here, we present a novel approach to solve the problem of reconstructing perceived stimuli from brain responses by combining probabilistic inference with deep learning. Our approach first inverts the linear transformation from latent features to brain responses with maximum a posteriori estimation and then inverts the nonlinear transformation from perceived stimuli to latent features with adversarial training of convolutional neural networks. We test our approach with a functional magnetic resonance imaging experiment and show that it can generate state-of-the-art reconstructions of perceived faces from brain activations.
[Abstract](https://arxiv.org/abs/1705.07109), [PDF](https://arxiv.org/pdf/1705.07109)


### #409: Efficient Use of Limited-Memory Resources to Accelerate Linear Learning
_Celestine Dünner,  Thomas Parnell,  Martin Jaggi_

We propose a generic algorithmic building block to accelerate training of machine learning models on heterogenous compute systems. The scheme allows to efficiently employ compute accelerators such as GPUs and FPGAs for the training of large-scale machine learning models, when the training data exceeds their memory capacity. Also, it provides adaptivity to any system's memory hierarchy in terms of size and processing speed. Our technique builds upon primal-dual coordinate methods, and uses duality gap information to dynamically decide which part of the data should be made available for fast processing. We provide a strong theoretical motivation for our gap-based selection scheme and provide an efficient practical implementation thereof. To illustrate the power of our approach we demonstrate its performance for training of generalized linear models on large scale datasets exceeding the memory size of a modern GPU, showing an order-of-magnitude speedup over existing approaches.
[Abstract](https://arxiv.org/abs/1708.05357), [PDF](https://arxiv.org/pdf/1708.05357)


### #410: Temporal Coherency based Criteria for Predicting Video Frames using Deep Multi-stage Generative Adversarial Networks

### #411: Sobolev Training for Neural Networks
_Wojciech Marian Czarnecki,  Simon Osindero,  Max Jaderberg,  Grzegorz Świrszcz,  Razvan Pascanu_

At the heart of deep learning we aim to use neural networks as function approximators - training them to produce outputs from inputs in emulation of a ground truth function or data creation process. In many cases we only have access to input-output pairs from the ground truth, however it is becoming more common to have access to derivatives of the target output with respect to the input - for example when the ground truth function is itself a neural network such as in network compression or distillation. Generally these target derivatives are not computed, or are ignored. This paper introduces Sobolev Training for neural networks, which is a method for incorporating these target derivatives in addition the to target values while training. By optimising neural networks to not only approximate the function's outputs but also the function's derivatives we encode additional information about the target function within the parameters of the neural network. Thereby we can improve the quality of our predictors, as well as the data-efficiency and generalization capabilities of our learned function approximation. We provide theoretical justifications for such an approach as well as examples of empirical evidence on three distinct domains: regression on classical optimisation datasets, distilling policies of an agent playing Atari, and on large-scale applications of synthetic gradients. In all three domains the use of Sobolev Training, employing target derivatives in addition to target values, results in models with higher accuracy and stronger generalisation.
[Abstract](https://arxiv.org/abs/1706.04859), [PDF](https://arxiv.org/pdf/1706.04859)


### #412: Multi-Information Source Optimization
_Matthias Poloczek,  Jialei Wang,  Peter I. Frazier_

We consider Bayesian optimization of an expensive-to-evaluate black-box objective function, where we also have access to cheaper approximations of the objective. In general, such approximations arise in applications such as reinforcement learning, engineering, and the natural sciences, and are subject to an inherent, unknown bias. This model discrepancy is caused by an inadequate internal model that deviates from reality and can vary over the domain, making the utilization of these approximations a non-trivial task. We present a novel algorithm that provides a rigorous mathematical treatment of the uncertainties arising from model discrepancies and noisy observations. Its optimization decisions rely on a value of information analysis that extends the Knowledge Gradient factor to the setting of multiple information sources that vary in cost: each sampling decision maximizes the predicted benefit per unit cost. We conduct an experimental evaluation that demonstrates that the method consistently outperforms other state-of-the-art techniques: it finds designs of considerably higher objective value and additionally inflicts less cost in the exploration process.
[Abstract](https://arxiv.org/abs/1603.00389), [PDF](https://arxiv.org/pdf/1603.00389)


### #413: Deep Reinforcement Learning from Human Preferences

### #414: On the Fine-Grained Complexity of Empirical Risk Minimization: Kernel Methods and Neural Networks
_Arturs Backurs,  Piotr Indyk,  Ludwig Schmidt_

Empirical risk minimization (ERM) is ubiquitous in machine learning and underlies most supervised learning methods. While there has been a large body of work on algorithms for various ERM problems, the exact computational complexity of ERM is still not understood. We address this issue for multiple popular ERM problems including kernel SVMs, kernel ridge regression, and training the final layer of a neural network. In particular, we give conditional hardness results for these problems based on complexity-theoretic assumptions such as the Strong Exponential Time Hypothesis. Under these assumptions, we show that there are no algorithms that solve the aforementioned ERM problems to high accuracy in sub-quadratic time. We also give similar hardness results for computing the gradient of the empirical loss, which is the main computational burden in many non-convex learning tasks.
[Abstract](https://arxiv.org/abs/1704.02958), [PDF](https://arxiv.org/pdf/1704.02958)


### #415: Policy Gradient With Value Function Approximation For Collective Multiagent Planning

### #416: Adversarial Symmetric Variational Autoencoder

### #417: Tensor encoding and decomposition of brain connectomes with application to tractography evaluation

### #418: A Minimax Optimal Algorithm for Crowdsourcing

### #419: Estimating Accuracy from Unlabeled Data: A Probabilistic Logic Approach
_Emmanouil A. Platanios,  Hoifung Poon,  Tom M. Mitchell,  Eric Horvitz_

We propose an efficient method to estimate the accuracy of classifiers using only unlabeled data. We consider a setting with multiple classification problems where the target classes may be tied together through logical constraints. For example, a set of classes may be mutually exclusive, meaning that a data instance can belong to at most one of them. The proposed method is based on the intuition that: (i) when classifiers agree, they are more likely to be correct, and (ii) when the classifiers make a prediction that violates the constraints, at least one classifier must be making an error. Experiments on four real-world data sets produce accuracy estimates within a few percent of the true accuracy, using solely unlabeled data. Our models also outperform existing state-of-the-art solutions in both estimating accuracies, and combining multiple classifier outputs. The results emphasize the utility of logical constraints in estimating accuracy, thus validating our intuition.
[Abstract](https://arxiv.org/abs/1705.07086), [PDF](https://arxiv.org/pdf/1705.07086)


### #420: A Decomposition of Forecast Error in Prediction Markets
_Miroslav Dudík,  Sébastien Lahaie,  Ryan Rogers,  Jennifer Wortman Vaughan_

We introduce and analyze sources of error in prediction market forecasts in order to characterize and bound the difference between a security's price and its ground truth value. We consider cost-function-based prediction markets in which an automated market maker adjusts security prices according to the history of trade. We decompose the forecasting error into four components: \emph{sampling error}, occurring because traders only possess noisy estimates of ground truth; \emph{risk-aversion effect}, arising because traders reveal beliefs only through self-interested trade; \emph{market-maker bias}, resulting from the use of a particular market maker (i.e., cost function) to facilitate trade; and finally, \emph{convergence error}, arising because, at any point in time, market prices may still be in flux. Our goal is to understand the tradeoffs between these error components, and how they are influenced by design decisions such as the functional form of the cost function and the amount of liquidity in the market. We specifically consider a model in which traders have exponential utility and exponential-family beliefs drawn with an independent noise relative to ground truth. In this setting, sampling error and risk-aversion effect vanish as the number of traders grows, but there is a tradeoff between the other two components: decreasing the market maker's liquidity results in smaller market-maker bias, but may also slow down convergence. We provide both upper and lower bounds on market-maker bias and convergence error, and demonstrate via numerical simulations that these bounds are tight. Our results yield new insights into the question of how to set the market's liquidity parameter, and into the extent to which markets that enforce coherent prices across securities produce better predictions than markets that price securities independently.
[Abstract](https://arxiv.org/abs/1702.07810), [PDF](https://arxiv.org/pdf/1702.07810)


### #421: Safe Adaptive Importance Sampling

### #422: Variational Walkback: Learning a Transition Operator as a Stochastic Recurrent Net

### #423: Polynomial Codes: an Optimal Design for High-Dimensional Coded Matrix Multiplication
_Qian Yu,  Mohammad Ali Maddah-Ali,  A. Salman Avestimehr_

We consider a large-scale matrix multiplication problem where the computation is carried out using a distributed system with a master node and multiple worker nodes, where each worker can store parts of the input matrices. We propose a computation strategy that leverages ideas from coding theory to design intermediate computations at the worker nodes, in order to efficiently deal with straggling workers. The proposed strategy, named as \emph{polynomial codes}, achieves the optimum recovery threshold, defined as the minimum number of workers that the master needs to wait for in order to compute the output. Furthermore, by leveraging the algebraic structure of polynomial codes, we can map the reconstruction problem of the final output to a polynomial interpolation problem, which can be solved efficiently. Polynomial codes provide order-wise improvement over the state of the art in terms of recovery threshold, and are also optimal in terms of several other metrics. Furthermore, we extend this code to distributed convolution and show its order-wise optimality.
[Abstract](https://arxiv.org/abs/1705.10464), [PDF](https://arxiv.org/pdf/1705.10464)


### #424: Unsupervised Learning of Disentangled Representations from Video
_Emily Denton,  Vighnesh Birodkar_

We present a new model DrNET that learns disentangled image representations from video. Our approach leverages the temporal coherence of video and a novel adversarial loss to learn a representation that factorizes each frame into a stationary part and a temporally varying component. The disentangled representation can be used for a range of tasks. For example, applying a standard LSTM to the time-vary components enables prediction of future frames. We evaluate our approach on a range of synthetic and real videos, demonstrating the ability to coherently generate hundreds of steps into the future.
[Abstract](https://arxiv.org/abs/1705.10915), [PDF](https://arxiv.org/pdf/1705.10915)


### #425: Federated Multi-Task Learning
_Virginia Smith,  Chao-Kai Chiang,  Maziar Sanjabi,  Ameet Talwalkar_

Federated learning poses new statistical and systems challenges in training machine learning models over distributed networks of devices. In this work, we show that multi-task learning is naturally suited to handle the statistical challenges of this setting, and propose a novel systems-aware optimization method, MOCHA, that is robust to practical systems issues. Our method and theory for the first time consider issues of high communication cost, stragglers, and fault tolerance for distributed multi-task learning. The resulting method achieves significant speedups compared to alternatives in the federated setting, as we demonstrate through simulations on real-world federated datasets.
[Abstract](https://arxiv.org/abs/1705.10467), [PDF](https://arxiv.org/pdf/1705.10467)


### #426: Is Input Sparsity Time Possible for Kernel Low-Rank Approximation?

### #427: The Expxorcist: Nonparametric Graphical Models Via Conditional Exponential Densities

### #428: Improved Graph Laplacian via Geometric Self-Consistency

### #429: Dual Path Networks
_Yunpeng Chen,  Jianan Li,  Huaxin Xiao,  Xiaojie Jin,  Shuicheng Yan,  Jiashi Feng_

In this work, we present a simple, highly efficient and modularized Dual Path Network (DPN) for image classification which presents a new topology of connection paths internally. By revealing the equivalence of the state-of-the-art Residual Network (ResNet) and Densely Convolutional Network (DenseNet) within the HORNN framework, we find that ResNet enables feature re-usage while DenseNet enables new features exploration which are both important for learning good representations. To enjoy the benefits from both path topologies, our proposed Dual Path Network shares common features while maintaining the flexibility to explore new features through dual path architectures. Extensive experiments on three benchmark datasets, ImagNet-1k, Places365 and PASCAL VOC, clearly demonstrate superior performance of the proposed DPN over state-of-the-arts. In particular, on the ImagNet-1k dataset, a shallow DPN surpasses the best ResNeXt-101(64x4d) with 26% smaller model size, 25% less computational cost and 8% lower memory consumption, and a deeper DPN (DPN-131) further pushes the state-of-the-art single model performance with about 2 times faster training speed. Experiments on the Places365 large-scale scene dataset, PASCAL VOC detection dataset, and PASCAL VOC segmentation dataset also demonstrate its consistently better performance than DenseNet, ResNet and the latest ResNeXt model over various applications.
[Abstract](https://arxiv.org/abs/1707.01629), [PDF](https://arxiv.org/pdf/1707.01629)


### #430: Faster and Non-ergodic O(1/K) Stochastic Alternating Direction Method of Multipliers

### #431: A Probabilistic Framework for Nonlinearities in Stochastic Neural Networks
_Qinliang Su,  Xuejun Liao,  Lawrence Carin_

We present a probabilistic framework for nonlinearities, based on doubly truncated Gaussian distributions. By setting the truncation points appropriately, we are able to generate various types of nonlinearities within a unified framework, including sigmoid, tanh and ReLU, the most commonly used nonlinearities in neural networks. The framework readily integrates into existing stochastic neural networks (with hidden units characterized as random variables), allowing one for the first time to learn the nonlinearities alongside model weights in these networks. Extensive experiments demonstrate the performance improvements brought about by the proposed framework when integrated with the restricted Boltzmann machine (RBM), temporal RBM and the truncated Gaussian graphical model (TGGM).
[Abstract](https://arxiv.org/abs/1709.06123), [PDF](https://arxiv.org/pdf/1709.06123)


### #432: DisTraL: Robust multitask reinforcement learning

### #433: Online Learning of Optimal Bidding Strategy in Repeated Multi-Commodity Auctions
_Sevi Baltaoglu,  Lang Tong,  Qing Zhao_

We study the online learning problem of a bidder who participates in repeated auctions. With the goal of maximizing his total T-period payoff, the bidder wants to determine the optimal allocation of his fixed budget among his bids for $K$ different goods at each period. As a bidding strategy, we propose a polynomial time algorithm, referred to as dynamic programming on discrete set (DPDS), which is inspired by the dynamic programming approach to Knapsack problems. We show that DPDS achieves the regret order of $O(\sqrt{T\log{T}})$. Also, by showing that the regret growth rate is lower bounded by $\Omega(\sqrt{T})$ for any bidding strategy, we conclude that DPDS algorithm is order optimal up to a $\sqrt{\log{T}}$ term. We also evaluate the performance of DPDS empirically in the context of virtual bidding in wholesale electricity markets by using historical data from the New York energy market.
[Abstract](https://arxiv.org/abs/1703.02567), [PDF](https://arxiv.org/pdf/1703.02567)


### #434: Trimmed Density Ratio Estimation

### #435: Training recurrent networks to generate hypotheses about how the brain solves hard navigation problems
_Ingmar Kanitscheider,  Ila Fiete_

Self-localization during navigation with noisy sensors in an ambiguous world is computationally challenging, yet animals and humans excel at it. In robotics, Simultaneous Location and Mapping (SLAM) algorithms solve this problem though joint sequential probabilistic inference of their own coordinates and those of external spatial landmarks. We generate the first neural solution to the SLAM problem by training recurrent LSTM networks to perform a set of hard 2D navigation tasks that include generalization to completely novel trajectories and environments. The hidden unit representations exhibit several key properties of hippocampal place cells, including stable tuning curves that remap between environments. Our result is also a proof of concept for end-to-end-learning of a SLAM algorithm using recurrent networks, and a demonstration of why this approach may have some advantages for robotic SLAM.
[Abstract](https://arxiv.org/abs/1609.09059), [PDF](https://arxiv.org/pdf/1609.09059)


### #436: Visual Interaction Networks
_Nicholas Watters,  Andrea Tacchetti,  Theophane Weber,  Razvan Pascanu,  Peter Battaglia,  Daniel Zoran_

From just a glance, humans can make rich predictions about the future state of a wide range of physical systems. On the other hand, modern approaches from engineering, robotics, and graphics are often restricted to narrow domains and require direct measurements of the underlying states. We introduce the Visual Interaction Network, a general-purpose model for learning the dynamics of a physical system from raw visual observations. Our model consists of a perceptual front-end based on convolutional neural networks and a dynamics predictor based on interaction networks. Through joint training, the perceptual front-end learns to parse a dynamic visual scene into a set of factored latent object representations. The dynamics predictor learns to roll these states forward in time by computing their interactions and dynamics, producing a predicted physical trajectory of arbitrary length. We found that from just six input video frames the Visual Interaction Network can generate accurate future trajectories of hundreds of time steps on a wide range of physical systems. Our model can also be applied to scenes with invisible objects, inferring their future states from their effects on the visible objects, and can implicitly infer the unknown mass of objects. Our results demonstrate that the perceptual module and the object-based dynamics predictor module can induce factored latent representations that support accurate dynamical predictions. This work opens new opportunities for model-based decision-making and planning from raw sensory observations in complex physical environments.
[Abstract](https://arxiv.org/abs/1706.01433), [PDF](https://arxiv.org/pdf/1706.01433)


### #437: Reconstruct & Crush Network
_Huijie Yang,  Wenxu Wang,  Tao Zhou,  Binghong ang,  Fangcui Zhao_

A number of recent works have concentrated on a few statistical properties of complex networks, such as the clustering, the right-skewed degree distribution and the community, which are common to many real world networks. In this paper, we address the hierarchy property sharing among a large amount of networks. Based upon the eigenvector centrality (EC) measure, a method is proposed to reconstruct the hierarchical structure of a complex network. It is tested on the Santa Fe Institute collaboration network, whose structure is well known. We also apply it to a Mathematicians' collaboration network and the protein interaction network of Yeast. The method can detect significantly hierarchical structures in these networks.
[Abstract](https://arxiv.org/abs/physics/0508026), [PDF](https://arxiv.org/pdf/physics/0508026)


### #438: Streaming Robust Submodular Maximization:A Partitioned Thresholding Approach

### #439: Simple strategies for recovering inner products from coarsely quantized random projections

### #440: Discovering Potential Influence via Information Bottleneck

### #441: Doubly Stochastic Variational Inference for Deep Gaussian Processes
_Hugh Salimbeni,  Marc Deisenroth_

Gaussian processes (GPs) are a good choice for function approximation as they are flexible, robust to over-fitting, and provide well-calibrated predictive uncertainty. Deep Gaussian processes (DGPs) are multi-layer generalisations of GPs, but inference in these models has proved challenging. Existing approaches to inference in DGP models assume approximate posteriors that force independence between the layers, and do not work well in practice. We present a doubly stochastic variational inference algorithm, which does not force independence between layers. With our method of inference we demonstrate that a DGP model can be used effectively on data ranging in size from hundreds to a billion points. We provide strong empirical evidence that our inference scheme for DGPs works well in practice in both classification and regression.
[Abstract](https://arxiv.org/abs/1705.08933), [PDF](https://arxiv.org/pdf/1705.08933)


### #442: Ranking Data with Continuous Labels through Oriented Recursive Partitions

### #443: Scalable Model Selection for Belief Networks

### #444: Targeting EEG/LFP Synchrony with Neural Nets

### #445: Near-Optimal Edge Evaluation in Explicit Generalized Binomial Graphs
_Sanjiban Choudhury,  Shervin Javdani,  Siddhartha Srinivasa,  Sebastian Scherer_

Robotic motion-planning problems, such as a UAV flying fast in a partially-known environment or a robot arm moving around cluttered objects, require finding collision-free paths quickly. Typically, this is solved by constructing a graph, where vertices represent robot configurations and edges represent potentially valid movements of the robot between these configurations. The main computational bottlenecks are expensive edge evaluations to check for collisions. State of the art planning methods do not reason about the optimal sequence of edges to evaluate in order to find a collision free path quickly. In this paper, we do so by drawing a novel equivalence between motion planning and the Bayesian active learning paradigm of decision region determination (DRD). Unfortunately, a straight application of existing methods requires computation exponential in the number of edges in a graph. We present BISECT, an efficient and near-optimal algorithm to solve the DRD problem when edges are independent Bernoulli random variables. By leveraging this property, we are able to significantly reduce computational complexity from exponential to linear in the number of edges. We show that BISECT outperforms several state of the art algorithms on a spectrum of planning problems for mobile robots, manipulators, and real flight data collected from a full scale helicopter.
[Abstract](https://arxiv.org/abs/1706.09351), [PDF](https://arxiv.org/pdf/1706.09351)


### #446: Non-Stationary Spectral Kernels
_Sami Remes,  Markus Heinonen,  Samuel Kaski_

We propose non-stationary spectral kernels for Gaussian process regression. We propose to model the spectral density of a non-stationary kernel function as a mixture of input-dependent Gaussian process frequency density surfaces. We solve the generalised Fourier transform with such a model, and present a family of non-stationary and non-monotonic kernels that can learn input-dependent and potentially long-range, non-monotonic covariances between inputs. We derive efficient inference using model whitening and marginalized posterior, and show with case studies that these kernels are necessary when modelling even rather simple time series, image or geospatial data with non-stationary characteristics.
[Abstract](https://arxiv.org/abs/1705.08736), [PDF](https://arxiv.org/pdf/1705.08736)


### #447: Overcoming Catastrophic Forgetting by Incremental Moment Matching
_Sang-Woo Lee,  Jin-Hwa Kim,  Jaehyun Jun,  Jung-Woo Ha,  Byoung-Tak Zhang_

Catastrophic forgetting is a problem of neural networks that loses the information of the first task after training the second task. Here, we propose incremental moment matching (IMM) to resolve this problem. IMM incrementally matches the moment of the posterior distribution of neural networks, which is trained for the first and the second task, respectively. To make the search space of posterior parameter smooth, the IMM procedure is complemented by various transfer learning techniques including weight transfer, L2-norm of the old and the new parameter, and a variant of dropout with the old parameter. We analyze our approach on various datasets including the MNIST, CIFAR-10, Caltech-UCSD-Birds, and Lifelog datasets. Experimental results show that IMM achieves state-of-the-art performance in a variety of datasets and can balance the information between an old and a new network.
[Abstract](https://arxiv.org/abs/1703.08475), [PDF](https://arxiv.org/pdf/1703.08475)


### #448: Balancing information exposure in social networks

### #449: SafetyNets: Verifiable Execution of Deep Neural Networks on an Untrusted Cloud
_Zahra Ghodsi,  Tianyu Gu,  Siddharth Garg_

Inference using deep neural networks is often outsourced to the cloud since it is a computationally demanding task. However, this raises a fundamental issue of trust. How can a client be sure that the cloud has performed inference correctly? A lazy cloud provider might use a simpler but less accurate model to reduce its own computational load, or worse, maliciously modify the inference results sent to the client. We propose SafetyNets, a framework that enables an untrusted server (the cloud) to provide a client with a short mathematical proof of the correctness of inference tasks that they perform on behalf of the client. Specifically, SafetyNets develops and implements a specialized interactive proof (IP) protocol for verifiable execution of a class of deep neural networks, i.e., those that can be represented as arithmetic circuits. Our empirical results on three- and four-layer deep neural networks demonstrate the run-time costs of SafetyNets for both the client and server are low. SafetyNets detects any incorrect computations of the neural network by the untrusted server with high probability, while achieving state-of-the-art accuracy on the MNIST digit recognition (99.4%) and TIMIT speech recognition tasks (75.22%).
[Abstract](https://arxiv.org/abs/1706.10268), [PDF](https://arxiv.org/pdf/1706.10268)


### #450: Query Complexity of Clustering with Side Information
_Arya Mazumdar,  Barna Saha_

Suppose, we are given a set of $n$ elements to be clustered into $k$ (unknown) clusters, and an oracle/expert labeler that can interactively answer pair-wise queries of the form, "do two elements $u$ and $v$ belong to the same cluster?". The goal is to recover the optimum clustering by asking the minimum number of queries. In this paper, we initiate a rigorous theoretical study of this basic problem of query complexity of interactive clustering, and provide strong information theoretic lower bounds, as well as nearly matching upper bounds. Most clustering problems come with a similarity matrix, which is used by an automated process to cluster similar points together. Our main contribution in this paper is to show the dramatic power of side information aka similarity matrix on reducing the query complexity of clustering. A similarity matrix represents noisy pair-wise relationships such as one computed by some function on attributes of the elements. A natural noisy model is where similarity values are drawn independently from some arbitrary probability distribution $f_+$ when the underlying pair of elements belong to the same cluster, and from some $f_-$ otherwise. We show that given such a similarity matrix, the query complexity reduces drastically from $\Theta(nk)$ (no similarity matrix) to $O(\frac{k^2\log{n}}{\cH^2(f_+\|f_-)})$ where $\cH^2$ denotes the squared Hellinger divergence. Moreover, this is also information-theoretic optimal within an $O(\log{n})$ factor. Our algorithms are all efficient, and parameter free, i.e., they work without any knowledge of $k, f_+$ and $f_-$, and only depend logarithmically with $n$. Along the way, our work also reveals intriguing connection to popular community detection models such as the {\em stochastic block model}, significantly generalizes them, and opens up many venues for interesting future research.
[Abstract](https://arxiv.org/abs/1706.07719), [PDF](https://arxiv.org/pdf/1706.07719)


### #451: QMDP-Net: Deep Learning for Planning under Partial Observability
_Peter Karkus,  David Hsu,  Wee Sun Lee_

This paper introduces the QMDP-net, a neural network architecture for planning under partial observability. The QMDP-net combines the strengths of model-free learning and model-based planning. It is a recurrent policy network, but it represents a policy by connecting a model with a planning algorithm that solves the model, thus embedding the solution structure of planning in a network learning architecture. The QMDP-net is fully differentiable and allows end-to-end training. We train a QMDP-net in a set of different environments so that it can generalize over new ones and "transfer" to larger environments as well. In preliminary experiments, QMDP-net showed strong performance on several robotic tasks in simulation. Interestingly, while QMDP-net encodes the QMDP algorithm, it sometimes outperforms the QMDP algorithm in the experiments, because of QMDP-net's increased robustness through end-to-end learning.
[Abstract](https://arxiv.org/abs/1703.06692), [PDF](https://arxiv.org/pdf/1703.06692)


### #452: Robust Optimization for Non-Convex Objectives
_Robert Chen,  Brendan Lucier,  Yaron Singer,  Vasilis Syrgkanis_

We consider robust optimization problems, where the goal is to optimize in the worst case over a class of objective functions. We develop a reduction from robust improper optimization to Bayesian optimization: given an oracle that returns $\alpha$-approximate solutions for distributions over objectives, we compute a distribution over solutions that is $\alpha$-approximate in the worst case. We show that de-randomizing this solution is NP-hard in general, but can be done for a broad class of statistical learning tasks. We apply our results to robust neural network training and submodular optimization. We evaluate our approach experimentally on corrupted character classification, and robust influence maximization in networks.
[Abstract](https://arxiv.org/abs/1707.01047), [PDF](https://arxiv.org/pdf/1707.01047)


### #453: Thy Friend is My Friend: Iterative Collaborative Filtering for Sparse Matrix Estimation

### #454: Adaptive Classification for Prediction Under a Budget
_Feng Nan,  Venkatesh Saligrama_

We propose a novel adaptive approximation approach for test-time resource-constrained prediction. Given an input instance at test-time, a gating function identifies a prediction model for the input among a collection of models. Our objective is to minimize overall average cost without sacrificing accuracy. We learn gating and prediction models on fully labeled training data by means of a bottom-up strategy. Our novel bottom-up method first trains a high-accuracy complex model. Then a low-complexity gating and prediction model are subsequently learned to adaptively approximate the high-accuracy model in regions where low-cost models are capable of making highly accurate predictions. We pose an empirical loss minimization problem with cost constraints to jointly train gating and prediction models. On a number of benchmark datasets our method outperforms state-of-the-art achieving higher accuracy for the same cost.
[Abstract](https://arxiv.org/abs/1705.10194), [PDF](https://arxiv.org/pdf/1705.10194)


### #455: Convergence rates of a partition based Bayesian multivariate density estimation method

### #456: Affine-Invariant Online Optimization

### #457: Beyond Worst-case: A Probabilistic Analysis of Affine Policies in Dynamic Optimization 
_Omar El Housni,  Vineet Goyal_

Affine policies (or control) are widely used as a solution approach in dynamic optimization where computing an optimal adjustable solution is usually intractable. While the worst case performance of affine policies can be significantly bad, the empirical performance is observed to be near-optimal for a large class of problem instances. For instance, in the two-stage dynamic robust optimization problem with linear covering constraints and uncertain right hand side, the worst-case approximation bound for affine policies is $O(\sqrt m)$ that is also tight (see Bertsimas and Goyal (2012)), whereas observed empirical performance is near-optimal. In this paper, we aim to address this stark-contrast between the worst-case and the empirical performance of affine policies. In particular, we show that affine policies give a good approximation for the two-stage adjustable robust optimization problem with high probability on random instances where the constraint coefficients are generated i.i.d. from a large class of distributions; thereby, providing a theoretical justification of the observed empirical performance. On the other hand, we also present a distribution such that the performance bound for affine policies on instances generated according to that distribution is $\Omega(\sqrt m)$ with high probability; however, the constraint coefficients are not i.i.d.. This demonstrates that the empirical performance of affine policies can depend on the generative model for instances.
[Abstract](https://arxiv.org/abs/1706.05737), [PDF](https://arxiv.org/pdf/1706.05737)


### #458: A unified approach to interpreting model predictions
_Scott Lundberg,  Su-In Lee_

Understanding why a model made a certain prediction is crucial in many applications. However, with large modern datasets the best accuracy is often achieved by complex models even experts struggle to interpret, such as ensemble or deep learning models. This creates a tension between accuracy and interpretability. In response, a variety of methods have recently been proposed to help users interpret the predictions of complex models. Here, we present a unified framework for interpreting predictions, namely SHAP (SHapley Additive exPlanations, which assigns each feature an importance for a particular prediction. The key novel components of the SHAP framework are the identification of a class of additive feature importance measures and theoretical results that there is a unique solution in this class with a set of desired properties. This class unifies six existing methods, and several recent methods in this class do not have these desired properties. This means that our framework can inform the development of new methods for explaining prediction models. We demonstrate that several new methods we presented in this paper based on the SHAP framework show better computational performance and better consistency with human intuition than existing methods.
[Abstract](https://arxiv.org/abs/1705.07874), [PDF](https://arxiv.org/pdf/1705.07874)


### #459: Stochastic Approximation for Canonical Correlation Analysis
_Raman Arora,  Teodor V. Marinov,  Poorya Mianjy_

We study canonical correlation analysis (CCA) as a stochastic optimization problem. We show that regularized CCA is efficiently PAC-learnable. We give stochastic approximation (SA) algorithms that are instances of stochastic mirror descent, which achieve $\epsilon$-suboptimality in the population objective in time $\operatorname{poly}(\frac{1}{\epsilon},\frac{1}{\delta},d)$ with probability $1-\delta$, where $d$ is the input dimensionality.
[Abstract](https://arxiv.org/abs/1702.06818), [PDF](https://arxiv.org/pdf/1702.06818)


### #460: Investigating the learning dynamics of deep neural networks using random matrix theory

### #461: Sample and Computationally Efficient Learning Algorithms under S-Concave Distributions

### #462: Scalable Variational Inference for Dynamical Systems
_Nico S. Gorbach,  Stefan Bauer,  Joachim M. Buhmann_

Gradient matching is a promising tool for learning parameters and state dynamics of ordinary differential equations. It is a grid free inference approach which for fully observable systems is at times competitive with numerical integration. However for many real-world applications, only sparse observations are available or even unobserved variables are included in the model description. In these cases most gradient matching methods are difficult to apply or simply do not provide satisfactory results. That is why despite the high computational cost numerical integration is still the gold standard in many applications. Using an existing gradient matching approach, we propose a scalable variational inference framework, which can infer states and parameters simultaneously, offers computational speedups, improved accuracy and works well even under model misspecifications in a partially observable system.
[Abstract](https://arxiv.org/abs/1705.07079), [PDF](https://arxiv.org/pdf/1705.07079)


### #463: Context Selection for Embedding Models

### #464: Working hard to know your neighbor's margins: Local descriptor learning loss
_Anastasiya Mishchuk,  Dmytro Mishkin,  Filip Radenovic,  Jiri Matas_

We introduce a novel loss for learning local feature descriptors which is inspired by the Lowe's matching criterion for SIFT. We show that the proposed loss that maximizes the distance between the closest positive and closest negative patch in the batch is better than complex regularization methods; it works well for both shallow and deep convolution network architectures. Applying the novel loss to the L2Net CNN architecture results in a compact descriptor -- it has the same dimensionality as SIFT (128) that shows state-of-art performance in wide baseline stereo, patch verification and instance retrieval benchmarks. It is fast, computing a descriptor takes about 1 millisecond on a low-end GPU.
[Abstract](https://arxiv.org/abs/1705.10872), [PDF](https://arxiv.org/pdf/1705.10872)


### #465: Accelerated Stochastic Greedy Coordinate Descent by Soft Thresholding Projection onto Simplex

### #466: Multi-Task Learning for Contextual Bandits
_Aniket Anand Deshmukh,  Urun Dogan,  Clayton Scott_

Contextual bandits are a form of multi-armed bandit in which the agent has access to predictive side information (known as the context) for each arm at each time step, and have been used to model personalized news recommendation, ad placement, and other applications. In this work, we propose a multi-task learning framework for contextual bandit problems. Like multi-task learning in the batch setting, the goal is to leverage similarities in contexts for different arms so as to improve the agent's ability to predict rewards from contexts. We propose an upper confidence bound-based multi-task learning algorithm for contextual bandits, establish a corresponding regret bound, and interpret this bound to quantify the advantages of learning in the presence of high task (arm) similarity. We also describe an effective scheme for estimating task similarity from data, and demonstrate our algorithm's performance on several data sets.
[Abstract](https://arxiv.org/abs/1705.08618), [PDF](https://arxiv.org/pdf/1705.08618)


### #467: Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon
_Xin Dong,  Shangyu Chen,  Sinno Jialin Pan_

How to develop slim and accurate deep neural networks has become crucial for real- world applications, especially for those employed in embedded systems. Though previous work along this research line has shown some promising results, most existing methods either fail to significantly compress a well-trained deep network or require a heavy retraining process for the pruned deep network to re-boost its prediction performance. In this paper, we propose a new layer-wise pruning method for deep neural networks. In our proposed method, parameters of each individual layer are pruned independently based on second order derivatives of a layer-wise error function with respect to the corresponding parameters. We prove that the final prediction performance drop after pruning is bounded by a linear combination of the reconstructed errors caused at each layer. Therefore, there is a guarantee that one only needs to perform a light retraining process on the pruned network to resume its original prediction performance. We conduct extensive experiments on benchmark datasets to demonstrate the effectiveness of our pruning method compared with several state-of-the-art baseline methods.
[Abstract](https://arxiv.org/abs/1705.07565), [PDF](https://arxiv.org/pdf/1705.07565)


### #468: Accelerated First-order Methods for Geodesically Convex Optimization on Riemannian Manifolds

### #469: Selective Classification for Deep Neural Networks
_Yonatan Geifman,  Ran El-Yaniv_

Selective classification techniques (also known as reject option) have not yet been considered in the context of deep neural networks (DNNs). These techniques can potentially significantly improve DNNs prediction performance by trading-off coverage. In this paper we propose a method to construct a selective classifier given a trained neural network. Our method allows a user to set a desired risk level. At test time, the classifier rejects instances as needed, to grant the desired risk (with high probability). Empirical results over CIFAR and ImageNet convincingly demonstrate the viability of our method, which opens up possibilities to operate DNNs in mission-critical applications. For example, using our method an unprecedented 2% error in top-5 ImageNet classification can be guaranteed with probability 99.9%, and almost 60% test coverage.
[Abstract](https://arxiv.org/abs/1705.08500), [PDF](https://arxiv.org/pdf/1705.08500)


### #470: Minimax Estimation of Bandable Precision Matrices

### #471: Monte-Carlo Tree Search by Best Arm Identification
_Emilie Kaufmann (CNRS, CRIStAL, SEQUEL),  Wouter Koolen (CWI)_

Recent advances in bandit tools and techniques for sequential learning are steadily enabling new applications and are promising the resolution of a range of challenging related problems. We study the game tree search problem, where the goal is to quickly identify the optimal move in a given game tree by sequentially sampling its stochastic payoffs. We develop new algorithms for trees of arbitrary depth, that operate by summarizing all deeper levels of the tree into confidence intervals at depth one, and applying a best arm identification procedure at the root. We prove new sample complexity guarantees with a refined dependence on the problem instance. We show experimentally that our algorithms outperform existing elimination-based algorithms and match previous special-purpose methods for depth-two trees.
[Abstract](https://arxiv.org/abs/1706.02986), [PDF](https://arxiv.org/pdf/1706.02986)


### #472: Group Additive Structure Identification for Kernel Nonparametric Regression

### #473: Fast, Sample-Efficient Algorithms for Structured Phase Retrieval

### #474: Hash Embeddings for Efficient Word Representations
_Dan Svenstrup,  Jonas Meinertz Hansen,  Ole Winther_

We present hash embeddings, an efficient method for representing words in a continuous vector form. A hash embedding may be seen as an interpolation between a standard word embedding and a word embedding created using a random hash function (the hashing trick). In hash embeddings each token is represented by $k$ $d$-dimensional embeddings vectors and one $k$ dimensional weight vector. The final $d$ dimensional representation of the token is the product of the two. Rather than fitting the embedding vectors for each token these are selected by the hashing trick from a shared pool of $B$ embedding vectors. Our experiments show that hash embeddings can easily deal with huge vocabularies consisting of millions of tokens. When using a hash embedding there is no need to create a dictionary before training nor to perform any kind of vocabulary pruning after training. We show that models trained using hash embeddings exhibit at least the same level of performance as models trained using regular embeddings across a wide range of tasks. Furthermore, the number of parameters needed by such an embedding is only a fraction of what is required by a regular embedding. Since standard embeddings and embeddings constructed using the hashing trick are actually just special cases of a hash embedding, hash embeddings can be considered an extension and improvement over the existing regular embedding types.
[Abstract](https://arxiv.org/abs/1709.03933), [PDF](https://arxiv.org/pdf/1709.03933)


### #475: Online Learning for Multivariate Hawkes Processes

### #476: Maximum Margin Interval Trees

### #477: DropoutNet: Addressing Cold Start in Recommender Systems

### #478: A simple neural network module for relational reasoning
_Adam Santoro,  David Raposo,  David G.T. Barrett,  Mateusz Malinowski,  Razvan Pascanu,  Peter Battaglia,  Timothy Lillicrap_

Relational reasoning is a central component of generally intelligent behavior, but has proven difficult for neural networks to learn. In this paper we describe how to use Relation Networks (RNs) as a simple plug-and-play module to solve problems that fundamentally hinge on relational reasoning. We tested RN-augmented networks on three tasks: visual question answering using a challenging dataset called CLEVR, on which we achieve state-of-the-art, super-human performance; text-based question answering using the bAbI suite of tasks; and complex reasoning about dynamic physical systems. Then, using a curated dataset called Sort-of-CLEVR we show that powerful convolutional networks do not have a general capacity to solve relational questions, but can gain this capacity when augmented with RNs. Our work shows how a deep learning architecture equipped with an RN module can implicitly discover and learn to reason about entities and their relations.
[Abstract](https://arxiv.org/abs/1706.01427), [PDF](https://arxiv.org/pdf/1706.01427)


### #479: Q-LDA: Uncovering Latent Patterns in Text-based Sequential Decision Processes

### #480: Online Reinforcement Learning in Stochastic Games

### #481: Position-based Multiple-play Multi-armed Bandit Problem with Unknown Position Bias

### #482: Active Exploration for Learning Symbolic Representations
_Garrett Andersen,  George Konidaris_

We introduce an online active exploration algorithm for data-efficiently learning an abstract symbolic model of an environment. Our algorithm is divided into two parts: the first part quickly generates an intermediate Bayesian symbolic model from the data that the agent has collected so far, which the agent can then use along with the second part to guide its future exploration towards regions of the state space that the model is uncertain about. We show that our algorithm outperforms random and greedy exploration policies on two different computer game domains. The first domain is an Asteroids-inspired game with complex dynamics, but basic logical structure. The second is the Treasure Game, with simpler dynamics, but more complex logical structure.
[Abstract](https://arxiv.org/abs/1709.01490), [PDF](https://arxiv.org/pdf/1709.01490)


### #483: Clone MCMC: Parallel High-Dimensional Gaussian Gibbs Sampling

### #484: Fair Clustering Through Fairlets

### #485: Polynomial time algorithms for dual volume sampling

### #486: Hindsight Experience Replay
_Marcin Andrychowicz,  Filip Wolski,  Alex Ray,  Jonas Schneider,  Rachel Fong,  Peter Welinder,  Bob McGrew,  Josh Tobin,  Pieter Abbeel,  Wojciech Zaremba_

Dealing with sparse rewards is one of the biggest challenges in Reinforcement Learning (RL). We present a novel technique called Hindsight Experience Replay which allows sample-efficient learning from rewards which are sparse and binary and therefore avoid the need for complicated reward engineering. It can be combined with an arbitrary off-policy RL algorithm and may be seen as a form of implicit curriculum. We demonstrate our approach on the task of manipulating objects with a robotic arm. In particular, we run experiments on three different tasks: pushing, sliding, and pick-and-place, in each case using only binary rewards indicating whether or not the task is completed. Our ablation studies show that Hindsight Experience Replay is a crucial ingredient which makes training possible in these challenging environments. We show that our policies trained on a physics simulation can be deployed on a physical robot and successfully complete the task.
[Abstract](https://arxiv.org/abs/1707.01495), [PDF](https://arxiv.org/pdf/1707.01495)


### #487: Stochastic and Adversarial Online Learning without Hyperparameters

### #488: Teaching Machines to Describe Images with Natural Language Feedback

### #489: Perturbative Black Box Variational Inference
_Robert Bamler,  Cheng Zhang,  Manfred Opper,  Stephan Mandt_

Black box variational inference (BBVI) with reparameterization gradients triggered the exploration of divergence measures other than the Kullback-Leibler (KL) divergence, such as alpha divergences. These divergences can be tuned to be more mass-covering (preventing overfitting in complex models), but are also often harder to optimize using Monte-Carlo gradients. In this paper, we view BBVI with generalized divergences as a form of biased importance sampling. The choice of divergence determines a bias-variance tradeoff between the tightness of the bound (low bias) and the variance of its gradient estimators. Drawing on variational perturbation theory of statistical physics, we use these insights to construct a new variational bound which is tighter than the KL bound and more mass covering. Compared to alpha-divergences, its reparameterization gradients have a lower variance. We show in several experiments on Gaussian Processes and Variational Autoencoders that the resulting posterior covariances are closer to the true posterior and lead to higher likelihoods on held-out data.
[Abstract](https://arxiv.org/abs/1709.07433), [PDF](https://arxiv.org/pdf/1709.07433)


### #490: GibbsNet: Iterative Adversarial Inference for Deep Graphical Models

### #491: PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

### #492: Regularizing Deep Neural Networks by Noise: Its Interpretation and Optimization

### #493: Learning Graph Embeddings with Embedding Propagation

### #494: Efficient Modeling of Latent Information in Supervised Learning using Gaussian Processes
_Zhenwen Dai,  Mauricio A. Álvarez,  Neil D. Lawrence_

Often in machine learning, data are collected as a combination of multiple conditions, e.g., the voice recordings of multiple persons, each labeled with an ID. How could we build a model that captures the latent information related to these conditions and generalize to a new one with few data? We present a new model called Latent Variable Multiple Output Gaussian Processes (LVMOGP) and that allows to jointly model multiple conditions for regression and generalize to a new condition with a few data points at test time. LVMOGP infers the posteriors of Gaussian processes together with a latent space representing the information about different conditions. We derive an efficient variational inference method for LVMOGP, of which the computational complexity is as low as sparse Gaussian processes. We show that LVMOGP significantly outperforms related Gaussian process methods on various tasks with both synthetic and real data.
[Abstract](https://arxiv.org/abs/1705.09862), [PDF](https://arxiv.org/pdf/1705.09862)


### #495: A-NICE-MC: Adversarial Training for MCMC
_Jiaming Song,  Shengjia Zhao,  Stefano Ermon_

Existing Markov Chain Monte Carlo (MCMC) methods are either based on general-purpose and domain-agnostic schemes which can lead to slow convergence, or hand-crafting of problem-specific proposals by an expert. We propose A-NICE-MC, a novel method to train flexible parametric Markov chain kernels to produce samples with desired properties. First, we propose an efficient likelihood-free adversarial training method to train a Markov chain and mimic a given data distribution. Then, we leverage flexible volume preserving flows to obtain parametric kernels for MCMC. Using a bootstrap approach, we show how to train efficient Markov chains to sample from a prescribed posterior distribution by iteratively improving the quality of both the model and the samples. A-NICE-MC provides the first framework to automatically design efficient domain-specific MCMC proposals. Empirical results demonstrate that A-NICE-MC combines the strong guarantees of MCMC with the expressiveness of deep neural networks, and is able to significantly outperform competing methods such as Hamiltonian Monte Carlo.
[Abstract](https://arxiv.org/abs/1706.07561), [PDF](https://arxiv.org/pdf/1706.07561)


### #496: Excess Risk Bounds for the Bayes Risk using Variational Inference in Latent Gaussian Models

### #497: Real-Time Bidding with Side Information

### #498: Saliency-based Sequential Image Attention with Multiset Prediction

### #499: Variational Inference for Gaussian Process Models with Linear Complexity
_Chris Lloyd,  Tom Gunter,  Michael A. Osborne,  Stephen J. Roberts_

We present the first fully variational Bayesian inference scheme for continuous Gaussian-process-modulated Poisson processes. Such point processes are used in a variety of domains, including neuroscience, geo-statistics and astronomy, but their use is hindered by the computational cost of existing inference schemes. Our scheme: requires no discretisation of the domain; scales linearly in the number of observed events; and is many orders of magnitude faster than previous sampling based approaches. The resulting algorithm is shown to outperform standard methods on synthetic examples, coal mining disaster data and in the prediction of Malaria incidences in Kenya.

### #500: K-Medoids For K-Means Seeding
_James Newling,  François Fleuret_

We run experiments showing that algorithm clarans (Ng et al., 2005) finds better K-medoids solutions than the Voronoi iteration algorithm. This finding, along with the similarity between the Voronoi iteration algorithm and Lloyd's K-means algorithm, suggests that clarans may be an effective K-means initializer. We show that this is the case, with clarans outperforming other seeding algorithms on 23/23 datasets with a mean decrease over k-means++ of 30% for initialization mse and 3% or final mse. We describe how the complexity and runtime of clarans can be improved, making it a viable initialization scheme for large datasets.
[Abstract](https://arxiv.org/abs/1609.04723), [PDF](https://arxiv.org/pdf/1609.04723)


### #501: Identifying Outlier Arms in Multi-Armed Bandit

### #502: Online Learning with Transductive Regret

### #503: Riemannian approach to batch normalization
_Minhyung Cho,  Jaehyung Lee_

Batch Normalization (BN) has proven to be an effective algorithm for deep neural network training by normalizing the input to each neuron and reducing the internal covariate shift. The space of weight vectors in the BN layer can be naturally interpreted as a Riemannian manifold, which is invariant to linear scaling of weights. Following the intrinsic geometry of this manifold provides a new learning rule that is more efficient and easier to analyze. We also propose intuitive and effective gradient clipping and regularization methods for the proposed algorithm by utilizing the geometry of the manifold. The resulting algorithm consistently outperforms the original BN on various types of network architectures and datasets.

### #504: Self-supervised Learning of Motion Capture

### #505: Triangle Generative Adversarial Networks
_Zhe Gan,  Liqun Chen,  Weiyao Wang,  Yunchen Pu,  Yizhe Zhang,  Hao Liu,  Chunyuan Li,  Lawrence Carin_

A Triangle Generative Adversarial Network ($\Delta$-GAN) is developed for semi-supervised cross-domain joint distribution matching, where the training data consists of samples from each domain, and supervision of domain correspondence is provided by only a few paired samples. $\Delta$-GAN consists of four neural networks, two generators and two discriminators. The generators are designed to learn the two-way conditional distributions between the two domains, while the discriminators implicitly define a ternary discriminative function, which is trained to distinguish real data pairs and two kinds of fake data pairs. The generators and discriminators are trained together using adversarial learning. Under mild assumptions, in theory the joint distributions characterized by the two generators concentrate to the data distribution. In experiments, three different kinds of domain pairs are considered, image-label, image-image and image-attribute pairs. Experiments on semi-supervised image classification, image-to-image translation and attribute-based image generation demonstrate the superiority of the proposed approach.
[Abstract](https://arxiv.org/abs/1709.06548), [PDF](https://arxiv.org/pdf/1709.06548)


### #506: Preserving Proximity and Global Ranking for Node Embedding

### #507: Bayesian Optimization with Gradients
_Jian Wu,  Matthias Poloczek,  Andrew Gordon Wilson,  Peter I. Frazier_

In recent years, Bayesian optimization has proven successful for global optimization of expensive-to-evaluate multimodal objective functions. However, unlike most optimization methods, Bayesian optimization typically does not use derivative information. In this paper we show how Bayesian optimization can exploit derivative information to decrease the number of objective function evaluations required for good performance. In particular, we develop a novel Bayesian optimization algorithm, the derivative-enabled knowledge-gradient (dKG), for which we show one-step Bayes-optimality, asymptotic consistency, and greater one-step value of information than is possible in the derivative-free setting. Our procedure accommodates noisy and incomplete derivative information, and comes in both sequential and batch forms. We show dKG provides state-of-the-art performance compared to a wide range of optimization procedures with and without gradients, on benchmarks including logistic regression, kernel learning, and k-nearest neighbors.
[Abstract](https://arxiv.org/abs/1703.04389), [PDF](https://arxiv.org/pdf/1703.04389)


### #508: Second-order Optimization in Deep Reinforcement Learning using Kronecker-factored Approximation

### #509: Renyi Differential Privacy Mechanisms for Posterior Sampling

### #510: Online Learning with a Hint

### #511: Identification of Gaussian Process State Space Models
_Stefanos Eleftheriadis,  Thomas F.W. Nicholson,  Marc Peter Deisenroth,  James Hensman_

The Gaussian process state space model (GPSSM) is a non-linear dynamical system, where unknown transition and/or measurement mappings are described by GPs. Most research in GPSSMs has focussed on the state estimation problem. However, the key challenge in GPSSMs has not been satisfactorily addressed yet: system identification. To address this challenge, we impose a structured Gaussian variational posterior distribution over the latent states, which is parameterised by a recognition model in the form of a bi-directional recurrent neural network. Inference with this structure allows us to recover a posterior smoothed over the entire sequence(s) of data. We provide a practical algorithm for efficiently computing a lower bound on the marginal likelihood using the reparameterisation trick. This additionally allows arbitrary kernels to be used within the GPSSM. We demonstrate that we can efficiently generate plausible future trajectories of the system we seek to model with the GPSSM, requiring only a small number of interactions with the true system.
[Abstract](https://arxiv.org/abs/1705.10888), [PDF](https://arxiv.org/pdf/1705.10888)


### #512: Robust Imitation of Diverse Behaviors
_Ziyu Wang,  Josh Merel,  Scott Reed,  Greg Wayne,  Nando de Freitas,  Nicolas Heess_

Deep generative models have recently shown great promise in imitation learning for motor control. Given enough data, even supervised approaches can do one-shot imitation learning; however, they are vulnerable to cascading failures when the agent trajectory diverges from the demonstrations. Compared to purely supervised methods, Generative Adversarial Imitation Learning (GAIL) can learn more robust controllers from fewer demonstrations, but is inherently mode-seeking and more difficult to train. In this paper, we show how to combine the favourable aspects of these two approaches. The base of our model is a new type of variational autoencoder on demonstration trajectories that learns semantic policy embeddings. We show that these embeddings can be learned on a 9 DoF Jaco robot arm in reaching tasks, and then smoothly interpolated with a resulting smooth interpolation of reaching behavior. Leveraging these policy representations, we develop a new version of GAIL that (1) is much more robust than the purely-supervised controller, especially with few demonstrations, and (2) avoids mode collapse, capturing many diverse behaviors when GAIL on its own does not. We demonstrate our approach on learning diverse gaits from demonstration on a 2D biped and a 62 DoF 3D humanoid in the MuJoCo physics environment.
[Abstract](https://arxiv.org/abs/1707.02747), [PDF](https://arxiv.org/pdf/1707.02747)


### #513: Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent
_Xiangru Lian,  Ce Zhang,  Huan Zhang,  Cho-Jui Hsieh,  Wei Zhang,  Ji Liu_

Most distributed machine learning systems nowadays, including TensorFlow and CNTK, are built in a centralized fashion. One bottleneck of centralized algorithms lies on high communication cost on the central node. Motivated by this, we ask, can decentralized algorithms be faster than its centralized counterpart? Although decentralized PSGD (D-PSGD) algorithms have been studied by the control community, existing analysis and theory do not show any advantage over centralized PSGD (C-PSGD) algorithms, simply assuming the application scenario where only the decentralized network is available. In this paper, we study a D-PSGD algorithm and provide the first theoretical analysis that indicates a regime in which decentralized algorithms might outperform centralized algorithms for distributed stochastic gradient descent. This is because D-PSGD has comparable total computational complexities to C-PSGD but requires much less communication cost on the busiest node. We further conduct an empirical study to validate our theoretical analysis across multiple frameworks (CNTK and Torch), different network configurations, and computation platforms up to 112 GPUs. On network configurations with low bandwidth or high latency, D-PSGD can be up to one order of magnitude faster than its well-optimized centralized counterparts.
[Abstract](https://arxiv.org/abs/1705.09056), [PDF](https://arxiv.org/pdf/1705.09056)


### #514: Local Aggregative Games

### #515: A Sample Complexity Measure with Applications to Learning Optimal Auctions
_Vasilis Syrgkanis_

We introduce a new sample complexity measure, which we refer to as split-sample growth rate. For any hypothesis $H$ and for any sample $S$ of size $m$, the split-sample growth rate $\hat{\tau}_H(m)$ counts how many different hypotheses can empirical risk minimization output on any sub-sample of $S$ of size $m/2$. We show that the expected generalization error is upper bounded by $O\left(\sqrt{\frac{\log(\hat{\tau}_H(2m))}{m}}\right)$. Our result is enabled by a strengthening of the Rademacher complexity analysis of the expected generalization error. We show that this sample complexity measure, greatly simplifies the analysis of the sample complexity of optimal auction design, for many auction classes studied in the literature. Their sample complexity can be derived solely by noticing that in these auction classes, ERM on any sample or sub-sample will pick parameters that are equal to one of the points in the sample.
[Abstract](https://arxiv.org/abs/1704.02598), [PDF](https://arxiv.org/pdf/1704.02598)


### #516: Thinking Fast and Slow with Deep Learning and Tree Search
_Thomas Anthony,  Zheng Tian,  David Barber_

Solving sequential decision making problems, such as text parsing, robotic control, and game playing, requires a combination of planning policies and generalisation of those plans. In this paper, we present Expert Iteration, a novel algorithm which decomposes the problem into separate planning and generalisation tasks. Planning new policies is performed by tree search, while a deep neural network generalises those plans. In contrast, standard Deep Reinforcement Learning algorithms rely on a neural network not only to generalise plans, but to discover them too. We show that our method substantially outperforms Policy Gradients in the board game Hex, winning 84.4% of games against it when trained for equal time.
[Abstract](https://arxiv.org/abs/1705.08439), [PDF](https://arxiv.org/pdf/1705.08439)


### #517: EEG-GRAPH: A Factor Graph Based Model for Capturing Spatial, Temporal, and Observational Relationships in Electroencephalograms

### #518: Improving the Expected Improvement Algorithm
_Chao Qin,  Diego Klabjan,  Daniel Russo_

The expected improvement (EI) algorithm is a popular strategy for information collection in optimization under uncertainty. The algorithm is widely known to be too greedy, but nevertheless enjoys wide use due to its simplicity and ability to handle uncertainty and noise in a coherent decision theoretic framework. To provide rigorous insight into EI, we study its properties in a simple setting of Bayesian optimization where the domain consists of a finite grid of points. This is the so-called best-arm identification problem, where the goal is to allocate measurement effort wisely to confidently identify the best arm using a small number of measurements. In this framework, one can show formally that EI is far from optimal. To overcome this shortcoming, we introduce a simple modification of the expected improvement algorithm. Surprisingly, this simple change results in an algorithm that is asymptotically optimal for Gaussian best-arm identification problems, and provably outperforms standard EI by an order of magnitude.
[Abstract](https://arxiv.org/abs/1705.10033), [PDF](https://arxiv.org/pdf/1705.10033)


### #519: Hybrid Reward Architecture for Reinforcement Learning
_Harm van Seijen,  Mehdi Fatemi,  Joshua Romoff,  Romain Laroche,  Tavian Barnes,  Jeffrey Tsang_

One of the main challenges in reinforcement learning (RL) is generalisation. In typical deep RL methods this is achieved by approximating the optimal value function with a low-dimensional representation using a deep network. While this approach works well in many domains, in domains where the optimal value function cannot easily be reduced to a low-dimensional representation, learning can be very slow and unstable. This paper contributes towards tackling such challenging domains, by proposing a new method, called Hybrid Reward Architecture (HRA). HRA takes as input a decomposed reward function and learns a separate value function for each component reward function. Because each component typically only depends on a subset of all features, the overall value function is much smoother and can be easier approximated by a low-dimensional representation, enabling more effective learning. We demonstrate HRA on a toy-problem and the Atari game Ms. Pac-Man, where HRA achieves above-human performance.
[Abstract](https://arxiv.org/abs/1706.04208), [PDF](https://arxiv.org/pdf/1706.04208)


### #520: Approximate Supermodularity Bounds for Experimental Design

### #521: Maximizing Subset Accuracy with Recurrent Neural Networks in Multi-label Classification

### #522: AdaGAN: Boosting Generative Models
_Ilya Tolstikhin,  Sylvain Gelly,  Olivier Bousquet,  Carl-Johann Simon-Gabriel,  Bernhard Schölkopf_

Generative Adversarial Networks (GAN) (Goodfellow et al., 2014) are an effective method for training generative models of complex data such as natural images. However, they are notoriously hard to train and can suffer from the problem of missing modes where the model is not able to produce examples in certain regions of the space. We propose an iterative procedure, called AdaGAN, where at every step we add a new component into a mixture model by running a GAN algorithm on a reweighted sample. This is inspired by boosting algorithms, where many potentially weak individual predictors are greedily aggregated to form a strong composite predictor. We prove that such an incremental procedure leads to convergence to the true distribution in a finite number of steps if each step is optimal, and convergence at an exponential rate otherwise. We also illustrate experimentally that this procedure addresses the problem of missing modes.
[Abstract](https://arxiv.org/abs/1701.02386), [PDF](https://arxiv.org/pdf/1701.02386)


### #523: Straggler Mitigation in Distributed Optimization Through Data Encoding

### #524: Multi-View Decision Processes

### #525: A Greedy Approach for Budgeted Maximum Inner Product Search
_Hsiang-Fu Yu,  Cho-Jui Hsieh,  Qi Lei,  Inderjit S. Dhillon_

Maximum Inner Product Search (MIPS) is an important task in many machine learning applications such as the prediction phase of a low-rank matrix factorization model for a recommender system. There have been some works on how to perform MIPS in sub-linear time recently. However, most of them do not have the flexibility to control the trade-off between search efficient and search quality. In this paper, we study the MIPS problem with a computational budget. By carefully studying the problem structure of MIPS, we develop a novel Greedy-MIPS algorithm, which can handle budgeted MIPS by design. While simple and intuitive, Greedy-MIPS yields surprisingly superior performance compared to state-of-the-art approaches. As a specific example, on a candidate set containing half a million vectors of dimension 200, Greedy-MIPS runs 200x faster than the naive approach while yielding search results with the top-5 precision greater than 75\%.
[Abstract](https://arxiv.org/abs/1610.03317), [PDF](https://arxiv.org/pdf/1610.03317)


### #526: SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks

### #527: Plan, Attend, Generate: Planning for Sequence-to-Sequence Models

### #528: Task-based End-to-end Model Learning in Stochastic Optimization

### #529: Towards Understanding Adversarial Learning for Joint Distribution Matching
_Chunyuan Li,  Hao Liu,  Changyou Chen,  Yunchen Pu,  Liqun Chen,  Ricardo Henao,  Lawrence Carin_

We investigate the non-identifiability issues associated with bidirectional adversarial training for joint distribution matching. Within a framework of conditional entropy, we propose both adversarial and non-adversarial approaches to learn desirable matched joint distributions for unsupervised and supervised tasks. We unify a broad family of adversarial models as joint distribution matching problems. Our approach stabilizes learning of unsupervised bidirectional adversarial learning methods. Further, we introduce an extension for semi-supervised learning tasks. Theoretical results are validated in synthetic data and real-world applications.
[Abstract](https://arxiv.org/abs/1709.01215), [PDF](https://arxiv.org/pdf/1709.01215)


### #530: Finite sample analysis of the GTD Policy Evaluation Algorithms in Markov Setting

### #531: On the Complexity of Learning Neural Networks
_Le Song,  Santosh Vempala,  John Wilmes,  Bo Xie_

The stunning empirical successes of neural networks currently lack rigorous theoretical explanation. What form would such an explanation take, in the face of existing complexity-theoretic lower bounds? A first step might be to show that data generated by neural networks with a single hidden layer, smooth activation functions and benign input distributions can be learned efficiently. We demonstrate here a comprehensive lower bound ruling out this possibility: for a wide class of activation functions (including all currently used), and inputs drawn from any logconcave distribution, there is a family of one-hidden-layer functions whose output is a sum gate, that are hard to learn in a precise sense: any statistical query algorithm (which includes all known variants of stochastic gradient descent with any loss function) needs an exponential number of queries even using tolerance inversely proportional to the input dimensionality. Moreover, this hard family of functions is realizable with a small (sublinear in dimension) number of activation units in the single hidden layer. The lower bound is also robust to small perturbations of the true weights. Systematic experiments illustrate a phase transition in the training error as predicted by the analysis.
[Abstract](https://arxiv.org/abs/1707.04615), [PDF](https://arxiv.org/pdf/1707.04615)


### #532: Hierarchical Implicit Models and Likelihood-Free Variational Inference

### #533: Improved Semi-supervised Learning with GANs using Manifold Invariances
_Abhishek Kumar,  Prasanna Sattigeri,  P. Thomas Fletcher_

Semi-supervised learning methods using Generative Adversarial Networks (GANs) have shown promising empirical success recently. Most of these methods use a shared discriminator/classifier which discriminates real examples from fake while also predicting the class label. Motivated by the ability of the GANs generator to capture the data manifold well, we propose to estimate the tangent space to the data manifold using GANs and employ it to inject invariances into the classifier. In the process, we propose enhancements over existing methods for learning the inverse mapping (i.e., the encoder) which greatly improves in terms of semantic similarity of the reconstructed sample with the input sample. We observe considerable empirical gains in semi-supervised learning over baselines, particularly in the cases when the number of labeled examples is low. We also provide insights into how fake examples influence the semi-supervised learning procedure.
[Abstract](https://arxiv.org/abs/1705.08850), [PDF](https://arxiv.org/pdf/1705.08850)


### #534: Approximation and Convergence Properties of Generative Adversarial Learning
_Shuang Liu,  Olivier Bousquet,  Kamalika Chaudhuri_

Generative adversarial networks (GAN) approximate a target data distribution by jointly optimizing an objective function through a "two-player game" between a generator and a discriminator. Despite their empirical success, however, two very basic questions on how well they can approximate the target distribution remain unanswered. First, it is not known how restricting the discriminator family affects the approximation quality. Second, while a number of different objective functions have been proposed, we do not understand when convergence to the global minima of the objective function leads to convergence to the target distribution under various notions of distributional convergence. In this paper, we address these questions in a broad and unified setting by defining a notion of adversarial divergences that includes a number of recently proposed objective functions. We show that if the objective function is an adversarial divergence with some additional conditions, then using a restricted discriminator family has a moment-matching effect. Additionally, we show that for objective functions that are strict adversarial divergences, convergence in the objective function implies weak convergence, thus generalizing previous results.
[Abstract](https://arxiv.org/abs/1705.08991), [PDF](https://arxiv.org/pdf/1705.08991)


### #535: From Bayesian Sparsity to Gated Recurrent Nets
_Hao He,  Bo Xin,  David Wipf_

The iterations of many first-order algorithms, when applied to minimizing common regularized regression functions, often resemble neural network layers with pre-specified weights. This observation has prompted the development of learning-based approaches that purport to replace these iterations with enhanced surrogates forged as DNN models from available training data. For example, important NP-hard sparse estimation problems have recently benefitted from this genre of upgrade, with simple feedforward or recurrent networks ousting proximal gradient-based iterations. Analogously, this paper demonstrates that more powerful Bayesian algorithms for promoting sparsity, which rely on complex multi-loop majorization-minimization techniques, mirror the structure of more sophisticated long short-term memory (LSTM) networks, or alternative gated feedback networks previously designed for sequence prediction. As part of this development, we examine the parallels between latent variable trajectories operating across multiple time-scales during optimization, and the activations within deep network structures designed to adaptively model such characteristic sequences. The resulting insights lead to a novel sparse estimation system that, when granted training data, can estimate optimal solutions efficiently in regimes where other algorithms fail, including practical direction-of-arrival (DOA) and 3D geometry recovery problems. The underlying principles we expose are also suggestive of a learning process for a richer class of multi-loop algorithms in other domains.
[Abstract](https://arxiv.org/abs/1706.02815), [PDF](https://arxiv.org/pdf/1706.02815)


### #536: Min-Max Propagation

### #537: What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
_Alex Kendall,  Yarin Gal_

There are two major types of uncertainty one can model. Aleatoric uncertainty captures noise inherent in the observations. On the other hand, epistemic uncertainty accounts for uncertainty in the model -- uncertainty which can be explained away given enough data. Traditionally it has been difficult to model epistemic uncertainty in computer vision, but with new Bayesian deep learning tools this is now possible. We study the benefits of modeling epistemic vs. aleatoric uncertainty in Bayesian deep learning models for vision tasks. For this we present a Bayesian deep learning framework combining input-dependent aleatoric uncertainty together with epistemic uncertainty. We study models under the framework with per-pixel semantic segmentation and depth regression tasks. Further, our explicit uncertainty formulation leads to new loss functions for these tasks, which can be interpreted as learned attenuation. This makes the loss more robust to noisy data, also giving new state-of-the-art results on segmentation and depth regression benchmarks.
[Abstract](https://arxiv.org/abs/1703.04977), [PDF](https://arxiv.org/pdf/1703.04977)


### #538: Gradient descent GAN optimization is locally stable
_Vaishnavh Nagarajan,  J. Zico Kolter_

Despite their growing prominence, optimization in generative adversarial networks (GANs) is still a poorly-understood topic. In this paper, we analyze the "gradient descent" form of GAN optimization (i.e., the natural setting where we simultaneously take small gradient steps in both generator and discriminator parameters). We show that even though GAN optimization does not correspond to a convex-concave game, even for simple parameterizations, under proper conditions, equilibrium points of this optimization procedure are still locally asymptotically stable for the traditional GAN formulation. On the other hand, we show that the recently-proposed Wasserstein GAN can have non-convergent limit cycles near equilibrium. Motivated by this stability analysis, we propose an additional regularization term for gradient descent GAN updates, which is able to guarantee local stability for both the WGAN and for the traditional GAN, and also shows practical promise in speeding up convergence and addressing mode collapse.
[Abstract](https://arxiv.org/abs/1706.04156), [PDF](https://arxiv.org/pdf/1706.04156)


### #539: Toward Robustness against Label Noise in Training Deep Discriminative Neural Networks
_Arash Vahdat_

Collecting large training datasets, annotated with high quality labels, is a costly process. This paper proposes a novel framework for training deep convolutional neural networks from noisy labeled datasets. The problem is formulated using an undirected graphical model that represents the relationship between noisy and clean labels, trained in a semi-supervised setting. In the proposed structure, the inference over latent clean labels is tractable and is regularized during training using auxiliary sources of information. The proposed model is applied to the image labeling problem and is shown to be effective in labeling unseen images as well as reducing label noise in training on CIFAR-10 and MS COCO datasets.
[Abstract](https://arxiv.org/abs/1706.00038), [PDF](https://arxiv.org/pdf/1706.00038)


### #540: Dualing GANs
_Yujia Li,  Alexander Schwing,  Kuan-Chieh Wang,  Richard Zemel_

Generative adversarial nets (GANs) are a promising technique for modeling a distribution from samples. It is however well known that GAN training suffers from instability due to the nature of its maximin formulation. In this paper, we explore ways to tackle the instability problem by dualizing the discriminator. We start from linear discriminators in which case conjugate duality provides a mechanism to reformulate the saddle point objective into a maximization problem, such that both the generator and the discriminator of this 'dualing GAN' act in concert. We then demonstrate how to extend this intuition to non-linear formulations. For GANs with linear discriminators our approach is able to remove the instability in training, while for GANs with nonlinear discriminators our approach provides an alternative to the commonly used GAN training algorithm.
[Abstract](https://arxiv.org/abs/1706.06216), [PDF](https://arxiv.org/pdf/1706.06216)


### #541: Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model
_Xingjian Shi,  Zhihan Gao,  Leonard Lausen,  Hao Wang,  Dit-Yan Yeung,  Wai-kin Wong,  Wang-chun Woo_

With the goal of making high-resolution forecasts of regional rainfall, precipitation nowcasting has become an important and fundamental technology underlying various public services ranging from rainstorm warnings to flight safety. Recently, the convolutional LSTM (ConvLSTM) model has been shown to outperform traditional optical flow based methods for precipitation nowcasting, suggesting that deep learning models have a huge potential for solving the problem. However, the convolutional recurrence structure in ConvLSTM-based models is location-invariant while natural motion and transformation (e.g., rotation) are location-variant in general. Furthermore, since deep-learning-based precipitation nowcasting is a newly emerging area, clear evaluation protocols have not yet been established. To address these problems, we propose both a new model and a benchmark for precipitation nowcasting. Specifically, we go beyond ConvLSTM and propose the Trajectory GRU (TrajGRU) model that can actively learn the location-variant structure for recurrent connections. Besides, we provide a benchmark that includes a real-world large-scale dataset from the Hong Kong Observatory, a new training loss, and a comprehensive evaluation protocol to facilitate future research and gauge the state of the art.
[Abstract](https://arxiv.org/abs/1706.03458), [PDF](https://arxiv.org/pdf/1706.03458)


### #542: Do Deep Neural Networks Suffer from Crowding?
_Anna Volokitin,  Gemma Roig,  Tomaso Poggio_

Crowding is a visual effect suffered by humans, in which an object that can be recognized in isolation can no longer be recognized when other objects, called flankers, are placed close to it. In this work, we study the effect of crowding in artificial Deep Neural Networks for object recognition. We analyze both standard deep convolutional neural networks (DCNNs) as well as a new version of DCNNs which is 1) multi-scale and 2) with size of the convolution filters change depending on the eccentricity wrt to the center of fixation. Such networks, that we call eccentricity-dependent, are a computational model of the feedforward path of the primate visual cortex. Our results reveal that the eccentricity-dependent model, trained on target objects in isolation, can recognize such targets in the presence of flankers, if the targets are near the center of the image, whereas DCNNs cannot. Also, for all tested networks, when trained on targets in isolation, we find that recognition accuracy of the networks decreases the closer the flankers are to the target and the more flankers there are. We find that visual similarity between the target and flankers also plays a role and that pooling in early layers of the network leads to more crowding. Additionally, we show that incorporating the flankers into the images of the training set does not improve performance with crowding.
[Abstract](https://arxiv.org/abs/1706.08616), [PDF](https://arxiv.org/pdf/1706.08616)


### #543: Learning from Complementary Labels
_Takashi Ishida,  Gang Niu,  Masashi Sugiyama_

Collecting labeled data is costly and thus is a critical bottleneck in real-world classification tasks. To mitigate the problem, we consider a complementary label, which specifies a class that a pattern does not belong to. Collecting complementary labels would be less laborious than ordinary labels since users do not have to carefully choose the correct class from many candidate classes. However, complementary labels are less informative than ordinary labels and thus a suitable approach is needed to better learn from complementary labels. In this paper, we show that an unbiased estimator of the classification risk can be obtained only from complementary labels, if a loss function satisfies a particular symmetric condition. We theoretically prove the estimation error bounds for the proposed method, and experimentally demonstrate the usefulness of the proposed algorithms.
[Abstract](https://arxiv.org/abs/1705.07541), [PDF](https://arxiv.org/pdf/1705.07541)


### #544: More powerful and flexible rules for online FDR control with memory and weights

### #545: Learning from uncertain curves: The 2-Wasserstein metric for Gaussian processes

### #546: Discriminative State Space Models

### #547: On Fairness and Calibration
_Geoff Pleiss,  Manish Raghavan,  Felix Wu,  Jon Kleinberg,  Kilian Q. Weinberger_

The machine learning community has become increasingly concerned with the potential for bias and discrimination in predictive models, and this has motivated a growing line of work on what it means for a classification procedure to be "fair." In particular, we investigate the tension between minimizing error disparity across different population groups while maintaining calibrated probability estimates. We show that calibration is compatible only with a single error constraint (i.e. equal false-negatives rates across groups), and show that any algorithm that satisfies this relaxation is no better than randomizing a percentage of predictions for an existing classifier. These unsettling findings, which extend and generalize existing results, are empirically confirmed on several datasets.
[Abstract](https://arxiv.org/abs/1709.02012), [PDF](https://arxiv.org/pdf/1709.02012)


### #548: Imagination-Augmented Agents for Deep Reinforcement Learning
_Théophane Weber,  Sébastien Racanière,  David P. Reichert,  Lars Buesing,  Arthur Guez,  Danilo Jimenez Rezende,  Adria Puigdomènech Badia,  Oriol Vinyals,  Nicolas Heess,  Yujia Li,  Razvan Pascanu,  Peter Battaglia,  David Silver,  Daan Wierstra_

We introduce Imagination-Augmented Agents (I2As), a novel architecture for deep reinforcement learning combining model-free and model-based aspects. In contrast to most existing model-based reinforcement learning and planning methods, which prescribe how a model should be used to arrive at a policy, I2As learn to interpret predictions from a learned environment model to construct implicit plans in arbitrary ways, by using the predictions as additional context in deep policy networks. I2As show improved data efficiency, performance, and robustness to model misspecification compared to several baselines.
[Abstract](https://arxiv.org/abs/1707.06203), [PDF](https://arxiv.org/pdf/1707.06203)


### #549: Extracting low-dimensional dynamics from multiple large-scale neural population recordings by learning to predict correlations

### #550: Unifying PAC and Regret: Uniform PAC Bounds for Episodic Reinforcement Learning

### #551: Gradients of Generative Models for Improved Discriminative Analysis of Tandem Mass Spectra

### #552: Asynchronous Parallel Coordinate Minimization for MAP Inference

### #553: Multiscale Quantization for Fast Similarity Search

### #554: Diverse and Accurate Image Description Using a Variational Auto-Encoder with an Additive Gaussian Encoding Space

### #555: Improved Training of Wasserstein GANs
_Ishaan Gulrajani,  Faruk Ahmed,  Martin Arjovsky,  Vincent Dumoulin,  Aaron Courville_

Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but can still generate low-quality samples or fail to converge in some settings. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to pathological behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.
[Abstract](https://arxiv.org/abs/1704.00028), [PDF](https://arxiv.org/pdf/1704.00028)


### #556: Optimally Learning Populations of Parameters
_Kevin Tian,  Weihao Kong,  Gregory Valiant_

Consider the following fundamental estimation problem: there are $n$ entities, each with an unknown parameter $p_i \in [0,1]$, and we observe $n$ independent random variables, $X_1,\ldots,X_n$, with $X_i \sim $ Binomial$(t, p_i)$. How accurately can one recover the "histogram" (i.e. cumulative density function) of the $p_i$s? While the empirical estimates would recover the histogram to earth mover distance $\Theta(\frac{1}{\sqrt{t}})$ (equivalently, $\ell_1$ distance between the CDFs), we show that, provided $n$ is sufficiently large, we can achieve error $O(\frac{1}{t})$ which is information theoretically optimal. We also extend our results to the multi-dimensional parameter case, capturing settings where each member of the population has multiple associated parameters. Beyond the theoretical results, we demonstrate that the recovery algorithm performs well in practice on a variety of datasets, providing illuminating insights into several domains, including politics, and sports analytics.
[Abstract](https://arxiv.org/abs/1709.02707), [PDF](https://arxiv.org/pdf/1709.02707)


### #557: Clustering with Noisy Queries
_Arya Mazumdar,  Barna Saha_

In this paper, we initiate a rigorous theoretical study of clustering with noisy queries (or a faulty oracle). Given a set of $n$ elements, our goal is to recover the true clustering by asking minimum number of pairwise queries to an oracle. Oracle can answer queries of the form : "do elements $u$ and $v$ belong to the same cluster?" -- the queries can be asked interactively (adaptive queries), or non-adaptively up-front, but its answer can be erroneous with probability $p$. In this paper, we provide the first information theoretic lower bound on the number of queries for clustering with noisy oracle in both situations. We design novel algorithms that closely match this query complexity lower bound, even when the number of clusters is unknown. Moreover, we design computationally efficient algorithms both for the adaptive and non-adaptive settings. The problem captures/generalizes multiple application scenarios. It is directly motivated by the growing body of work that use crowdsourcing for {\em entity resolution}, a fundamental and challenging data mining task aimed to identify all records in a database referring to the same entity. Here crowd represents the noisy oracle, and the number of queries directly relates to the cost of crowdsourcing. Another application comes from the problem of {\em sign edge prediction} in social network, where social interactions can be both positive and negative, and one must identify the sign of all pair-wise interactions by querying a few pairs. Furthermore, clustering with noisy oracle is intimately connected to correlation clustering, leading to improvement therein. Finally, it introduces a new direction of study in the popular {\em stochastic block model} where one has an incomplete stochastic block model matrix to recover the clusters.
[Abstract](https://arxiv.org/abs/1706.07510), [PDF](https://arxiv.org/pdf/1706.07510)


### #558: Higher-Order Total Variation Classes on Grids: Minimax Theory and Trend Filtering Methods

### #559: Training Quantized Nets: A Deeper Understanding
_Hao Li,  Soham De,  Zheng Xu,  Christoph Studer,  Hanan Samet,  Tom Goldstein_

Currently, deep neural networks are deployed on low-power embedded devices by first training a full-precision model using powerful computing hardware, and then deriving a corresponding low-precision model for efficient inference on such systems. However, training models directly with coarsely quantized weights is a key step towards learning on embedded platforms that have limited computing resources, memory capacity, and power consumption. Numerous recent publications have studied methods for training quantized network, but these studies have mostly been empirical. In this work, we investigate training methods for quantized neural networks from a theoretical viewpoint. We first explore accuracy guarantees for training methods under convexity assumptions. We then look at the behavior of algorithms for non-convex problems, and we show that training algorithms that exploit high-precision representations have an important annealing property that purely quantized training methods lack, which explains many of the observed empirical differences between these types of algorithms.
[Abstract](https://arxiv.org/abs/1706.02379), [PDF](https://arxiv.org/pdf/1706.02379)


### #560: Permutation-based Causal Inference Algorithms with Interventions
_Yuhao Wang,  Liam Solus,  Karren Dai Yang,  Caroline Uhler_

Learning Bayesian networks using both observational and interventional data is now a fundamentally important problem due to recent technological developments in genomics that generate such single-cell gene expression data at a very large scale. In order to utilize this data for learning gene regulatory networks, efficient and reliable causal inference algorithms are needed that can make use of both observational and interventional data. In this paper, we present two algorithms of this type and prove that both are consistent under the faithfulness assumption. These algorithms are interventional adaptations of the Greedy SP algorithm and are the first algorithms using both observational and interventional data with consistency guarantees. Moreover, these algorithms have the advantage that they are non-parametric, which makes them useful also for analyzing non-Gaussian data. In this paper, we present these two algorithms and their consistency guarantees, and we analyze their performance on simulated data, protein signaling data, and single-cell gene expression data.
[Abstract](https://arxiv.org/abs/1705.10220), [PDF](https://arxiv.org/pdf/1705.10220)


### #561: Time-dependent spatially varying graphical models, with application to brain fMRI data analysis

### #562: Gradient Methods for Submodular Maximization
_Hamed Hassani,  Mahdi Soltanolkotabi,  Amin Karbasi_

In this paper, we study the problem of maximizing continuous submodular functions that naturally arise in many learning applications such as those involving utility functions in active learning and sensing, matrix approximations and network inference. Despite the apparent lack of convexity in such functions, we prove that stochastic projected gradient methods can provide strong approximation guarantees for maximizing continuous submodular functions with convex constraints. More specifically, we prove that for monotone continuous DR-submodular functions, all fixed points of projected gradient ascent provide a factor $1/2$ approximation to the global maxima. We also study stochastic gradient and mirror methods and show that after $\mathcal{O}(1/\epsilon^2)$ iterations these methods reach solutions which achieve in expectation objective values exceeding $(\frac{\text{OPT}}{2}-\epsilon)$. An immediate application of our results is to maximize submodular functions that are defined stochastically, i.e. the submodular function is defined as an expectation over a family of submodular functions with an unknown distribution. We will show how stochastic gradient methods are naturally well-suited for this setting, leading to a factor $1/2$ approximation when the function is monotone. In particular, it allows us to approximately maximize discrete, monotone submodular optimization problems via projected gradient descent on a continuous relaxation, directly connecting the discrete and continuous domains. Finally, experiments on real data demonstrate that our projected gradient methods consistently achieve the best utility compared to other continuous baselines while remaining competitive in terms of computational effort.
[Abstract](https://arxiv.org/abs/1708.03949), [PDF](https://arxiv.org/pdf/1708.03949)


### #563: Smooth Primal-Dual Coordinate Descent Algorithms for Nonsmooth Convex Optimization

### #564: Maximizing the Spread of Influence from Training Data

### #565: Multiplicative Weights Update with Constant Step-Size in Congestion Games:  Convergence, Limit Cycles and Chaos
_Gerasimos Palaiopanos,  Ioannis Panageas,  Georgios Piliouras_

The Multiplicative Weights Update (MWU) method is a ubiquitous meta-algorithm that works as follows: A distribution is maintained on a certain set, and at each step the probability assigned to element $\gamma$ is multiplied by $(1 -\epsilon C(\gamma))>0$ where $C(\gamma)$ is the "cost" of element $\gamma$ and then rescaled to ensure that the new values form a distribution. We analyze MWU in congestion games where agents use \textit{arbitrary admissible constants} as learning rates $\epsilon$ and prove convergence to \textit{exact Nash equilibria}. Our proof leverages a novel connection between MWU and the Baum-Welch algorithm, the standard instantiation of the Expectation-Maximization (EM) algorithm for hidden Markov models (HMM). Interestingly, this convergence result does not carry over to the nearly homologous MWU variant where at each step the probability assigned to element $\gamma$ is multiplied by $(1 -\epsilon)^{C(\gamma)}$ even for the most innocuous case of two-agent, two-strategy load balancing games, where such dynamics can provably lead to limit cycles or even chaotic behavior.
[Abstract](https://arxiv.org/abs/1703.01138), [PDF](https://arxiv.org/pdf/1703.01138)


### #566: Learning Neural Representations of Human Cognition across Many fMRI Studies

### #567: A KL-LUCB algorithm for Large-Scale Crowdsourcing

### #568: Collaborative Deep Learning in Fixed Topology Networks
_Zhanhong Jiang,  Aditya Balu,  Chinmay Hegde,  Soumik Sarkar_

There is significant recent interest to parallelize deep learning algorithms in order to handle the enormous growth in data and model sizes. While most advances focus on model parallelization and engaging multiple computing agents via using a central parameter server, aspect of data parallelization along with decentralized computation has not been explored sufficiently. In this context, this paper presents a new consensus-based distributed SGD (CDSGD) (and its momentum variant, CDMSGD) algorithm for collaborative deep learning over fixed topology networks that enables data parallelization as well as decentralized computation. Such a framework can be extremely useful for learning agents with access to only local/private data in a communication constrained environment. We analyze the convergence properties of the proposed algorithm with strongly convex and nonconvex objective functions with fixed and diminishing step sizes using concepts of Lyapunov function construction. We demonstrate the efficacy of our algorithms in comparison with the baseline centralized SGD and the recently proposed federated averaging algorithm (that also enables data parallelism) based on benchmark datasets such as MNIST, CIFAR-10 and CIFAR-100.
[Abstract](https://arxiv.org/abs/1706.07880), [PDF](https://arxiv.org/pdf/1706.07880)


### #569: Fast-Slow Recurrent Neural Networks
_Asier Mujika,  Florian Meier,  Angelika Steger_

Processing sequential data of variable length is a major challenge in a wide range of applications, such as speech recognition, language modeling, generative image modeling and machine translation. Here, we address this challenge by proposing a novel recurrent neural network (RNN) architecture, the Fast-Slow RNN (FS-RNN). The FS-RNN incorporates the strengths of both multiscale RNNs and deep transition RNNs as it processes sequential data on different timescales and learns complex transition functions from one time step to the next. We evaluate the FS-RNN on two character level language modeling data sets, Penn Treebank and Hutter Prize Wikipedia, where we improve state of the art results to $1.19$ and $1.25$ bits-per-character (BPC), respectively. In addition, an ensemble of two FS-RNNs achieves $1.20$ BPC on Hutter Prize Wikipedia outperforming the best known compression algorithm with respect to the BPC measure. We also present an empirical investigation of the learning and network dynamics of the FS-RNN, which explains the improved performance compared to other RNN architectures. Our approach is general as any kind of RNN cell is a possible building block for the FS-RNN architecture, and thus can be flexibly applied to different tasks.
[Abstract](https://arxiv.org/abs/1705.08639), [PDF](https://arxiv.org/pdf/1705.08639)


### #570: Learning Disentangled Representations with Semi-Supervised Deep Generative Models
_N. Siddharth,  Brooks Paige,  Jan-Willem Van de Meent,  Alban Desmaison,  Frank Wood,  Noah D. Goodman,  Pushmeet Kohli,  Philip H.S. Torr_

Variational autoencoders (VAEs) learn representations of data by jointly training a probabilistic encoder and decoder network. Typically these models encode all features of the data into a single variable. Here we are interested in learning disentangled representations that encode distinct aspects of the data into separate variables. We propose to learn such representations using model architectures that generalize from standard VAEs, employing a general graphical model structure in the encoder and decoder. This allows us to train partially-specified models that make relatively strong assumptions about a subset of interpretable variables and rely on the flexibility of neural networks to learn representations for the remaining variables. We further define a general objective for semi-supervised learning in this model class, which can be approximated using an importance sampling procedure. We evaluate our framework's ability to learn disentangled representations, both by qualitative exploration of its generative capacity, and quantitative evaluation of its discriminative ability on a variety of models and datasets.
[Abstract](https://arxiv.org/abs/1706.00400), [PDF](https://arxiv.org/pdf/1706.00400)


### #571: Learning to Generalize Intrinsic Images with a Structured Disentangling Autoencoder

### #572: Exploring Generalization in Deep Learning
_Behnam Neyshabur,  Srinadh Bhojanapalli,  David McAllester,  Nathan Srebro_

With a goal of understanding what drives generalization in deep networks, we consider several recently suggested explanations, including norm-based control, sharpness and robustness. We study how these measures can ensure generalization, highlighting the importance of scale normalization, and making a connection between sharpness and PAC-Bayes theory. We then investigate how well the measures explain different observed phenomena.
[Abstract](https://arxiv.org/abs/1706.08947), [PDF](https://arxiv.org/pdf/1706.08947)


### #573: A framework for Multi-A(rmed)/B(andit) Testing with Online FDR Control

### #574: Fader Networks: Generating Image Variations by Sliding Attribute Values

### #575: Action Centered Contextual Bandits

### #576: Estimating Mutual Information for Discrete-Continuous Mixtures
_Weihao Gao,  Sreeram Kannan,  Sewoong Oh,  Pramod Viswanath_

Estimating mutual information from observed samples is a basic primitive, useful in several machine learning tasks including correlation mining, information bottleneck clustering, learning a Chow-Liu tree, and conditional independence testing in (causal) graphical models. While mutual information is a well-defined quantity in general probability spaces, existing estimators can only handle two special cases of purely discrete or purely continuous pairs of random variables. The main challenge is that these methods first estimate the (differential) entropies of X, Y and the pair (X;Y) and add them up with appropriate signs to get an estimate of the mutual information. These 3H-estimators cannot be applied in general mixture spaces, where entropy is not well-defined. In this paper, we design a novel estimator for mutual information of discrete-continuous mixtures. We prove that the proposed estimator is consistent. We provide numerical experiments suggesting superiority of the proposed estimator compared to other heuristics of adding small continuous noise to all the samples and applying standard estimators tailored for purely continuous variables, and quantizing the samples and applying standard estimators tailored for purely discrete variables. This significantly widens the applicability of mutual information estimation in real-world applications, where some variables are discrete, some continuous, and others are a mixture between continuous and discrete components.
[Abstract](https://arxiv.org/abs/1709.06212), [PDF](https://arxiv.org/pdf/1709.06212)


### #577: Attention is All you Need

### #578: Recurrent Ladder Networks
_Alexander Ilin,  Isabeau Prémont-Schwarz,  Tele Hotloo Hao,  Antti Rasmus,  Rinu Boney,  Harri Valpola_

We propose a recurrent extension of the Ladder networks whose structure is motivated by the inference required in hierarchical latent variable models. We demonstrate that the recurrent Ladder is able to handle a wide variety of complex learning tasks that benefit from iterative inference and temporal modeling. The architecture shows close-to-optimal results on temporal modeling of video data, competitive results on music modeling, and improved perceptual grouping based on higher order abstractions, such as stochastic textures and motion cues. We present results for fully supervised, semi-supervised, and unsupervised tasks. The results suggest that the proposed architecture and principles are powerful tools for learning a hierarchy of abstractions, learning iterative inference and handling temporal information.
[Abstract](https://arxiv.org/abs/1707.09219), [PDF](https://arxiv.org/pdf/1707.09219)


### #579: Parameter-Free Online Learning via Model Selection

### #580: Bregman Divergence for Stochastic Variance Reduction: Saddle-Point and Adversarial Prediction

### #581: Unbounded cache model for online language modeling with open vocabulary

### #582: Predictive State Recurrent Neural Networks
_Carlton Downey,  Ahmed Hefny,  Boyue Li,  Byron Boots,  Geoffrey Gordon_

We present a new model, Predictive State Recurrent Neural Networks (PSRNNs), for filtering and prediction in dynamical systems. PSRNNs draw on insights from both Recurrent Neural Networks (RNNs) and Predictive State Representations (PSRs), and inherit advantages from both types of models. Like many successful RNN architectures, PSRNNs use (potentially deeply composed) bilinear transfer functions to combine information from multiple sources. We show that such bilinear functions arise naturally from state updates in Bayes filters like PSRs, in which observations can be viewed as gating belief states. We also show that PSRNNs can be learned effectively by combining Backpropogation Through Time (BPTT) with an initialization derived from a statistically consistent learning algorithm for PSRs called two-stage regression (2SR). Finally, we show that PSRNNs can be factorized using tensor decomposition, reducing model size and suggesting interesting connections to existing multiplicative architectures such as LSTMs. We applied PSRNNs to 4 datasets, and showed that we outperform several popular alternative approaches to modeling dynamical systems in all cases.
[Abstract](https://arxiv.org/abs/1705.09353), [PDF](https://arxiv.org/pdf/1705.09353)


### #583: Early stopping for kernel boosting algorithms: A general analysis with localized complexities
_Yuting Wei,  Fanny Yang,  Martin J. Wainwright_

Early stopping of iterative algorithms is a widely-used form of regularization in statistics, commonly used in conjunction with boosting and related gradient-type algorithms. Although consistency results have been established in some settings, such estimators are less well-understood than their analogues based on penalized regularization. In this paper, for a relatively broad class of loss functions and boosting algorithms (including L2-boost, LogitBoost and AdaBoost, among others), we exhibit a direct connection between the performance of a stopped iterate and the localized Gaussian complexity of the associated function class. This connection allows us to show that local fixed point analysis of Gaussian or Rademacher complexities, now standard in the analysis of penalized estimators, can be used to derive optimal stopping rules. We derive such stopping rules in detail for various kernel classes, and illustrate the correspondence of our theory with practice for Sobolev kernel classes.
[Abstract](https://arxiv.org/abs/1707.01543), [PDF](https://arxiv.org/pdf/1707.01543)


### #584: SVCCA: Singular Vector Canonical Correlation Analysis for Deep Understanding and Improvement
_Maithra Raghu,  Justin Gilmer,  Jason Yosinski,  Jascha Sohl-Dickstein_

With the continuing empirical successes of deep networks, it becomes increasingly important to develop better methods for understanding training of models and the representations learned within. In this paper we propose Singular Vector Canonical Correlation Analysis (SVCCA), a tool for quickly comparing two representations in a way that is both invariant to affine transform (allowing comparison between different layers and networks) and fast to compute (allowing more comparisons to be calculated than with previous methods). We deploy this tool to measure the intrinsic dimensionality of layers, showing in some cases needless over-parameterization; to probe learning dynamics throughout training, finding that networks converge to final representations from the bottom up; to show where class-specific information in networks is formed; and to suggest new training regimes that simultaneously save computation and overfit less.
[Abstract](https://arxiv.org/abs/1706.05806), [PDF](https://arxiv.org/pdf/1706.05806)


### #585: Convolutional Phase Retrieval

### #586: Estimating High-dimensional Non-Gaussian Multiple Index Models via Stein’s Lemma

### #587: Gaussian Quadrature for Kernel Features
_Tri Dao,  Christopher De Sa,  Christopher Ré_

Kernel methods have recently attracted resurgent interest, matching the performance of deep neural networks in tasks such as speech recognition. The random Fourier features map is a technique commonly used to scale up kernel machines, but employing the randomized feature map means that $O(\epsilon^{-2})$ samples are required to achieve an approximation error of at most $\epsilon$. In this paper, we investigate some alternative schemes for constructing feature maps that are deterministic, rather than random, by approximating the kernel in the frequency domain using Gaussian quadrature. We show that deterministic feature maps can be constructed, for any $\gamma > 0$, to achieve error $\epsilon$ with $O(e^{\gamma} + \epsilon^{-1/\gamma})$ samples as $\epsilon$ goes to 0. We validate our methods on datasets in different domains, such as MNIST and TIMIT, showing that deterministic features are faster to generate and achieve comparable accuracy to the state-of-the-art kernel methods based on random Fourier features.
[Abstract](https://arxiv.org/abs/1709.02605), [PDF](https://arxiv.org/pdf/1709.02605)


### #588: Value Prediction Network
_Junhyuk Oh,  Satinder Singh,  Honglak Lee_

This paper proposes a novel deep reinforcement learning (RL) architecture, called Value Prediction Network (VPN), which integrates model-free and model-based RL methods into a single neural network. In contrast to typical model-based RL methods, VPN learns a dynamics model whose abstract states are trained to make option-conditional predictions of future values (discounted sum of rewards) rather than of future observations. Our experimental results show that VPN has several advantages over both model-free and model-based baselines in a stochastic environment where careful planning is required but building an accurate observation-prediction model is difficult. Furthermore, VPN outperforms Deep Q-Network (DQN) on several Atari games even with short-lookahead planning, demonstrating its potential as a new way of learning a good state representation.
[Abstract](https://arxiv.org/abs/1707.03497), [PDF](https://arxiv.org/pdf/1707.03497)


### #589: On Learning Errors of Structured Prediction with Approximate Inference

### #590: Efficient Second-Order Online Kernel Learning with Adaptive Embedding

### #591: Implicit Regularization in Matrix Factorization
_Suriya Gunasekar,  Blake Woodworth,  Srinadh Bhojanapalli,  Behnam Neyshabur,  Nathan Srebro_

We study implicit regularization when optimizing an underdetermined quadratic objective over a matrix $X$ with gradient descent on a factorization of $X$. We conjecture and provide empirical and theoretical evidence that with small enough step sizes and initialization close enough to the origin, gradient descent on a full dimensional factorization converges to the minimum nuclear norm solution.
[Abstract](https://arxiv.org/abs/1705.09280), [PDF](https://arxiv.org/pdf/1705.09280)


### #592: Optimal Shrinkage of Singular Values Under Random Data Contamination

### #593: Delayed Mirror Descent in Continuous Games

### #594: Asynchronous Coordinate Descent under More Realistic Assumptions
_Tao Sun,  Robert Hannah,  Wotao Yin_

Asynchronous-parallel algorithms have the potential to vastly speed up algorithms by eliminating costly synchronization. However, our understanding to these algorithms is limited because the current convergence of asynchronous (block) coordinate descent algorithms are based on somewhat unrealistic assumptions. In particular, the age of the shared optimization variables being used to update a block is assumed to be independent of the block being updated. Also, it is assumed that the updates are applied to randomly chosen blocks. In this paper, we argue that these assumptions either fail to hold or will imply less efficient implementations. We then prove the convergence of asynchronous-parallel block coordinate descent under more realistic assumptions, in particular, always without the independence assumption. The analysis permits both the deterministic (essentially) cyclic and random rules for block choices. Because a bound on the asynchronous delays may or may not be available, we establish convergence for both bounded delays and unbounded delays. The analysis also covers nonconvex, weakly convex, and strongly convex functions. We construct Lyapunov functions that directly model both objective progress and delays, so delays are not treated errors or noise. A continuous-time ODE is provided to explain the construction at a high level.
[Abstract](https://arxiv.org/abs/1705.08494), [PDF](https://arxiv.org/pdf/1705.08494)


### #595: Linear Convergence of a Frank-Wolfe Type Algorithm over Trace-Norm Balls
_Zeyuan Allen-Zhu,  Elad Hazan,  Wei Hu,  Yuanzhi Li_

We propose a rank-$k$ variant of the classical Frank-Wolfe algorithm to solve convex optimization over a trace-norm ball. Our algorithm replaces the top singular-vector computation ($1$-SVD) in Frank-Wolfe with a top-$k$ singular-vector computation ($k$-SVD), which can be done by repeatedly applying $1$-SVD $k$ times. Our algorithm has a linear convergence rate when the objective function is smooth and strongly convex, and the optimal solution has rank at most $k$. This improves the convergence rate and the total complexity of the Frank-Wolfe method and its variants.
[Abstract](https://arxiv.org/abs/1708.02105), [PDF](https://arxiv.org/pdf/1708.02105)


### #596: Hierarchical Clustering Beyond the Worst-Case

### #597: Invariance and Stability of Deep Convolutional Representations

### #598: Statistical Cost Sharing
_Eric Balkanski,  Umar Syed,  Sergei Vassilvitskii_

We study the cost sharing problem for cooperative games in situations where the cost function $C$ is not available via oracle queries, but must instead be derived from data, represented as tuples $(S, C(S))$, for different subsets $S$ of players. We formalize this approach, which we call statistical cost sharing, and consider the computation of the core and the Shapley value, when the tuples are drawn from some distribution $\mathcal{D}$. Previous work by Balcan et al. in this setting showed how to compute cost shares that satisfy the core property with high probability for limited classes of functions. We expand on their work and give an algorithm that computes such cost shares for any function with a non-empty core. We complement these results by proving an inapproximability lower bound for a weaker relaxation. We then turn our attention to the Shapley value. We first show that when cost functions come from the family of submodular functions with bounded curvature, $\kappa$, the Shapley value can be approximated from samples up to a $\sqrt{1 - \kappa}$ factor, and that the bound is tight. We then define statistical analogues of the Shapley axioms, and derive a notion of statistical Shapley value. We show that these can always be approximated arbitrarily well for general functions over any distribution $\mathcal{D}$.
[Abstract](https://arxiv.org/abs/1703.03111), [PDF](https://arxiv.org/pdf/1703.03111)


### #599: The Expressive Power of Neural Networks: A View from the Width
_Zhou Lu,  Hongming Pu,  Feicheng Wang,  Zhiqiang Hu,  Liwei Wang_

The expressive power of neural networks is important for understanding deep learning. Most existing works consider this problem from the view of the depth of a network. In this paper, we study how width affects the expressiveness of neural networks. Classical results state that \emph{depth-bounded} (e.g. depth-$2$) networks with suitable activation functions are universal approximators. We show a universal approximation theorem for \emph{width-bounded} ReLU networks: width-$(n+4)$ ReLU networks, where $n$ is the input dimension, are universal approximators. Moreover, except for a measure zero set, all functions cannot be approximated by width-$n$ ReLU networks, which exhibits a phase transition. Several recent works demonstrate the benefits of depth by proving the depth-efficiency of neural networks. That is, there are classes of deep networks which cannot be realized by any shallow network whose size is no more than an \emph{exponential} bound. Here we pose the dual question on the width-efficiency of ReLU networks: Are there wide networks that cannot be realized by narrow networks whose size is not substantially larger? We show that there exist classes of wide networks which cannot be realized by any narrow network whose depth is no more than a \emph{polynomial} bound. On the other hand, we demonstrate by extensive experiments that narrow networks whose size exceed the polynomial bound by a constant factor can approximate wide and shallow network with high accuracy. Our results provide more comprehensive evidence that depth is more effective than width for the expressiveness of ReLU networks.
[Abstract](https://arxiv.org/abs/1709.02540), [PDF](https://arxiv.org/pdf/1709.02540)


### #600: Spectrally-normalized margin bounds for neural networks
_Peter Bartlett,  Dylan J. Foster,  Matus Telgarsky_

This paper presents a margin-based multiclass generalization bound for neural networks which scales with their margin-normalized "spectral complexity": their Lipschitz constant, meaning the product of the spectral norms of the weight matrices, times a certain correction factor. This bound is empirically investigated for a standard AlexNet network on the mnist and cifar10 datasets, with both original and random labels, where it tightly correlates with the observed excess risks.
[Abstract](https://arxiv.org/abs/1706.08498), [PDF](https://arxiv.org/pdf/1706.08498)


### #601: Robust and Efficient Transfer Learning with Hidden Parameter Markov Decision Processes
_Taylor Killian,  Samuel Daulton,  George Konidaris,  Finale Doshi-Velez_

We introduce a new formulation of the Hidden Parameter Markov Decision Process (HiP-MDP), a framework for modeling families of related tasks using low-dimensional latent embeddings. We replace the original Gaussian Process-based model with a Bayesian Neural Network. Our new framework correctly models the joint uncertainty in the latent parameters and the state space and has more scalable inference, thus expanding the scope the HiP-MDP to applications with higher dimensions and more complex dynamics.
[Abstract](https://arxiv.org/abs/1706.06544), [PDF](https://arxiv.org/pdf/1706.06544)


### #602: Population Matching Discrepancy and Applications in Deep Learning

### #603: Scalable Planning with Tensorflow for Hybrid Nonlinear Domains
_Ga Wu,  Buser Say,  Scott Sanner_

Given recent deep learning results that demonstrate the ability to effectively optimize high-dimensional non-convex functions with gradient descent optimization on GPUs, we ask in this paper whether symbolic gradient optimization tools such as Tensorflow can be effective for planning in hybrid (mixed discrete and continuous) nonlinear domains with high dimensional state and action spaces? To this end, we demonstrate that hybrid planning with Tensorflow and RMSProp gradient descent is competitive with mixed integer linear program (MILP) based optimization on piecewise linear planning domains (where we can compute optimal solutions) and substantially outperforms state-of-the-art interior point methods for nonlinear planning domains. Furthermore, we remark that Tensorflow is highly scalable, converging to a strong policy on a large-scale concurrent domain with a total of 576,000 continuous actions over a horizon of 96 time steps in only 4 minutes. We provide a number of insights that clarify such strong performance including observations that despite long horizons, RMSProp avoids both the vanishing and exploding gradients problem. Together these results suggest a new frontier for highly scalable planning in nonlinear hybrid domains by leveraging GPUs and the power of recent advances in gradient descent with highly optmized toolkits like Tensorflow.
[Abstract](https://arxiv.org/abs/1704.07511), [PDF](https://arxiv.org/pdf/1704.07511)


### #604: Boltzmann Exploration Done Right
_Nicolò Cesa-Bianchi,  Claudio Gentile,  Gábor Lugosi,  Gergely Neu_

Boltzmann exploration is a classic strategy for sequential decision-making under uncertainty, and is one of the most standard tools in Reinforcement Learning (RL). Despite its widespread use, there is virtually no theoretical understanding about the limitations or the actual benefits of this exploration scheme. Does it drive exploration in a meaningful way? Is it prone to misidentifying the optimal actions or spending too much time exploring the suboptimal ones? What is the right tuning for the learning rate? In this paper, we address several of these questions in the classic setup of stochastic multi-armed bandits. One of our main results is showing that the Boltzmann exploration strategy with any monotone learning-rate sequence will induce suboptimal behavior. As a remedy, we offer a simple non-monotone schedule that guarantees near-optimal performance, albeit only when given prior access to key problem parameters that are typically not available in practical situations (like the time horizon $T$ and the suboptimality gap $\Delta$). More importantly, we propose a novel variant that uses different learning rates for different arms, and achieves a distribution-dependent regret bound of order $\frac{K\log^2 T}{\Delta}$ and a distribution-independent bound of order $\sqrt{KT}\log K$ without requiring such prior knowledge. To demonstrate the flexibility of our technique, we also propose a variant that guarantees the same performance bounds even if the rewards are heavy-tailed.
[Abstract](https://arxiv.org/abs/1705.10257), [PDF](https://arxiv.org/pdf/1705.10257)


### #605: Towards the ImageNet-CNN of NLP: Pretraining Sentence Encoders with Machine Translation

### #606: Neural Discrete Representation Learning

### #607: Generalizing GANs: A Turing Perspective

### #608: Scalable Log Determinants for Gaussian Process Kernel Learning

### #609: Poincaré Embeddings for Learning Hierarchical Representations

### #610: Learning Combinatorial Optimization Algorithms over Graphs
_Hanjun Dai,  Elias B. Khalil,  Yuyu Zhang,  Bistra Dilkina,  Le Song_

The design of good heuristics or approximation algorithms for NP-hard combinatorial optimization problems often requires significant specialized knowledge and trial-and-error. Can we automate this challenging, tedious process, and learn the algorithms instead? In many real-world applications, it is typically the case that the same optimization problem is solved again and again on a regular basis, maintaining the same problem structure but differing in the data. This provides an opportunity for learning heuristic algorithms that exploit the structure of such recurring problems. In this paper, we propose a unique combination of reinforcement learning and graph embedding to address this challenge. The learned greedy policy behaves like a meta-algorithm that incrementally constructs a solution, and the action is determined by the output of a graph embedding network capturing the current state of the solution. We show our framework can be applied to a diverse range of optimization problems over graphs, and learns effective algorithms for the Minimum Vertex Cover, Maximum Cut and Traveling Salesman problems.
[Abstract](https://arxiv.org/abs/1704.01665), [PDF](https://arxiv.org/pdf/1704.01665)


### #611: Robust Conditional Probabilities
_Yoav Wald,  Amir Globerson_

Conditional probabilities are a core concept in machine learning. For example, optimal prediction of a label $Y$ given an input $X$ corresponds to maximizing the conditional probability of $Y$ given $X$. A common approach to inference tasks is learning a model of conditional probabilities. However, these models are often based on strong assumptions (e.g., log-linear models), and hence their estimate of conditional probabilities is not robust and is highly dependent on the validity of their assumptions. Here we propose a framework for reasoning about conditional probabilities without assuming anything about the underlying distributions, except knowledge of their second order marginals, which can be estimated from data. We show how this setting leads to guaranteed bounds on conditional probabilities, which can be calculated efficiently in a variety of settings, including structured-prediction. Finally, we apply them to semi-supervised deep learning, obtaining results competitive with variational autoencoders.
[Abstract](https://arxiv.org/abs/1708.02406), [PDF](https://arxiv.org/pdf/1708.02406)


### #612: Learning with Bandit Feedback in Potential Games

### #613: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
_Ryan Lowe,  Yi Wu,  Aviv Tamar,  Jean Harb,  Pieter Abbeel,  Igor Mordatch_

We explore deep reinforcement learning methods for multi-agent domains. We begin by analyzing the difficulty of traditional algorithms in the multi-agent case: Q-learning is challenged by an inherent non-stationarity of the environment, while policy gradient suffers from a variance that increases as the number of agents grows. We then present an adaptation of actor-critic methods that considers action policies of other agents and is able to successfully learn policies that require complex multi-agent coordination. Additionally, we introduce a training regimen utilizing an ensemble of policies for each agent that leads to more robust multi-agent policies. We show the strength of our approach compared to existing methods in cooperative as well as competitive scenarios, where agent populations are able to discover various physical and informational coordination strategies.
[Abstract](https://arxiv.org/abs/1706.02275), [PDF](https://arxiv.org/pdf/1706.02275)


### #614: Communication-Efficient Distributed Learning of Discrete Distributions

### #615: Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
_Balaji Lakshminarayanan,  Alexander Pritzel,  Charles Blundell_

Deep neural networks are powerful black box predictors that have recently achieved impressive performance on a wide spectrum of tasks. Quantifying predictive uncertainty in neural networks is a challenging and yet unsolved problem. Bayesian neural networks, which learn a distribution over weights, are currently the state-of-the-art for estimating predictive uncertainty; however these require significant modifications to the training procedure and are computationally expensive compared to standard (non-Bayesian) neural neural networks. We propose an alternative to Bayesian neural networks, that is simple to implement, readily parallelisable and yields high quality predictive uncertainty estimates. Through a series of experiments on classification and regression benchmarks, we demonstrate that our method produces well-calibrated uncertainty estimates which are as good or better than approximate Bayesian neural networks. To assess robustness to dataset shift, we evaluate the predictive uncertainty on test examples from known and unknown distributions, and show that our method is able to express higher uncertainty on unseen data. We demonstrate the scalability of our method by evaluating predictive uncertainty estimates on ImageNet.
[Abstract](https://arxiv.org/abs/1612.01474), [PDF](https://arxiv.org/pdf/1612.01474)


### #616: When Worlds Collide: Integrating Different Counterfactual Assumptions in Fairness

### #617: Matrix Norm Estimation from a Few Entries

### #618: Deep Networks for Decoding Natural Images from Retinal Signals

### #619: Causal Effect Inference with Deep Latent Variable Models
_Christos Louizos,  Uri Shalit,  Joris Mooij,  David Sontag,  Richard Zemel,  Max Welling_

Learning individual-level causal effects from observational data, such as inferring the most effective medication for a specific patient, is a problem of growing importance for policy makers. The most important aspect of inferring causal effects from observational data is the handling of confounders, factors that affect both an intervention and its outcome. A carefully designed observational study attempts to measure all important confounders. However, even if one does not have direct access to all confounders, there may exist noisy and uncertain measurement of proxies for confounders. We build on recent advances in latent variable modelling to simultaneously estimate the unknown latent space summarizing the confounders and the causal effect. Our method is based on Variational Autoencoders (VAE) which follow the causal structure of inference with proxies. We show our method is significantly more robust than existing methods, and matches the state-of-the-art on previous benchmarks focused on individual treatment effects.
[Abstract](https://arxiv.org/abs/1705.08821), [PDF](https://arxiv.org/pdf/1705.08821)


### #620: Learning Identifiable Gaussian Bayesian Networks in Polynomial Time and Sample Complexity
_Asish Ghoshal,  Jean Honorio_

Learning the directed acyclic graph (DAG) structure of a Bayesian network from observational data is a notoriously difficult problem for which many hardness results are known. In this paper we propose a provably polynomial-time algorithm for learning sparse Gaussian Bayesian networks with equal noise variance --- a class of Bayesian networks for which the DAG structure can be uniquely identified from observational data --- under high-dimensional settings. We show that $O(k^4 \log p)$ number of samples suffices for our method to recover the true DAG structure with high probability, where $p$ is the number of variables and $k$ is the maximum Markov blanket size. We obtain our theoretical guarantees under a condition called Restricted Strong Adjacency Faithfulness, which is strictly weaker than strong faithfulness --- a condition that other methods based on conditional independence testing need for their success. The sample complexity of our method matches the information-theoretic limits in terms of the dependence on $p$. We show that our method out-performs existing state-of-the-art methods for learning Gaussian Bayesian networks in terms of recovering the true DAG structure while being comparable in speed to heuristic methods.
[Abstract](https://arxiv.org/abs/1703.01196), [PDF](https://arxiv.org/pdf/1703.01196)


### #621: Gradient Episodic Memory for Continuum Learning
_David Lopez-Paz,  Marc'Aurelio Ranzato_

One major obstacle towards artificial intelligence is the poor ability of models to quickly solve new problems, without forgetting previously acquired knowledge. To better understand this issue, we study the problem of continual learning, where the model observes, once and one by one, examples concerning a sequence of tasks. First, we propose a set of metrics to evaluate models learning over a continuum of data. These metrics characterize models not only by their test accuracy, but also in terms of their ability to transfer knowledge across tasks. Second, we propose a model for continual learning, called Gradient of Episodic Memory (GEM), which alleviates forgetting while allowing beneficial transfer of knowledge to previous tasks. Our experiments on variants of MNIST and CIFAR-100 demonstrate the strong performance of GEM when compared to the state-of-the-art.
[Abstract](https://arxiv.org/abs/1706.08840), [PDF](https://arxiv.org/pdf/1706.08840)


### #622: Radon Machines: Effective Parallelisation for Machine Learning

### #623: Semisupervised Clustering, AND-Queries and Locally Encodable Source Coding

### #624: Clustering Stable Instances of Euclidean k-means.

### #625: Good Semi-supervised Learning That Requires a Bad GAN

### #626: On Blackbox Backpropagation and Jacobian Sensing

### #627: Protein Interface Prediction using Graph Convolutional Networks

### #628: Solid Harmonic Wavelet Scattering: Predicting Quantum Molecular Energy from Invariant Descriptors of 3D  Electronic Densities

### #629: Towards Generalization and Simplicity in Continuous Control
_Aravind Rajeswaran,  Kendall Lowrey,  Emanuel Todorov,  Sham Kakade_

This work shows that policies with simple linear and RBF parameterizations can be trained to solve a variety of continuous control tasks, including the OpenAI gym benchmarks. The performance of these trained policies are competitive with state of the art results, obtained with more elaborate parameterizations such as fully connected neural networks. Furthermore, existing training and testing scenarios are shown to be very limited and prone to over-fitting, thus giving rise to only trajectory-centric policies. Training with a diverse initial state distribution is shown to produce more global policies with better generalization. This allows for interactive control scenarios where the system recovers from large on-line perturbations; as shown in the supplementary video.
[Abstract](https://arxiv.org/abs/1703.02660), [PDF](https://arxiv.org/pdf/1703.02660)


### #630: Random Projection Filter Bank for Time Series Data

### #631: Filtering Variational Objectives
_Chris J. Maddison,  Dieterich Lawson,  George Tucker,  Nicolas Heess,  Mohammad Norouzi,  Andriy Mnih,  Arnaud Doucet,  Yee Whye Teh_

When used as a surrogate objective for maximum likelihood estimation in latent variable models, the evidence lower bound (ELBO) produces state-of-the-art results. Inspired by this, we consider the extension of the ELBO to a family of lower bounds defined by a particle filter's estimator of the marginal likelihood, the filtering variational objectives (FIVOs). FIVOs take the same arguments as the ELBO, but can exploit a model's sequential structure to form tighter bounds. We present results that relate the tightness of FIVO's bound to the variance of the particle filter's estimator by considering the generic case of bounds defined as log-transformed likelihood estimators. Experimentally, we show that training with FIVO results in substantial improvements over training with ELBO on sequential data.
[Abstract](https://arxiv.org/abs/1705.09279), [PDF](https://arxiv.org/pdf/1705.09279)


### #632: On Frank-Wolfe and Equilibrium Computation

### #633: Modulating early visual processing by language
_Harm de Vries,  Florian Strub,  Jérémie Mary,  Hugo Larochelle,  Olivier Pietquin,  Aaron Courville_

It is commonly assumed that language refers to high-level visual concepts while leaving low-level visual processing unaffected. This view dominates the current literature in computational models for language-vision tasks, where visual and linguistic input are mostly processed independently before being fused into a single representation. In this paper, we deviate from this classic pipeline and propose to modulate the \emph{entire visual processing} by linguistic input. Specifically, we condition the batch normalization parameters of a pretrained residual network (ResNet) on a language embedding. This approach, which we call MOdulated RESnet (\MRN), significantly improves strong baselines on two visual question answering tasks. Our ablation study shows that modulating from the early stages of the visual processing is beneficial.
[Abstract](https://arxiv.org/abs/1707.00683), [PDF](https://arxiv.org/pdf/1707.00683)


### #634: Learning Mixture of Gaussians with Streaming Data
_Aditi Raghunathan,  Ravishankar Krishnaswamy,  Prateek Jain_

In this paper, we study the problem of learning a mixture of Gaussians with streaming data: given a stream of $N$ points in $d$ dimensions generated by an unknown mixture of $k$ spherical Gaussians, the goal is to estimate the model parameters using a single pass over the data stream. We analyze a streaming version of the popular Lloyd's heuristic and show that the algorithm estimates all the unknown centers of the component Gaussians accurately if they are sufficiently separated. Assuming each pair of centers are $C\sigma$ distant with $C=\Omega((k\log k)^{1/4}\sigma)$ and where $\sigma^2$ is the maximum variance of any Gaussian component, we show that asymptotically the algorithm estimates the centers optimally (up to constants); our center separation requirement matches the best known result for spherical Gaussians \citep{vempalawang}. For finite samples, we show that a bias term based on the initial estimate decreases at $O(1/{\rm poly}(N))$ rate while variance decreases at nearly optimal rate of $\sigma^2 d/N$. Our analysis requires seeding the algorithm with a good initial estimate of the true cluster centers for which we provide an online PCA based clustering algorithm. Indeed, the asymptotic per-step time complexity of our algorithm is the optimal $d\cdot k$ while space complexity of our algorithm is $O(dk\log k)$. In addition to the bias and variance terms which tend to $0$, the hard-thresholding based updates of streaming Lloyd's algorithm is agnostic to the data distribution and hence incurs an approximation error that cannot be avoided. However, by using a streaming version of the classical (soft-thresholding-based) EM method that exploits the Gaussian distribution explicitly, we show that for a mixture of two Gaussians the true means can be estimated consistently, with estimation error decreasing at nearly optimal rate, and tending to $0$ for $N\rightarrow \infty$.
[Abstract](https://arxiv.org/abs/1707.02391), [PDF](https://arxiv.org/pdf/1707.02391)


### #635: Practical Hash Functions for Similarity Estimation and Dimensionality Reduction

### #636: Two Time-Scale Update Rule for Generative Adversarial Nets

### #637: The Scaling Limit of High-Dimensional Online Independent Component Analysis

### #638: Approximation Algorithms for $\ell_0$-Low Rank Approximation

### #639: The power of absolute discounting: all-dimensional distribution estimation

### #640: Supervised Adversarial Domain Adaptation

### #641: Spectral Mixture Kernels for Multi-Output Gaussian Processes
_Gabriel Parra,  Felipe Tobar_

Initially, multiple-output Gaussian processes models (MOGPs) were constructed as linear combinations of independent, latent, single-output Gaussian processes (GPs). This resulted in cross-covariance functions with limited parametric interpretation, thus conflicting with single-output GPs and their intuitive understanding of lengthscales, frequencies and magnitudes to name but a few. On the contrary, current approaches to MOGP are able to better interpret the relationship between different channels by directly modelling the cross-covariances as a spectral mixture kernel with a phase shift. We propose a parametric family of complex-valued crossspectral densities and then build on Cramer's Theorem, the multivariate version of Bochner's Theorem, to provide a principled approach to design multivariate covariance functions. The so-constructed kernels are able to model delays among channels in addition to phase differences and are thus more expressive than previous methods, while also providing full parametric interpretation of the relationship across channels. The proposed method is first validated on synthetic data and then compared to existing MOGP methods on two real-world examples.
[Abstract](https://arxiv.org/abs/1709.01298), [PDF](https://arxiv.org/pdf/1709.01298)


### #642: Neural Expectation Maximization
_Klaus Greff,  Sjoerd van Steenkiste,  Jürgen Schmidhuber_

Many real world tasks such as reasoning and physical interaction require identification and manipulation of conceptual entities. A first step towards solving these tasks is the automated discovery of distributed symbol-like representations. In this paper, we explicitly formalize this problem as inference in a spatial mixture model where each component is parametrized by a neural network. Based on the Expectation Maximization framework we then derive a differentiable clustering method that simultaneously learns how to group and represent individual entities. We evaluate our method on the (sequential) perceptual grouping task and find that it is accurately able to recover the constituent objects. We demonstrate that the learned representations are useful for predictive coding.
[Abstract](https://arxiv.org/abs/1708.03498), [PDF](https://arxiv.org/pdf/1708.03498)


### #643: Online Learning of Linear Dynamical Systems

### #644: Z-Forcing: Training Stochastic Recurrent Networks

### #645: Thalamus Gated Recurrent Modules

### #646: Neural Variational Inference and Learning in Undirected Graphical Models
_Andriy Mnih,  Karol Gregor_

Highly expressive directed latent variable models, such as sigmoid belief networks, are difficult to train on large datasets because exact inference in them is intractable and none of the approximate inference methods that have been applied to them scale well. We propose a fast non-iterative approximate inference method that uses a feedforward network to implement efficient exact sampling from the variational posterior. The model and this inference network are trained jointly by maximizing a variational lower bound on the log-likelihood. Although the naive estimator of the inference model gradient is too high-variance to be useful, we make it practical by applying several straightforward model-independent variance reduction techniques. Applying our approach to training sigmoid belief networks and deep autoregressive networks, we show that it outperforms the wake-sleep algorithm on MNIST and achieves state-of-the-art results on the Reuters RCV1 document dataset.
[Abstract](https://arxiv.org/abs/1402.0030), [PDF](https://arxiv.org/pdf/1402.0030)


### #647: Subspace Clustering via Tangent Cones

### #648: The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process
_Hongyuan Mei,  Jason Eisner_

Many events occur in the world. Some event types are stochastically excited or inhibited---in the sense of having their probabilities elevated or decreased---by patterns in the sequence of previous events. Discovering such patterns can help us predict which type of event will happen next and when. We propose to model streams of discrete events in continuous time, by constructing a neurally self-modulating multivariate point process in which the intensities of multiple event types evolve according to a novel continuous-time LSTM. This generative model allows past events to influence the future in complex and realistic ways, by conditioning future event intensities on the hidden state of a recurrent neural network that has consumed the stream of past events. Our model has desirable qualitative properties. It achieves competitive likelihood and predictive accuracy on real and synthetic datasets, including under missing-data conditions.
[Abstract](https://arxiv.org/abs/1612.09328), [PDF](https://arxiv.org/pdf/1612.09328)


### #649: Inverse Reward Design

### #650: Structured Bayesian Pruning via Log-Normal Multiplicative Noise
_Kirill Neklyudov,  Dmitry Molchanov,  Arsenii Ashukha,  Dmitry Vetrov_

Dropout-based regularization methods can be regarded as injecting random noise with pre-defined magnitude to different parts of the neural network during training. It was recently shown that Bayesian dropout procedure not only improves generalization but also leads to extremely sparse neural architectures by automatically setting the individual noise magnitude per weight. However, this sparsity can hardly be used for acceleration since it is unstructured. In the paper, we propose a new Bayesian model that takes into account the computational structure of neural networks and provides structured sparsity, e.g. removes neurons and/or convolutional channels in CNNs. To do this, we inject noise to the neurons outputs while keeping the weights unregularized. We established the probabilistic model with a proper truncated log-uniform prior over the noise and truncated log-normal variational approximation that ensures that the KL-term in the evidence lower bound is computed in closed-form. The model leads to structured sparsity by removing elements with a low SNR from the computation graph and provides significant acceleration on a number of deep neural architectures. The model is very easy to implement as it only corresponds to the addition of one dropout-like layer in computation graph.
[Abstract](https://arxiv.org/abs/1705.07283), [PDF](https://arxiv.org/pdf/1705.07283)


### #651: Attend and Predict: Understanding Gene Regulation by Selective Attention on Chromatin
_Ritambhara Singh,  Jack Lanchantin,  Arshdeep Sekhon,  Yanjun Qi_

The past decade has seen a revolution in genomic technologies that enable a flood of genome-wide profiling of chromatin marks. Recent literature tried to understand gene regulation by predicting gene expression from large-scale chromatin measurements. Two fundamental challenges exist for such learning tasks: (1) genome-wide chromatin signals are spatially structured, high-dimensional and highly modular; and (2) the core aim is to understand what are the relevant factors and how they work together? Previous studies either failed to model complex dependencies among input signals or relied on separate feature analysis to explain the decisions. This paper presents an attention-based deep learning approach; we call AttentiveChrome, that uses a unified architecture to model and to interpret dependencies among chromatin factors for controlling gene regulation. AttentiveChrome uses a hierarchy of multiple Long short-term memory (LSTM) modules to encode the input signals and to model how various chromatin marks cooperate automatically. AttentiveChrome trains two levels of attention jointly with the target prediction, enabling it to attend differentially to relevant marks and to locate important positions per mark. We evaluate the model across 56 different cell types (tasks) in human. Not only is the proposed architecture more accurate, but its attention scores also provide a better interpretation than state-of-the-art feature visualization methods such as saliency map. Code and data are shared at www.deepchrome.org
[Abstract](https://arxiv.org/abs/1708.00339), [PDF](https://arxiv.org/pdf/1708.00339)


### #652: Acceleration and Averaging in Stochastic Descent Dynamics
_Walid Krichene,  Peter L. Bartlett_

We formulate and study a general family of (continuous-time) stochastic dynamics for accelerated first-order minimization of smooth convex functions. Building on an averaging formulation of accelerated mirror descent, we propose a stochastic variant in which the gradient is contaminated by noise, and study the resulting stochastic differential equation. We prove a bound on the rate of change of an energy function associated with the problem, then use it to derive estimates of convergence rates of the function values, (a.s. and in expectation) both for persistent and asymptotically vanishing noise. We discuss the interaction between the parameters of the dynamics (learning rate and averaging weights) and the covariation of the noise process, and show, in particular, how the asymptotic rate of covariation affects the choice of parameters and, ultimately, the convergence rate.
[Abstract](https://arxiv.org/abs/1707.06219), [PDF](https://arxiv.org/pdf/1707.06219)


### #653: Kernel functions based on triplet comparisons

### #654: An Error Detection and Correction Framework for Connectomics
_Jonathan Zung,  Ignacio Tartavull,  H. Sebastian Seung_

Significant advances have been made in recent years on the problem of neural circuit reconstruction from electron microscopic imagery. Improvements in image acquisition, image alignment, and boundary detection have greatly reduced the achievable error rate. In order to make further progress, we argue that automated error detection is essential for focusing the effort and attention of both human and machine. In this paper, we report on the use of automated error detection as an attention signal for a flood filling error correction module. We demonstrate significant improvements upon the state of the art in segmentation performance.
[Abstract](https://arxiv.org/abs/1708.02599), [PDF](https://arxiv.org/pdf/1708.02599)


### #655: Style Transfer from Non-parallel Text by Cross-Alignment

### #656: Cross-Spectral Factor Analysis

### #657: Stochastic Submodular Maximization: The Case of Coverage Functions

### #658: On Distributed Hierarchical Clustering

### #659: Unsupervised Transformation Learning via Convex Relaxations

### #660: A Sharp Error Analysis for the Fused Lasso, with Implications to Broader Settings  and Approximate Screening

### #661: Efficient Computation of Moments in Sum-Product Networks
_Han Zhao,  Geoff Gordon_

Bayesian online learning algorithms for Sum-Product Networks (SPNs) need to compute moments of model parameters under the one-step update posterior distribution. The best existing method for computing such moments scales quadratically in the size of the SPN, although it scales linearly for trees. We propose a linear-time algorithm that works even when the SPN is a directed acyclic graph (DAG). We achieve this goal by reducing the moment computation problem into a joint inference problem in SPNs and by taking advantage of a special structure of the one-step update posterior distribution: it is a multilinear polynomial with exponentially many monomials, and we can evaluate moments by differentiating. The latter is known as the \emph{differential trick}. We apply the proposed algorithm to develop a linear time assumed density filter (ADF) for SPN parameter learning. As an additional contribution, we conduct extensive experiments comparing seven different online learning algorithms for SPNs on 20 benchmark datasets. The new linear-time ADF method consistently achieves low runtime due to the efficient linear-time algorithm for moment computation; however, we discover that two other methods (CCCP and SMA) typically perform better statistically, while a third (BMM) is comparable to ADF. Interestingly, CCCP can be viewed as implicitly using the same differentiation trick that we make explicit here. The fact that two of the top four fastest methods use this trick suggests that the same trick might find other uses for SPN learning in the future.
[Abstract](https://arxiv.org/abs/1702.04767), [PDF](https://arxiv.org/pdf/1702.04767)


### #662: A Meta-Learning Perspective on Cold-Start Recommendations for Items

### #663: Predicting Scene Parsing and Motion Dynamics in the Future

### #664: Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference

### #665: Efficient Approximation Algorithms for Strings Kernel Based Sequence Classification

### #666: Kernel Feature Selection via Conditional Covariance Minimization
_Jianbo Chen,  Mitchell Stern,  Martin J. Wainwright,  Michael I. Jordan_

We propose a framework for feature selection that employs kernel-based measures of independence to find a subset of covariates that is maximally predictive of the response. Building on past work in kernel dimension reduction, we formulate our approach as a constrained optimization problem involving the trace of the conditional covariance operator, and additionally provide some consistency results. We then demonstrate on a variety of synthetic and real data sets that our method compares favorably with other state-of-the-art algorithms.
[Abstract](https://arxiv.org/abs/1707.01164), [PDF](https://arxiv.org/pdf/1707.01164)


### #667:  Statistical Convergence Analysis of Gradient EM on General Gaussian Mixture Models

### #668: Real Time Image Saliency for Black Box Classifiers
_Piotr Dabkowski,  Yarin Gal_

In this work we develop a fast saliency detection method that can be applied to any differentiable image classifier. We train a masking model to manipulate the scores of the classifier by masking salient parts of the input image. Our model generalises well to unseen images and requires a single forward pass to perform saliency detection, therefore suitable for use in real-time systems. We test our approach on CIFAR-10 and ImageNet datasets and show that the produced saliency maps are easily interpretable, sharp, and free of artifacts. We suggest a new metric for saliency and test our method on the ImageNet object localisation task. We achieve results outperforming other weakly supervised methods.
[Abstract](https://arxiv.org/abs/1705.07857), [PDF](https://arxiv.org/pdf/1705.07857)


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

