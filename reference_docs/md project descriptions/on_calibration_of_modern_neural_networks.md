# Reference Analysis: On Calibration of Modern Neural Networks

## 1. Core Summary
* **Definition of Perfect Calibration:** Defines calibration as the statistical alignment between a model's predicted probability scores ($\hat{P}$) and true correctness frequencies. Formally, perfect calibration is expressed as:

$$\mathbb{P}(\hat{Y} = Y \mid \hat{P} = p) = p, \quad \forall p \in [0, 1]$$

* **The Miscalibration Trend:** Uncovers a surprising trend: modern deep networks (such as a 110-layer ResNet) are significantly more miscalibrated and overconfident than historical architectures (such as a 5-layer LeNet), despite achieving superior overall generalization accuracy.
* **Architectural Drivers:** Identifies through controlled empirical testing that increased model depth, layer width, the addition of Batch Normalization layers, and a generalized reduction in L2 weight decay parameters are critical factors contributing to increased calibration error.
* **Loss Mismatch Analysis:** Explains that networks systematically overfit to the Negative Log-Likelihood (NLL) optimization objective during late training phases. This overfit manifests as inflated logit values that drive down NLL losses by maximizing confidence, even as the 0/1 classification error stabilizes.
* **The Practical Remedy:** Evaluates various non-parametric and parametric post-processing calibration methods (such as Histogram Binning, Isotonic Regression, and Matrix Scaling), demonstrating that **Temperature Scaling** is consistently the most memory-efficient and effective technique across common datasets.

---

## 2. Mathematical Formalizations

### Empirical Approximations via Binning
Because confidence parameter values are continuous, validation sets containing finite numbers of samples ($n$) are partitioned into $M$ equally-spaced intervals where each interval $I_m = (\frac{m-1}{M}, \frac{m}{M}]$. Let $B_m$ represent the set of sample indices whose prediction confidences fall directly within interval $I_m$.

The empirical accuracy of a specific bin $B_m$ is calculated as:

$$acc(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$

where $\hat{y}_i$ and $y_i$ represent the predicted and true class labels for sample $i$ respectively.

The average confidence within a specific bin $B_m$ is defined as:

$$conf(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

where $\hat{p}_i$ is the confidence value for sample $i$. For perfectly calibrated systems, $acc(B_m) = conf(B_m)$ across all bins.

### Calibration Error Metrics
* **Expected Calibration Error (ECE):** Measures the absolute difference between empirical accuracy and average confidence across all bins, weighted by the sample density of each bin:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$

* **Maximum Calibration Error (MCE):** Focuses on the worst-case deviation between accuracy and confidence parameters across all bins, critical for high-risk domains:

$$\text{MCE} = \max_{m \in \{1, \dots, M\}} |acc(B_m) - conf(B_m)|$$

### Optimization Target: Negative Log-Likelihood (NLL)
Standard cross-entropy optimization targets the conditional distribution quality via:

$$\mathcal{L} = -\sum_{i=1}^{n} \log(\hat{\pi}(y_i \mid x_i))$$

### Parametric Logit Scaling: Temperature Scaling
As an extension of Platt scaling, Temperature Scaling utilizes a single positive scalar variable $T > 0$ to scale the logit vector $\mathbf{z}_i$ prior to applying the final multi-class activation. The newly calibrated confidence distribution is defined as:

$$\hat{q}_i = \max_{k} \sigma_{\text{SM}}(\mathbf{z}_i / T)^{(k)}$$

The optimized temperature value $T$ is found by minimizing the NLL loss specifically over a held-out validation set, leaving the model's underlying weights completely frozen. Because the scale factor $T$ acts uniformly across all classes, the index ordering of the logit maximum remains unaltered, ensuring classification accuracy is completely preserved.

---

## 3. Comparative Distinctions & Enhancements
* **Domain & Task Divergence:** The reference text evaluates fixed-dimension, single-label categorical logit spaces (e.g., ImageNet, CIFAR-100, 20 News). This project scales calibration assessments up to open-ended autoregressive Vision-Language Model decoder configurations processing token sequences for visual question answering.
* **Static Post-Hoc vs. Generation-Time Inspection:** Temperature scaling is completely post-hoc, requiring a pre-computed validation set to fit a single optimization scalar parameter globally. Your project introduces a diagnostic methodology that inspects generation-time variations across multiple text outputs directly at the prompt level.
* **Epistemic vs. Aleatoric Focus:** Logit smoothing primarily corrects marginal probability distributions over static features. By altering the quantity of information accessible via input token limits ($m$), your project measures epistemic variation, checking how stable a prediction is when parts of the visual input are structurally removed.
