---
layout: center
---
# GAN Flavors

---

# GAN Flavors

<div class="grid grid-cols-[3fr_9fr] gap-10">
<div>

##### Nice website: [www.connectedpapers.com](https://www.connectedpapers.com)
<br>

##### 🦓 [GAN-Zoos](https://happy-jihye.github.io/gan/)
<br>

##### Implementation of GAN models in PyTorch:<br> [pytorch-gan-zoo](https://pypi.org/project/pytorch-gan-zoo/)
</div>
<div>
  <figure>
    <img src="/GANs_connected_papers.png" style="width: 450px !important;">
  </figure>
</div>
</div>

---

# GAN (again)

* Divergence minimization perspective
* Standard GAN formulation correspond to minimizing the [Jensen–Shannon divergence (JSD)](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
  * JSD is a symmetrized and smoothed version of the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

$$\min_{\textcolor{#6C8EBF}{G}} \max_{\textcolor{#B85450}{D}} L(\textcolor{#B85450}{D},\textcolor{#6C8EBF}{G}) = \mathbb{E}_{x \sim p_{data}(x)} \big[ \log \textcolor{#B85450}{D}(x)\big] + \mathbb{E}_{z \sim p_{z}(z)} \bigg[ \log \big(1 - \textcolor{#B85450}{D}(\textcolor{#6C8EBF}{G}(z))\big)\bigg]$$

* $\min\limits_{G}$ leads to:
  * $\min\limits_{p_g} JS (p_g \lVert p_{data}) - \log{4}$

<br>

##### Note: $- \log{4} = \mathbb{E}_{x \sim p_{data}(x)} \big[ - \log{2} \big] + \mathbb{E}_{x \sim p_g} \big[ - \log{2} \big]$ as training criterion $C(G) = \log \frac{1}{2} + \log \frac{1}{2} = - \log{4}$

---

# KL Divergence Playground <a href="https://gnarlyware.com/blog/kl-divergence-online-demo/">[link]</a>

<iframe src="https://gnarlyware.com/blog/kl-divergence-online-demo/" width="1100" height="550" style="-webkit-transform:scale(0.8);-moz-transform-scale(0.8); position: relative; top: -65px; left: -120px"></iframe>

---

# Wasserstein GAN

* Proposed by [Arjovsky et al. (2017)](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf)

* Motivated by the comparisons of “distance” between distributions:
  * $KL (p \lVert q) = \int\limits_x \log \big( \frac{p(x)}{q(x)} \big) p(x) dx$
  * $JS (p \lVert q) := KL \big(p \lVert \frac{p+q}{2} \big) + KL \big(q \lVert \frac{p+q}{2} \big)$
  * $W (p \lVert q) = \inf\limits_{\gamma \in \Pi(p,q)} \mathbb{E}_{x,y \sim \gamma}\big[ \lVert x - y \rVert \big])$
    * $W$ is known as [Earth mover's distance (EMD)](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) or  [Wasserstein metric $W_1$](https://en.wikipedia.org/wiki/Wasserstein_metric)

<br>
<br>
<br>
<br>
<br>

##### WGAN-related slides are based on [Gauthier Gidel's slides](https://gauthiergidel.github.io/ift_6756_gt_ml/slides/Lecture11.pdf)

---

# What is Wasserstein (Earth Mover’s) distance?

#### **Meaning**: minimum energy cost of moving and transforming a pile of dirt in the shape of one probability distribution to the shape of the other distribution.

* If we label the cost to pay to make $P_i$ and $Q_i$ match as $\delta_i$,<br> we would have $\delta_{i+1} = \delta_i + P_i - Q_i$ and in the example:

<br>
<div class="grid grid-cols-[3fr_6fr] gap-10">
<div>

#### $\delta_0 = 0$
#### $\delta_1 = 0 + 3 - 1 = 2$
#### $\delta_2 = 2 + 2 - 2 = 2$
#### $\delta_3 = 2 + 1 - 4 = -1$
#### $\delta_4 = -1 + 4 - 3 = 0$
<br>

#### Finally the Earth Mover’s distance is $W = \sum \lvert \delta_i \rvert = 5$
</div>
<div>
  <figure>
    <img src="/shovelfuls.png" style="width: 590px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://lilianweng.github.io/posts/2017-08-20-gan/">https://lilianweng.github.io/posts/2017-08-20-gan/</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Warm-up in Dimension 1

<br>
<div class="grid grid-cols-[3fr_3fr] gap-10">
<div>

#### $Z \sim U([0,1])$
<br>

#### $p_{\mathrm{target}} \sim (0, Z)$
<br>
  <figure>
    <img src="/WD.png" style="width: 270px !important;">
  </figure>
<br>

#### $W (p \lVert q) = \inf\limits_{\gamma \in \Pi(p,q)} \mathbb{E}_{x,y \sim \gamma}\big[ \lVert x - y \rVert \big])$
</div>
<div>

#### $g_\theta (z) = (\theta, z)$
<br>

#### $q_\theta \sim (\theta, Z)$
<br>
  <figure>
    <img src="/JSD.png" style="width: 270px !important;">
  </figure>
<br>

#### $JS (p \lVert q) := KL \big(p \lVert \frac{p+q}{2} \big) + KL \big(q \lVert \frac{p+q}{2} \big)$
</div>
</div>

---

# Motivation for Wasserstein Distance

#### **Theorem 1.** Let $\mathbb{P}_r$ be a fixed distribution over $\mathcal{X}$. Let $Z$ be a random variable (e.g Gaussian) over another space $\mathcal{Z}$. Let $\mathbb{P}_\theta$ denote the distribution of $g_\theta (Z)$,<br> where $g: (z, \theta) \in \mathcal{Z} \times \mathbb{R}^d \rightarrow g_\theta \in \mathcal{X}$. Then,
1. If $g$ is continuous in $\theta$, so is $W (\mathbb{P}_r, \mathbb{P}_\theta)$.
2. If $g$ is locally Lipschitz and satisfies regularity assumption 1, then $W (\mathbb{P}_r, \mathbb{P}_\theta)$ is continuous everywhere, and differentiable almost everywhere.
3. Statements 1-2 are false for the Jensen-Shannon divergence $JS(\mathbb{P}_r, \mathbb{P}_\theta)$ and all the $KLs$.

<br>
<div class="grid grid-cols-[3fr_3fr] gap-10">
<div>
  <figure>
    <img src="/WD.png" style="width: 170px !important;">
  </figure>

##### **Gradients**
</div>
<div>
  <figure>
    <img src="/JSD.png" style="width: 170px !important;">
  </figure>

##### **No Gradients**
</div>
</div>

---

# Dual Formulation

* Maximum in GANs is a divergence
$$JS (p_g \lVert p_d) = \max_D \mathbb{E}_{x \sim p_{data}} \big[ \log x\big] + \mathbb{E}_{x^\prime \sim p_{g}} \big[ \log \big(1 - D(x^\prime)\big)\big]$$

<br>

* Wasserstein can be written as following:
$$W (p_g, p_d) = \max_{\lVert F \rVert_L \leq 1} \mathbb{E}_{x \sim p_d} \big[ F(x) \big] - \mathbb{E}_{x^\prime \sim p_g} \big[ F(x^\prime) \big]$$

<br>
<br>
<br>

* Question: How close are these objectives?

---

# Dual Formulation

### How close are $W$ and $JS$? $D(x) = \sigma(F(x))$

$$JS (p_g \lVert p_d) = \max_D \mathbb{E}_{x \sim p_{data}} \big[ - \log ~(1 + e^{-F(x)}) \big] + \mathbb{E}_{x^\prime \sim p_{g}} \big[ \log ~(1 + e^{F(x^\prime)}) \big]$$

<br>

* $JS (p_g \lVert p_d) = \max\limits_F \mathbb{E}_{x \sim p_{data}} \big[ \lfloor F(x) \rfloor_{\textcolor{red}{-}} \big] - \mathbb{E}_{x^\prime \sim p_{g}} \big[ \lfloor F(x) \rfloor_{\textcolor{red}{+}} \big]$
  * If $F$ gets too good: Vanishing gradients for $G$

<br>

* $W (p_g, p_d) = \max\limits_{\textcolor{red}{\lVert F \rVert_L \leq 1}} \mathbb{E}_{x \sim p_d} \big[ F(x) \big] - \mathbb{E}_{x^\prime \sim p_g} \big[ F(x^\prime) \big]$
  * If $F$ cannot get “too” good

---

# Real New thing in WGAN: Lipschitz Constraint

* **Intuition**: Prevent discriminator to make gradient explode
  * because it cannot discriminate arbitrarily well

* **Question**: How do I enforce Discrimintor to be 1-Lipschitz function?

<div class="grid grid-cols-[5fr_2fr] gap-5">
<div>

* **Answer 1**: Not practical (at least exactly)
* **Answer 2**: Approximation:
  * Clipping (WGAN)
    * Very rough approximation
  * Gradient Penalty (WGAN-GP)
    * Better but harder to explicitly control the Lipschitz
  * Spectral Normalization (SN-GAN)
    * Explicit control... still an approximation
</div>
<div>
  <br>
  <figure>
    <img src="/Lipschitz_Visualisierung.gif" style="width: 300px !important;">
        <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source:
      <a href="https://en.wikipedia.org/wiki/Lipschitz_continuity">https://en.wikipedia.org/wiki/Lipschitz_continuity</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Clipping

<div class="grid grid-cols-[3fr_4fr] gap-15">
<div>

* **Idea**: a NN with bounded weights is Lipschitz
<br>

* **Pros**:
  * Fast to compute
  * Simple to implement
* **Cons**:
  * Does not control the Lipchitz well
    * Very rough approximation
  * Ex: $f(x) = \theta_L \cdot ... \cdot \theta_1 \cdot x$
</div>
<div>
  <figure>
    <img src="/WGAN_algorithm_1.png" style="width: 400px !important;">
  </figure>
</div>
</div>

---

# Gradient Penalty

* **Idea**: Bounded gradient is equivalent to Lipschitz<br>
$$\tilde{\mathcal{L}}_D = \mathcal{L}_D + \lambda \mathbb{E}_{\tilde{x} \sim \epsilon P_d + (1 - \epsilon) p_g} \big[ \big( \lVert \nabla_x D(\tilde{x}) \rVert_2 -1 \big)^2\big]$$
  * Incentive: Gradients of D close to 1
* **Pros**:
  * Tractable
  * Simple to implement
* **Cons**:
  * Does not control the Lipchitz explicitly
    * Very rough approximation
  * Only care about the Lipchitz on the supports of the distributions
  * Large $\lambda$ creates bad attractive points

---

# Spectral Normalization

* **Idea**:Compute an upper-bound on Lipschitz<br>
$$\lVert \sigma \big( W_L \cdots \sigma (W_1 x) \big) \rVert_{\mathrm{Lipschitz}} \leq \lVert W_L \rVert \cdots \lVert W_1 \rVert$$
<br>

* **Pros**:
  * Give better results
      * better control of the Lipchitz
* **Cons**:
  * Harder to implement
    * Very rough approximation
  * Still an approximation of the upper bound.