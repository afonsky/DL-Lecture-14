---
layout: center
---
# Generative Adversarial Networks (GANs)

---

<Youtube id="9reHvktowLY" width="700" height="450" />

---

# Introduction to GANs

* Proposed by [I. Goodfellow et al. (2014)](http://papers.nips.cc/paper/5423-generative-adversarial-nets)
* The original purpose is to **generate new data**
* Classically for generating new images, but applicable to wide range of domains
* Learns the training set distribution and can generate new images that have never been seen before
* In contrast to e.g., autoregressive models or RNNs (generating one word at a time), GANs generate the whole
output all at once
<br>
<br>
<br>
<br>

##### GAN-related slides are based on [Sebastian Raschka's slides](https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L17_gan_slides.pdf) and on [Gauthier Gidel's slides](https://gauthiergidel.github.io/ift_6756_gt_ml/slides/Lecture7.pdf)

---

# Introduction to GANs
<br>
<br>
<br>

#### How authentic is this image? 
  <figure>
    <img src="/GANs_1.png" style="width: 600px !important;">
    <figcaption style="font-size: 16px; position: absolute; right: 350px"> p(y = "real image" | x')
    </figcaption>   
  </figure>

---

# Introduction to GANs
<br>

#### $z \sim \mathcal{N}(0,1)$<br> or<br>  $z \sim \mathcal{U}(-1,1)$
  <figure>
    <img src="/GANs_2.png" style="width: 600px !important;">
    <figcaption style="font-size: 16px; position: absolute; right: 350px"> p(y = "real image" | x')
    </figcaption>
  </figure>

---

# Adversarial game
<br>
<br>

#### **Discriminator**: learns to become better as distinguishing real from generated images
<br>

  <figure>
    <img src="/GANs_3.png" style="width: 600px !important;">
  </figure>

#### **Generator**: learns to generate better images to fool the discriminator

---

# Why Are GANs Are Called Generative Models?

* The generative part comes from the fact that the model "generates" new data
* Usually, generative models use an approximation to compute the usually intractable distribution
  * here, the discriminator part does that approximation
  * So, it does learn $p(x)$
* Vanilla GANs cannot do conditional inference, though

---

# When Does a GAN Converge?
<br>
<br>
<br>
<br>

  <figure>
    <img src="/GANs_3.png" style="width: 600px !important;">
  </figure>

---

# GAN Objective
<br>
<br>
<br>

$$\min_{\textcolor{#6C8EBF}{G}} \max_{\textcolor{#B85450}{D}} L(\textcolor{#B85450}{D},\textcolor{#6C8EBF}{G}) = \mathbb{E}_{x \sim p_{data}(x)} \big[ \log \textcolor{#B85450}{D}(x)\big] + \mathbb{E}_{z \sim p_{z}(z)} \bigg[ \log \big(1 - \textcolor{#B85450}{D}(\textcolor{#6C8EBF}{G}(z))\big)\bigg]$$

<br>

  <figure>
    <img src="/GANs_3.png" style="width: 600px !important;">
  </figure>

---

# GAN Discriminator Gradient
<br>

$$\min_{\textcolor{#6C8EBF}{G}} \max_{\textcolor{#B85450}{D}} L(\textcolor{#B85450}{D},\textcolor{#6C8EBF}{G}) = \mathbb{E}_{x \sim p_{data}(x)} \big[ \log \textcolor{#B85450}{D}(x)\big] + \mathbb{E}_{z \sim p_{z}(z)} \bigg[ \log \big(1 - \textcolor{#B85450}{D}(\textcolor{#6C8EBF}{G}(z))\big)\bigg]$$

### Discriminator gradient for update (gradient ascent):

$$\nabla \bm{w}_{\textcolor{#B85450}{D}} \frac{1}{n} \sum_{i=1}^n \bigg[ \log \textcolor{#B85450}{D} \bigg( \bm{x}^{(i)}\bigg) + \log \bigg( 1 - \textcolor{#B85450}{D}\big(\textcolor{#6C8EBF}{G}(z^{(i)})\big)\bigg) \bigg]$$

* $\textcolor{#B85450}{D} \big( \bm{x}^{(i)}\big)$ predict well on real images
  * probability close to $1$
* $\textcolor{#B85450}{D}\big(\textcolor{#6C8EBF}{G}(z^{(i)})\big)$ predict well on fake images
  * probability close to $0$

---

# GAN Generator Gradient
<br>

$$\min_{\textcolor{#6C8EBF}{G}} \max_{\textcolor{#B85450}{D}} L(\textcolor{#B85450}{D},\textcolor{#6C8EBF}{G}) = \mathbb{E}_{x \sim p_{data}(x)} \big[ \log \textcolor{#B85450}{D}(x)\big] + \mathbb{E}_{z \sim p_{z}(z)} \bigg[ \log \big(1 - \textcolor{#B85450}{D}(\textcolor{#6C8EBF}{G}(z))\big)\bigg]$$

### Generator gradient for update (gradient descent):

$$\nabla \bm{w}_{\textcolor{#6C8EBF}{G}} \frac{1}{n} \sum_{i=1}^n \log \bigg( 1 - \textcolor{#B85450}{D}\big(\textcolor{#6C8EBF}{G}(z^{(i)})\big)\bigg)$$

* $\textcolor{#B85450}{D}\big(\textcolor{#6C8EBF}{G}(z^{(i)})\big)$ predict badly on fake images
  * probability close to $1$

---

# GAN Algorithm for Minibatch Stochastic GD

<div class="grid grid-cols-[9fr_2fr] gap-2">
<div>
  <figure>
    <img src="/GANs_algorithm.png" style="width: 670px !important;">
  </figure>
</div>
<div>

##### From [Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets) by I. Goodfellow et al. (2014)
</div>
</div>

---

# GAN Convergence

* Converges when [Nash-equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) (a concept from Game Theory) is reached in the minmax (zero-sum) game

$$\min_{\textcolor{#6C8EBF}{G}} \max_{\textcolor{#B85450}{D}} L(\textcolor{#B85450}{D},\textcolor{#6C8EBF}{G}) = \mathbb{E}_{x \sim p_{data}(x)} \big[ \log \textcolor{#B85450}{D}(x)\big] + \mathbb{E}_{z \sim p_{z}(z)} \bigg[ \log \big(1 - \textcolor{#B85450}{D}(\textcolor{#6C8EBF}{G}(z))\big)\bigg]$$

* Nash-Equilibrium in Game Theory is reached when the actions of one player won't change depending on the opponent's actions
  * Here, this means that the GAN produces realistic images and the discriminator outputs random predictions (probabilities close to $0.5$)

---

# GAN Convergence

<div class="grid grid-cols-[9fr_2fr] gap-2">
<div>
  <figure>
    <img src="/GANs_convergence.png" style="width: 690px !important;">
  </figure>
</div>
<div>

##### From [Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets) by I. Goodfellow et al. (2014)
</div>
</div>

---

# GAN Training Problems

* Oscillation between generator and discriminator loss
* Mode collapse
  * generator produces examples of a particular kind only
* Discriminator is too strong
  * the gradient for the generator vanishes and the generator can't keep up
* Discriminator is too weak
  * the generator produces nonrealistic images that fool it too easily
    * rare problem, though

---

# GAN Training Problems

* Discriminator is too strong
  * the gradient for the generator vanishes and the generator can't keep up
  * Can be fixed as follows:
<br>
<br>

#### Instead of **gradient descent** with
$$\nabla \bm{w}_{\textcolor{#6C8EBF}{G}} \frac{1}{n} \sum_{i=1}^n \log \bigg( 1 - \textcolor{#B85450}{D}\big(\textcolor{#6C8EBF}{G}(z^{(i)})\big)\bigg)$$

#### Do **gradient ascent** with
$$\nabla \bm{w}_{\textcolor{#6C8EBF}{G}} \frac{1}{n} \sum_{i=1}^n \log \bigg(\textcolor{#B85450}{D}\big(\textcolor{#6C8EBF}{G}(z^{(i)})\big)\bigg)$$

---

# GAN Loss Function in Practice

* Discriminator
  * Maximize prediction probability of real as real and fake as fake
  * Remember maximizing log likelihood is the same as minimizing negative log likelihood (i.e., minimizing cross-entropy)
* Generator
  * Minimize likelihood of the discriminator to make correct predictions (predict fake as fake; real as real), which can be achieved by maximizing the cross-entropy
  * This doesn't work well in practice though because of gradient issues (zero gradient if the D makes correct predictions, which is not what we want for the G)
  * Better: flip labels and minimize cross entropy (force the D to output high probability for fake if an image is real, and high probability for real if an
image is fake)

---

# GAN Playground <a href="https://reiinakano.com/gan-playground/">[link]</a>

<iframe src="https://reiinakano.com/gan-playground/" width="1100" height="550" style="-webkit-transform:scale(0.8);-moz-transform-scale(0.8); position: relative; top: -65px; left: -120px"></iframe>