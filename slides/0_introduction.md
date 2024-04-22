# Generative Modeling

* Unsupervised learning
* **Goal**:  Estimate the data distribution. Find $\theta$ such that $p_\theta \approx p_{data} (x)$

<br>
<div class="grid grid-cols-[3fr_6fr] gap-10">
<div>
  <figure>
    <img src="/sklearn_example.png" style="width: 590px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://scikit-learn.org">https://scikit-learn.org</a>
    </figcaption>
  </figure>
</div>
<div>

* Standard technique: Maximum Likelihood Estimation<br>
$\max\limits_\theta \prod\limits_{i=1}^n p_\theta (x_i)$
* Statisticians love it
* Suffers from curse of dimensionality
</div>
</div>

---

# Why Generative Modeling?

* Unsupervised learning
  * We can learn meaningful latent features

<br>

<figure>
  <img src="/bigan.png" style="width: 590px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
    <a href="https://arxiv.org/pdf/1605.09782.pdf">Donahue et al. 2017</a>
  </figcaption>
</figure>

---

# Why Generative Modeling?

* Super-resolution and image renovation

<br>

<figure>
  <img src="/super-resolution.png" style="width: 790px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
    <a href="https://arxiv.org/pdf/2005.05005.pdf">Yang et al. 2020</a>
  </figcaption>
</figure>

---

# Standard Technique Using Explicit Density

* From Bayes' rule:
$$p_{model} (x) = p_{model} (x_1) \prod_{i=2}^d p_{model} (x_i \lvert x_1, ..., x_{i-1})$$

<br>

* Problem:
  * $\mathcal{O}(d)$ generation cost
    * Slow
  * Not clear what the order we want to set for the variables
  * No **latent space**
    * No features extractions

---

# Implicit distribution

* **Idea**: Transform an easy to sample distribution into something else:

$$x \sim p_\theta \iff x = g_\theta (z), z \sim p_z$$

* $p_\theta$ is challenging to compute but easy to sample from
* $p_z$ is usually Gaussian

---

# What is the Latent Space?

<figure>
  <img src="/Platos-Allegory-of-the-Cave-Featured-Image-1.jpg" style="width: 700px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
    <a href="https://ourpolitics.net/the-allegory-of-the-cave-textual-analysis/">https://ourpolitics.net/the-allegory-of-the-cave-textual-analysis/</a>
  </figcaption>
</figure>

---

# Arithmetic in the Latent Space

#### Initial idea from [Radford et al [2016]](https://arxiv.org/pdf/1511.06434/1000.pdf)

<br>

<figure>
  <img src="/latent_space.png" style="width: 600px !important;">
</figure>

---

# Arithmetic in the Latent Space

#### Idea: learn the “latent directions” of these features:
* Age, Eyeglasses, Gender, pose

<figure>
  <img src="/interface_gan.png" style="width: 800px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
    <a href="https://genforce.github.io/interfacegan/">https://genforce.github.io/interfacegan/</a>
  </figcaption>
</figure>