---
layout: post
title:  "To Be Greedy, or Not to Be - That Is the Question for Population Based Training Variants [TMLR]"
date:   2025-06-02 13:45:00 +0200
comments: true
---

**TL;DR** Bayesian PBTs optimize the greedy objective more effectively than non-Bayesian PBTs, this can be good or bad (depends on the task & hyperparams) \| [Paper](https://openreview.net/forum?id=3qmnxysNbi) \| [Code](https://github.com/AwesomeLemon/PBT-Zoo)

<!--more-->

Population Based Training ([PBT](https://arxiv.org/abs/1711.09846)) optimizes a hyperparameter schedule by evolving a population of solutions (weights + hyperparams). It is general, parallel, and scalable. It has been extended to leverage Bayesian Optimization ([PB2](https://arxiv.org/abs/2002.02518), [PB2-Mix](https://arxiv.org/abs/2106.15883), [BG-PBT](https://arxiv.org/abs/2207.09405)) or be less greedy ([FIRE-PBT](https://arxiv.org/abs/2109.13800))

{:refdef: style="text-align: center;"}
![Greedier Bayesian PBTs can hurt final performance](/pics/pbtzoo/greedy-slide.jpeg){: style="width: 80%;" }
{: refdef}

We find that from the theoretical perspective, Bayesian PBTs are guaranteed to asymptotically approach the returns of the greedy schedule (rather than the optimal one, as claimed in prior work).

Mechanistically, the number of hyperparameter update steps can influence the greediness, and the absolute & relative performance of PBT variants (despite constant total compute). The trends are clear for image classification where only the learning rate is optimized... 

{:refdef: style="text-align: center;"}
![Accuracy vs update steps on Fashion-MNIST and CIFAR-10](/pics/pbtzoo/steps-cv.png){: style="width: 80%;" }
{: refdef}

... but not so clear for reinforcement learning (or image classification with larger search spaces) 

{:refdef: style="text-align: center;"}
![RL performance vs update steps on Hopper and Humanoid](/pics/pbtzoo/steps-rl.png){: style="width: 80%;" }
{: refdef}

Our impartial evaluation showed that no PBT variant is substantially better than others across tasks and settings (note that one limitation of our work is not fully exploring all hyperparameters of PBT variants)

{:refdef: style="text-align: center;"}
![Variant ranking across steps, search space, and population size](/pics/pbtzoo/rank-grid.jpeg){: style="width: 80%;" }
{: refdef}

Check out the [paper](https://openreview.net/forum?id=3qmnxysNbi) for details!


We also release our [code](https://github.com/AwesomeLemon/PBT-Zoo) containing task-agnostic implementations of five PBT variants, hopefully making future research and comparison of PBT variants easier!

P.S. Thank you to my supervisors and coauthors, [Tanja Alderliesten](https://scholar.google.com/citations?user=K5wZcYEAAAAJ) and [Peter Bosman](https://scholar.google.com/citations?user=YED2pAoAAAAJ).

{% include comments.html %}
