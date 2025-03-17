# Path integral measures of prior probability

Suppose that a learned neural network is a maximum a posteriori solution to $p(\theta|D) = p(D|\theta)p(\theta)/Z$.

Let $L_D(\theta) = -log p(D|\theta)$ and $L_0(\theta) = -log p(\theta)$. $L_D$ is just the regular loss summed over the dataset, while $L_0$ is a hypothetical "loss with respect to the prior".

Then at $\theta^*$, the optimal parameters (not the loss minimizing ones!)

$$\grad_\theta L_D(\theta) = - \grad_\theta L_0(\theta)$$

Consider a set of networks $\theta_\alpha^*$ such that $$\theta_\alpha^* = \argmin_\theta \left[\alpha L_D(\theta) + L_0(\theta)\right]$$. At $\alpha=0$, this is the prior mode (if we're training from scratch $\theta_0^*=0$, but I'm more interested in cases where $\theta_0^*$ is a pretrained network) and at $\alpha=1$ this is just $\theta^*$ above. We can measure the density ratio between the prior mode and the solution $\theta^*$ with

$$L_0(\theta^*) - L_0(\theta_0^*) = \int_0^1 L_0(\theta)d\phi(\alpha)$$
$$=\int_0^1 \alpha L_D(\theta) d\phi(\alpha)$$

where $\phi(\alpha):=\theta_\alpha^*$ is the optimal parameter for each $\alpha$.

Two concerns:
 1. We typically don't use networks at convergence, so the equality condition on gradients is doubtful (unless we consider some prior that is implicitly partly determined by early stopping, though it's not obvious to me how legitimate that is)
 2. It's probably difficult to directly estimate the path integral

Thanks to Nora, here's an approximate approach to path integration (Garipov et. al.: https://arxiv.org/pdf/1802.10026):

 - Take a pretrained network $\theta_0^*$ and finetune it to produce $\theta_1^*$
 - Define a surrogate path $u_w(\alpha)$ and tune the parameters $w$ (by gradient descent) to approximate $\phi(\alpha)$ for every $\alpha$ (see Eq. (2) in the paper linked above)
 - We then sample $t$ uniformly on the curve $u_w(\alpha)$ and compute $\mathbb{E}_{t\sim U[u_w(\alpha)]}\left[\alpha \grad_\theta L_D \right]$, which is an approximation of the density ratio we're after

One of my projects is implementing this, I want to use a polygonal chain for $u_w$ (see Garipov for definition).