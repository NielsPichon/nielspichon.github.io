---
title: "Terminal Velocity Matching"
date: "May 14th, 2026"
tags: ["Diffusion", "Literature Review"]
teaser: "Single Stage training is finally here?"
---

DISCLAIMER: This is a literature review blog post. See the source here: [Arxiv Paper](https://arxiv.org/pdf/2511.19797)

With Diffusion and Flow matching models we have a clear trade-off: quality VS inference time. Many techniques have attempted to reduce the number of diffusion steps but they all involve some form of extra trianing cost and time. For instance, Rectified Flows imply distilling k time the model to straighten the flow, allowing for near constant velocity along the transport trajectory and thus single (or very few) step sampling. Similarly consistency models will imply computing the prediction at 2 close time steps, but most importantly we do not retain any objective related to the explicit target distribution, which can impact the final quality.
Terminal Velocity Matching offers to get the same sort of gains, that is single or few sampling steps at inference, for no extra training cost, and with a more explicit connection to the target distribution. The paper claims to do all of this while hitting SoTA results when it comes to single step generation FID score on ImageNet 256x256.

## Terminal Velocity Matching

If we take most flow matching approaches, the idea is to take a sample $X_t$ at a given timestep $t$ and to estimate the derivative of the trajectory, the drift, at that point. In other words, we are predicting the "initial" velocity at that timestep. Then we can use this drift to solve, using e.g. Euler's method, the displacement along the trajectory.

In opposition, the proposed method predicts the net displacement between 2 arbitrary timesteps, which they show is equivalent to predicting the velocity at the "terminal" point of the displacement, hence the name.

This is expressed by the fact that, as they show in the appendix, we can bound the displacement error as so:

$$
    \mathcal{L}_\text{disp}^t(\theta) \leq \int_0^t \mathbb{E}_{x_t} \left[ \left\| \frac{d}{ds} f_\theta(x_s, t, s) - u(x_s, s) \right\|^2_2 \right] ds
$$

where $f_\theta$ is the net displacement. In other words, they show that if we can minimize the drift prediction at the terminal point $x_s$, for all $x_s$ in $[0, t]$, we also minimize the error on the net displacement itself.

They further demonstrate that this formalism is fully equivalent to the flow matching objective when $t\rightarrow s$, that is in the 0 net displacement limit.

In practice, what their network actually predicts is $F_\theta$ such that

$$
f_\theta(x_t, t, s) = (s - t)F_\theta(x_t, t, s)
$$

In other words, $F_\theta$ is the dual time conditioned instantaneous velocity. That is, at the limit where $t=s$, $F_\theta(x_t, t, t) = u_\theta(x_t, t)$, which is an estimate of the drift at $(x_t, t)$.

The problem is that in the general case, for arbitrary non 0 displacments, the above displacement is not fully equivalent to the flow matching objective, only in the limit. To alleviate this issue, the authors suggest to use the following approximation:

$$
    u(x_s, s) \sim u_\theta(x_t + f_\theta(x_t,t,s), s)
$$

And they now suggest the full formulation of the loss:

$$
\mathcal{L}_{TVM} = \mathbb{E}_{x_t, x_s, v_s}\left[\left\| \frac{d}{ds} f_\theta(x_s, t, s) - u_\theta(x_t + f_\theta(x_t,t,s), s) \right\|^2_2 + \left\|u_\theta(x_s, s) - v_s\right|^2_2\right]
$$

The goal here is to ensure that the drift prediction is correct while also optimizing for the displacement objective derived earlier.

To clarify things a bit and right everything as a function of the model's prediction and the ground truth drift:

$$
    \mathcal{L}_{TVM} = \mathbb{E}_{x_t, x_s, v_s}\left[\left\| F_\theta(x_t, t, s) + (s - t)\frac{\partial}{\partial s}F_\theta(x_t, t, s) - F_\theta(x_t + (s - t)F_\theta(x_t, t, s), s, s) \right\|^2_2 + \left\|F_\theta(x_s, s, s) - v_s\right|^2_2\right]
$$

So there are 3 estimates to be done here: one at $(x_t, t, s)$, one at $(x_s, s, s)$, and one at the predicted position of $(\tilde{x_s}, s, s)$, where $\tilde{x_s}$ it the predicted position of $x_s$. Ultimately, they end up using stop_gradient weights for the estimation of $\tilde{x_s}$ and the EMA stop gradient weights for estimating the velocity at that point. So that is a total of 4 forward passes per loss computation.


But wait! There is more! From this point, the authors show that under the assumption of $u_\theta$ Lipschitz-continuous, we can derive an upper bound for the Wasserstein distance between the distribution resulting from applying $f_\theta$ to $p_t$ and $p_0$. This is to say that, up to a constant, minimizing the TVM loss should allow use to approximate the transport from $p_t$ to $p_0$. But the catch is that trasnformer architectures with layer norm are not Lipschitz-continuous. Yet, using the typical AdaLN form e.g. DiT, but using RMSNorm rather than Layer norm, the problem should be closer to Lipschitz continuous, and empirically this seems to be the case.

Lastly, as an extra bonus contribution, the authors discuss the implementation of a dedicated kernel for computing $\frac{\partial}{\partial_s} F_\theta(x_t, t, s)$. I invite you to refer to the paper for more details on the topic, as I won't cover it here.

## Discussion
The results are pretty clear. Their method seems to outperform all referenced diffusion models in only 4 steps, and establishes a new SoTA FID for single step generation. Visually, the examples provided in the paper are stunning.

Now practically, the derived loss involves 4 forward passes plus the Jacobian-Vector product for computing $\frac{d}{ds}F_\theta$ which does add a lot of memory and compute cost, including with their custom kernel implementation. So this is by all means a very costly method. This is really a classic training time VS inference time trade-off.
