---
title: "Consistent Rectified flow: flow straight without reflow."
date: "May, 2026"
tags: ["Diffusion", "WIP"]
teaser: "Taking inspiration from consistency models with rectified flow"
# paper: "Own Research"
---

[Rectified flow](https://arxiv.org/pdf/2209.03003) models learns to approximate velocity between a pair $(X_0, X_1)$ sampled in $\pi_0$ and $\pi_1$ respectively, with as straight of a flow as possible. The straightness property is very important because it allows, at inference time, to perform a single sampling step at time $t$ and directly deduce $X_1$, such that $X_1 \approx X_t + v_\theta(X_t, t) * (1 - t)$ for any $t \in [0, 1]$.

Now the optimization process does not directly solve for straight flows. It solves for the drift which minimizes the variance around the mean velocity $X_1 - X_0$ over the trajectory:

$$
   \min_{v} \int_0^1 \mathbb{E}\left[\left\|(X_1 - X_0) - v(X_t, t)\right\|^2\right] dt
$$

As such, the flow is not guaranteed to be straight. This is the point of the "Reflow" process which iteratively straighten the flow by increasing the correlation between the source distribution and the target distribution, and gurantees that
$$
    \min_{k\in{0,..., K}} \int_0^1E[||(Z_1 - Z_0) - \dot{Z_t}||^2]dt \leq \frac{E\left[||X_1 - X_0||^2\right]}{K}
$$

This however implies distilling the model several times, which comes with its own set of issues. Among others, error compounds, as the reflow process relies on the fact that the teacher model correctly learns to predict the drift, and thus $dZ_t = v_\theta(Z_t, t) dt$. In practice, any error here may be pushed onto the student. To this we need to add the significant cost of training an additional model, especially granted the scale of modern datasets.

Instead, we observe that when perfectly straight, $v(Z_t, t) = (Z_1 - Z_0) = const$. Taking inspiration from [Consistency models](https://arxiv.org/pdf/2303.01469), we add the following loss to the training scheme:

$$
    \mathcal L_{\text{const}} = ||v_\theta(Z_{t_1}, t_1) - v_\theta(Z_{t_2}, t_2)||_2
$$

In theory we could make do with only this loss, having $v$ be independant of $t$ which the above loss encourages would mean that the rectified flow ODE would simplify to $Z1 = Z_t + v(Z_1, Z_0)\cdot(1 - t)$. For completeness, we will measure the impact of minimizing both losses jointly against only minimizing the consistency loss.
