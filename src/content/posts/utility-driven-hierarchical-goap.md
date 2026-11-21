---
title: "Utility-Driven Hierarchical GOAP with Model-Estimated Effects"
date: "November 19th, 2026"
tags: ["LLM", "Research Proposal"]
teaser: "Can a symbolic planner pursue open-ended goals using an LLM only as a calibrated estimator?"
---
DISCLAIMER: This is a research proposal. It describes a method I intend to investigate, not finished work: nothing here has been implemented or evaluated yet, and the design may change as experiments come through.

## 1. Motivation

We want an agent that pursues an abstract, open-ended objective (e.g. "maximize the profit of this company") in an explainable fashion, starting from common knowledge only, in a world that is stochastic, partially observable and slow to respond.

Two existing families each solve half of this problem:

- **Symbolic planners** (STRIPS, GOAP, HTN) produce explicit, auditable plans and are cheap at decision time. They assume a hand-written world model: known actions, deterministic effects, boolean goals. A business has none of these.
- **LLM agents** (ReAct-style loops) bring broad world knowledge and can propose actions and judge outcomes no one hand-coded. Their plans are implicit, inconsistent across calls, hard to audit, and they plan poorly over long horizons [Valmeekam et al. 2023; Kambhampati et al. 2024].

**The key specificity.** Every utility function and every action impact has one of two sources: it is **hard-coded**, by a human or by the agent itself, or it is **predicted** at runtime by a decision model. The agent can also **invent** new actions, landmarks and goals as the environment evolves, so the system is not limited to what its designer anticipated. Everything else in this document is the mathematics that makes planning over such mixed, partly guessed quantities reliable.

The planner remains the decision-maker. Every model-produced number is treated as a prior to be calibrated against observed outcomes, never as ground truth. Explainability follows from this structure: any decision can be traced to a goal's priority score, a path's cost, and the estimates behind them, each with its source and calibration record.

Four failure modes drive the design:

1. **Optimizer exploitation.** A search that maximizes over model-estimated effects selects the actions the model overestimates (the optimizer's curse [Smith & Winkler 2006]).
2. **Neglect of critical goals.** A weighted sum of utilities trades away survival for gains elsewhere.
3. **Objective drift.** A system allowed to create goals or reweight priorities finds it easier to change the measure than the world [Manheim & Garrabrant 2018; Skalse et al. 2022].
4. **Thrashing.** Reactive goal selection switches goals before delayed effects pay off [Kinny & Georgeff 1991].

## 2. Method overview

The agent runs a receding-horizon loop: select a goal, plan a satisficing path for it, execute the first action, observe, update beliefs, repeat. Preference lives only in the shape of the utility functions; learning lives only in the calibration of estimates.

**State.** The agent maintains:

- a set of **root goals** (one main goal plus survival and health goals, designer-defined or invented by the agent), each with a utility function $U_i \in [0, 1]$ where 0 means failure of the whole system;
- a **roadmap**: landmarks (bounded, measurable subgoals) derived from the main goal;
- **uncertainty goals**: one per belief whose uncertainty could change a decision;
- an **action library** with schemas, model-estimated effects and calibration statistics;
- a **belief state**: current estimates (mean, variance) of every utility and action effect.

**Loop.** At every decision point:

1. **Observe and update.** Ingest new observations and events. Update calibration of executed actions. Update the CUSUM surprise statistic.
2. **Roadmap check.** If accumulated surprise exceeds its threshold, regenerate the roadmap and review invented goals (§6).
3. **Goal selection.** On a fixed interval or on action completion, compute each root goal's priority score (log utility, corrected by opportunity and switching cost) and select the lowest (§5). If the main goal is selected, repeat the selection among its landmarks.
4. **Plan.** Run satisficing A\* from the current state until the selected goal passes its threshold, scoring paths by risk-adjusted log-utility loss on all other goals and pruning paths with excessive ruin probability (§7).
5. **Act.** Execute the first action only. Record its predicted effects for later calibration.

Goal selection decides *what to pursue*. The planner decides *how*, under one shared objective: the risk-adjusted sum of log utilities. The language model never chooses actions directly.

## 3. Utilities and aggregation

Each root goal has a utility $U_i \in [0, 1]$. The system state is scored by the product of utilities, equivalently the sum of their logs:

$$
J = \log \prod_i U_i = \sum_i \log U_i
$$

**Zero means death.** A utility reaches 0 only when that goal's failure ends the system (cash runs out, legal shutdown). Any state with some $U_i = 0$ has $J = -\infty$ and is never selected. This is a log-barrier [Boyd & Vandenberghe 2004, ch. 11]: constraints are encoded as utilities that diverge at the boundary, so no separate constraint layer is needed.

**Shape encodes preference.** Utility functions are designed so their curvature reflects how much the goal matters at each level. Runway is concave in months: three months is near 0, five versus ten years differs little. Goals whose failure is not fatal have a **floor** (e.g. profit utility never drops below 0.5), so they can shape decisions but never dominate them. Unbounded objectives use saturating functions.

**Criticality is automatic.** The marginal value of improving goal $i$ is

$$
\frac{\partial J}{\partial U_i} = \frac{1}{U_i}
$$

so the lowest utility has the steepest gradient. Raising one goal from 0.1 to 0.2 is worth as much as raising another from 0.5 to 1.0. Goal selection by lowest utility (§5) and path scoring by J (§7) therefore agree.

**No goal weights.** Earlier iterations scaled each utility by a weight adjusted by a decision model. Weights were removed: they made the model the author of strategy, and a consistent bias in its reweighting is indistinguishable from strategy and cannot be audited. Preference is fixed in utility shapes by the designer; bias correction happens on effect estimates (§9).

**Alternatives considered.**

- *Weighted sum of utilities* (classic utility AI [Mark 2009]): trades survival for gains elsewhere; a high enough profit weight accepts bankruptcy.
- *Separate hard-constraint layer, lexicographic ordering*: works, but discontinuous; the barrier gives the same guarantee with graded urgency near the boundary.
- *Product of utilities* is the Nash product [Nash 1950] and log utility is Bernoulli's [Bernoulli 1738]; maximizing expected log wealth is also the Kelly criterion, known to avoid ruin [Kelly 1956].

## 4. Uncertainty and risk-adjusted evaluation

Every estimated utility and effect is a distribution with mean $\mu$ and variance $\sigma^2$. The planner scores each term by its risk-adjusted log utility:

$$
\mathbb{E}[\log U] \approx \log \mu - \frac{\sigma^2}{2\mu^2}
$$

**Derivation.** log is concave, so by Jensen's inequality $\mathbb{E}[\log U] \le \log \mathbb{E}[U]$. A second-order Taylor expansion around $\mu$ (the delta method) gives the size of the gap: the linear term vanishes in expectation, the quadratic term contributes $-\sigma^2/(2\mu^2)$. Because E is linear, the sum over goals holds regardless of correlation between goals.

**Why this form.** The penalty is half the squared coefficient of variation. It scales with $1/\mu^2$, so the same uncertainty costs little on a healthy goal and a lot on a critical one. Risk aversion emerges from the objective rather than from a tuned parameter.

**Ruin pruning.** Under uncertainty, any path with nonzero probability of some $U_i = 0$ has expected log utility $-\infty$, which would discard every realistic path. Instead, paths are pruned when the estimated ruin probability exceeds a tolerance $\varepsilon$, and expected log utility is maximized over the rest (a chance constraint [Charnes & Cooper 1959]). Ruin is checked at every intermediate state, not only at the end of the path. Correlations matter here: goals that fail together raise the joint ruin probability.

**Sources of $\sigma$.** In order of reliability: spread of ensemble predictions [Lakshminarayanan et al. 2017]; spread of the decision model's output distribution on constrained outputs; the model's self-reported confidence, which is the least calibrated [Kadavath et al. 2022]. Only epistemic uncertainty is modeled; irreducible world noise is not part of the planning estimate (it re-enters in calibration, §9).

**Alternative considered.** Full distributional evaluation by Monte Carlo rollouts. More exact, far more expensive per edge, and dependent on a simulator of outcomes that we generally do not have.

## 5. Goal selection

The agent pursues the root goal with the lowest priority score, where the score corrects log utility for opportunity and switching cost:

$$
s_i = \log U_i - O_i + C_i \;=\; \log\!\left( U_i \, e^{-O_i} \, e^{C_i} \right), \qquad i^* = \arg\min_i s_i
$$

Both corrections act multiplicatively inside the log. A goal at $U = 0$ has $s = -\infty$, so no finite opportunity or switching cost can ever override it, and the opportunity needed to outrank a near-zero goal grows without bound. No separate criticality threshold is needed.

**Opportunity $O_i$.** An opportunity is a gain that disappears if deferred. It is defined as the value of acting now minus the value of acting after a delay $\Delta$, both in log-utility units:

$$
O_i = \Delta \log U_i^{\text{now}} - \Delta \log U_i^{\text{after } \Delta}
$$

Both terms are computed by running the planner speculatively for goal $i$ alone, so $O_i$ is evaluated per goal, independently of the others. It carries the same bias as action-effect estimates, not a strategic bias. The decision model estimates only the window length $\Delta$, which is calibrated like any other estimate (§9). Opportunities are detected from events (a sensor threshold, an incoming email) or surfaced by information-gathering actions.

**Switching cost $C_i$.** For every goal other than the current one, $C_i$ is the forward-looking value lost by abandoning the current plan: expected effects of in-flight actions that will not materialize, plus setup costs of the new plan. Sunk effort is excluded. For the current goal, $C_i = 0$.

**Scope.** $O_i$ and $C_i$ enter goal selection only, never path costs: they would distort the planner's objective in ways that do not reflect the world. When a goal is selected because of an opportunity, the action that created it is passed to the planner as a seed candidate, so selection and planning stay consistent.

**Cadence.** Selection is re-evaluated on a fixed interval and on action completion, not every tick.

**Hierarchy.** If the main goal is selected, the same rule is applied among its landmarks (§6).

**Alternatives considered.**

- *Pure lowest-utility selection* (as in priority-driven GOAP [Orkin 2006]): oscillates between near-equal goals and abandons work before delayed effects arrive.
- *Fixed hysteresis margin $\delta$*: simple, but $\delta$ is arbitrary and blind to how much in-flight work would be lost.
- *Additive opportunity bonus outside the log*: can override a critical goal; rejected.
- The question of when to reconsider commitments is the intention-reconsideration problem of BDI agents [Bratman 1987; Kinny & Georgeff 1991; Rao & Georgeff 1995]; switching cost is a quantitative answer to it.

## 6. Roadmap and invented goals

The main goal is usually unbounded and its effects are delayed by months, so it gives the planner almost no signal. The roadmap converts it into landmarks: bounded, measurable subgoals with a threshold and a horizon (e.g. "monthly churn below 3% by Q3"). Landmarks provide near-term reward for the delayed goal.

**Landmarks and invented goals share one mechanism.** Both are definitions the agent writes for itself: a measured metric, a utility function over it, a floor. Landmarks serve the main goal; invented goals (§6.1) capture new concerns the environment raises. Both are a lever for objective drift, since a new goal whose utility starts near zero would capture the planner through the $1/U_i$ gradient. Landmarks carry a floor $\varphi$; invented goals are guarded by a judge agent and human review (§6.1).

**How landmarks enter the system.**

- In root-level goal selection, the main goal's utility is estimated *in light of* its landmarks (progress on landmarks informs the estimate), but landmark utilities themselves do not compete with root goals.
- In landmark selection (when the main goal is chosen), the rule of §5 is applied among landmarks.
- In path evaluation, landmark utilities enter the objective alongside all root goals, including the main goal, with a high floor $\varphi$ (a method hyperparameter, e.g. 0.7). The floor lets landmarks shape paths but never dominate them or trigger ruin pruning.

**Why keep both main goal and landmarks.** Landmarks act as reward shaping. Shaping preserves the optimal policy when it is potential-based [Ng, Harada & Russell 1999], but a hand- or model-written roadmap is not guaranteed to be. Keeping the main goal's own utility in the objective is the anchor: if the roadmap is poorly designed, the planner does not optimize artificial rewards at the main goal's expense. The cost is some double counting, bounded by $\varphi$.

**Divergence diagnostic.** Landmarks imply a forecast of main-goal progress. Persistent divergence between that forecast and the main goal's measured utility means the roadmap rewards the wrong things, and feeds the surprise statistic below.

**Replanning trigger: CUSUM.** A full roadmap rewrite is expensive and causes churn, so it is gated on accumulated surprise. For each prediction the normalized error is $z_t = (\text{observed} - \text{predicted}) / \sigma_t$. A one-sided CUSUM [Page 1954] accumulates it:

$$
S_t = \max\left(0,\; S_{t-1} + |z_t| - k\right), \qquad \text{rewrite when } S_t > h
$$

The allowance $k$ absorbs ordinary noise; the threshold $h$ sets sensitivity. Isolated misses decay; systematic divergence (a wrong world model) triggers a rewrite.

**Alternatives considered.**

- *Fixed-interval rewrites*: either too slow to react or too frequent to let landmarks complete.
- *Rewrite every step*: maximal responsiveness, maximal thrashing.
- *Landmarks only, main goal removed*: no double counting, but no protection against a bad roadmap.

### 6.1 Invented goals

The agent may add root goals the designer did not anticipate, at whatever criticality the situation warrants, including fatal. A new regulation that could close the company must be represented by a goal whose utility can reach zero, from the moment it is recognized. Detecting and adopting such goals is what makes the system agile rather than bounded by its initial goal set. Because invented goals carry full authority, the safeguards against reward hacking are a judge agent and a human, not a floor.

**Triggers.** A proposal is made when surprise accumulates without being explained by any existing goal (§6 CUSUM), when an event names a risk or factor outside the current goal set, or when an information action surfaces one.

**Definition.** The decision model writes the goal like an action record: the measured metric; the utility function over it, including whether it can reach zero; a criticality claim stating what failure would cost and the evidence for it; the root goals it affects and its predicted contribution to them; and a falsifier. A goal that cannot name a measurement is not admitted.

**Urgency versus lasting criticality.** The two are handled separately. A deadline (comply within 90 days) is transient pressure and is carried by the opportunity term of goal selection (§5). The lasting cost of staying compliant is carried by the goal's utility function, which remains in the objective after the deadline passes.

**Safeguard 1: judge agent.** Every proposed goal, every change to an invented goal's utility function, and periodically every existing invented goal and landmark is reviewed by a separate judge agent. The judge runs in its own context with its own instructions, ideally on a different model, and has no stake in the planner's scores. It checks whether:

- the criticality claim is supported by the stated evidence;
- the goal could be satisfied without any change in the world (an easily gamed metric);
- the metric can be moved by the agent's own reporting rather than by outcomes;
- the goal duplicates an existing goal, which would double-count it;
- the goal diverts attention from a deteriorating designer-defined goal.

Verdicts are accept, accept with modifications (e.g. a less steep utility), reject, or escalate to a human. This follows the LLM-as-judge pattern [Zheng et al. 2023] and the idea of using one AI system to scrutinize another's claims [Irving, Christiano & Amodei 2018]. A judge built on the same model as the proposer can share its blind spots, which is why human review remains necessary.

**Safeguard 2: human in the loop.** The human holds final authority over the objective and is a structural part of the method, not a fallback. Human confirmation is required for:

- any invented goal whose utility can reach zero (a fatal goal);
- any proposal the judge escalates;
- any action above a cost or irreversibility threshold whose justification rests mainly on an unconfirmed invented goal.

Critical goals cannot wait for review, so a proposed goal becomes active immediately as **provisional**: it takes part in goal selection and path evaluation at its proposed criticality, but actions over the cost or irreversibility threshold that it motivates are held until a human confirms it. The agent can start preparing for a regulation the day it learns of it, without committing large or irreversible resources on a goal no one has validated. Each proposal reaches the human with the trace that explainability provides (§1): the triggering evidence, the judge's verdict, and the paths the goal changed. Human decisions are logged and used to measure the judge's agreement rate, so its verdicts can be weighted by their track record.

**Safeguard 3: calibration.** Contribution claims are calibrated by the same shrinkage as action effects (§9). A goal whose claimed consequences fail to materialize is flagged to the judge for downgrade or retirement, never changed silently. Retirement of a confirmed goal requires human confirmation. Designer-defined goals are never altered or retired by the agent.

**Alternatives considered.**

- *Fixed goal set*: safe, but blind to anything the designer did not anticipate.
- *Unrestricted goal creation* (as in game NPCs that invent goals once current ones are met): maximally adaptive, but gives the agent a direct way to redefine success.
- *Mandatory floor on invented goals, criticality earned through calibration*: an earlier draft. Safe, but it would treat a genuinely fatal new risk such as a regulation as minor for as long as evidence takes to accumulate.
- *Human approval before any new goal takes effect*: safe, but too slow for urgent risks; provisional activation with gated actions keeps the agent responsive.

## 7. Planning: satisficing A\*

Planning is a shortest-path search [Hart, Nilsson & Raphael 1968] that stops as soon as the selected goal or landmark passes its threshold. Because every path is scored by the same objective, the selected goal does not change the scoring: it defines the **termination test** and the **relevant action set** (actions that can affect it), which prunes the search.

**Goal test.** The predicted state reaches the target threshold, accounting for each action's effect lag (an action whose effect lands after the horizon does not count toward it).

**Edge cost.** With $R_j$ the risk-adjusted log utility of goal $j$ (§4), the cost of action $a$ is its resource cost plus the damage it does to every other goal and landmark:

$$
c(a) = \kappa \cdot \text{cost}(a) + \sum_{j \neq \text{target}} \max\!\left(0,\, -\Delta R_j(a)\right)
$$

A\* requires non-negative edge costs, so improvements to other goals are clamped out of the cost and used only as a tie-breaker between equal-cost paths. $\kappa$ converts resource cost to log-utility units and is the only scaling constant in the planner.

**Heuristic.** The remaining gap in the target's risk-adjusted log utility divided by the best calibrated improvement-per-cost among relevant actions. This is optimistic, keeping A\* close to admissible.

**Pruning.** Partial paths whose ruin probability exceeds $\varepsilon$ at any intermediate state are discarded (§4).

**Execution.** Only the first action of the plan is executed; the agent then observes and replans. This receding-horizon scheme is the logic of model predictive control [Mayne et al. 2000]. It recovers most of the closed-loop behavior a contingent planner would provide: after an observation, the next plan already conditions on it.

**No path found.** If no path passes the threshold within the search budget, the best partial path (largest progress toward the threshold) is returned and the gap is logged as a surprise signal.

**Alternatives considered.**

- *Monte Carlo tree search* [Kocsis & Szepesvári 2006; Browne et al. 2012], used with LLM priors in RAP [Hao et al. 2023]: handles stochastic branching and contingent plans, but requires an outcome simulator we lack, searches to a fixed depth instead of stopping when the goal is met, and costs far more model calls per decision.
- *Maximize the target goal instead of satisficing*: invites the planner to exploit the most overestimated actions (§1); a threshold bounds how hard the search pushes on any single estimate.

## 8. Information gathering as uncertainty goals

Information gathering is pursued through dedicated uncertainty goals, planned with the same satisficing search. Their utility is computed from current uncertainty only; the effect of information actions is a coarse belief, not a simulated outcome.

**Why not estimate information gain directly.** Value of information theory [Howard 1966] requires a model of what an observation might return (a survey's answers, a search's results, an analyst's report). In the general case no such model exists. What can be measured reliably is how uncertain the agent is *now*.

**Utility of an uncertainty goal.** Uncertainty only matters if it could change a decision. For each belief $k$, draw samples from its current distribution (ensemble members or output distribution), holding other beliefs at their means, and compute the expected regret of the currently preferred choice among the top candidate paths:

$$
\rho_k = \mathbb{E}_{\text{samples}}\left[ \max_p V_p - V_{\hat p} \right], \qquad U_k^{\text{unc}} = \varphi_u + (1 - \varphi_u)\, e^{-\rho_k / \tau}
$$

where $V_p$ is a path's value, $\hat p$ the path preferred under mean estimates, $\tau$ a scale, and $\varphi_u$ a floor so uncertainty is never fatal. If the top candidates are clearly separated, $\rho_k \approx 0$ and the goal is satisfied without any research. This needs only the current samples, no simulation of the information source. It is closely related to the probability of incorrect selection in ranking-and-selection and to the knowledge-gradient policy [Frazier, Powell & Dayanik 2008].

**Effect of information actions.** The decision model gives each information action a coarse believed effect on the targeted belief (an ordinal level mapped to a fractional reduction of $\sigma_k$). After execution the actual reduction is measured from the updated ensemble spread and the belief is calibrated like any action effect (§9). The search stops as soon as $\rho_k$ passes its threshold: the agent needs only enough belief that knowledge will improve, not an exact estimate.

**Credit only where earned.** Within a plan, the believed $\sigma$ reduction is credited only to the uncertainty goal, never fed into the risk-adjusted utilities of business goals. Otherwise the planner would count the variance penalty of §4 as recovered value. That gain is illusory: by the law of iterated expectations, learning without acting on it leaves expected utility unchanged; information pays off only through later decisions, which replanning (§7) captures.

**Alternatives considered.**

- *Bayesian posterior-variance formulas* (conjugate updates from sample size and noise): require knowing what the source will measure and how noisily; rarely available.
- *Uncertainty as a generic penalty on all beliefs*: drives research on beliefs that affect no decision.
- *Contingent planning over observation outcomes*: exact, but needs an observation model and a branching search (§7).

## 9. Action library, generation and calibration

Actions are structured records whose effects start as model-estimated priors and are corrected by Bayesian shrinkage as outcomes arrive. This is the only learned correction in the system.

**Schema.** Every action, hand-written or generated, must specify:

| Field         | Content                                                                             |
| ------------- | ----------------------------------------------------------------------------------- |
| Preconditions | State required to execute                                                           |
| Effects       | Per affected goal: mean and $\sigma$ of the change, hard-coded or model-estimated |
| Effect lag    | Delay before each effect lands, and its window                                      |
| Cost          | Resources consumed (money, time, headcount)                                         |
| Measurement   | Which observed metric will show the effect                                          |
| Falsifier     | Observation that would show the action failed                                       |

An action that cannot name a measurement and a falsifier is not admitted. This filters vague generated actions ("improve customer experience") whose effects could never be calibrated.

**Generation.** The decision model proposes new actions when the planner finds no path or when a path's best option is weak. Proposals are deduplicated against the library. Actions whose realized effects repeatedly fall short of prediction lose credibility through calibration and are eventually pruned. This mirrors the growing skill library of Voyager [Wang et al. 2023], with explicit effect models added.

**Calibration by shrinkage.** Each executed action yields a pair (predicted effect, observed effect). For an action type $a$ with $n_a$ observations and mean residual $\bar r_a$, the estimated bias is shrunk toward a pooled bias $b_0$ shared by all action types:

$$
\hat b_a = \frac{ \frac{n_a}{\sigma_{\text{obs}}^2}\, \bar r_a + \frac{1}{\sigma_b^2}\, b_0 }{ \frac{n_a}{\sigma_{\text{obs}}^2} + \frac{1}{\sigma_b^2} }
$$

Future estimates for type $a$ are corrected by $\hat b_a$ before entering the search. With few observations, the correction stays near the pooled bias; it moves to the action's own record only as evidence outweighs observation noise $\sigma_{\text{obs}}$. This is where world noise matters: without shrinkage, a good action that got unlucky would be marked as overestimated. The same scheme calibrates opportunity windows (§5), information-action effects (§8) and landmark-to-main-goal contributions (§6). See partial pooling in [Efron & Morris 1975; Gelman et al. 2013].

**Attribution with delayed effects.** Observed metric changes are attributed to the actions whose declared effect window covers them; when several overlap, by regression of the metric change on the active actions.

**Where reinforcement learning fits.** Shrinkage corrects action-level estimates, which come in large numbers. Learning better utility predictors (a value function over states [Sutton & Barto 2018]) needs many full trajectories, which one real company never provides. That learning belongs in simulation.

**Alternatives considered.**

- *Learned goal weights to absorb bias*: one trajectory per company is too little data, and weights mix strategy with estimation error.
- *Momentum or Adam on weight changes*: filters noise but preserves systematic bias (Adam normalizes step size, so a small consistent bias moves at full speed).
- *Trust raw model estimates*: maximally exposed to the optimizer's curse.

## 10. Role of the decision model

The language model is an estimator and a generator, never a decision-maker. Every quantity it produces is either calibrated against outcomes or bounded by a floor.

| Task                      | Output                                                     | Safeguard                                                                                                 |
| ------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Utility estimation        | Value of a utility when not directly measurable            | Ensemble spread → $\sigma$; risk adjustment (§4)                                                      |
| Action effect estimation  | Mean and $\sigma$ of effects, lags                       | Shrinkage calibration (§9)                                                                               |
| Action generation         | New action records                                         | Schema with measurement and falsifier; pruning on poor calibration                                        |
| Roadmap generation        | Landmarks with thresholds and horizons                     | Floor $\varphi$; main goal kept as anchor; CUSUM-gated rewrites (§6)                                   |
| Opportunity window        | $\Delta$ for a detected opportunity                      | Calibrated like effects; enters selection only (§5)                                                      |
| Information action effect | Ordinal $\sigma$-reduction level                         | Measured after execution and calibrated (§8)                                                             |
| Goal invention            | New root goals with utility function and criticality claim | Judge agent review; human confirmation for fatal goals; provisional activation with gated actions (§6.1) |

It does not choose goals, choose actions, set goal weights or alter designer-defined goals. Goals it invents are reviewed by a judge agent and, when fatal, confirmed by a human (§6.1). Goal choice is the argmin of §5; action choice is the search of §7; preferences are fixed in utility shapes.

## 11. Open questions and known limitations

- **Utility design is unassisted.** All preference lives in utility shapes and floors, written by the designer. There is no method yet for validating them.
- **Unmeasurable utilities.** Shrinkage calibrates action effects against observed metrics. Utilities that are themselves model-estimated (e.g. employee morale) have no ground truth to calibrate against.
- **Hyperparameters.** Ruin tolerance $\varepsilon$, landmark floor $\varphi$, uncertainty floor $\varphi_u$ and scale $\tau$, cost scale $\kappa$, CUSUM $k$ and $h$, selection interval. Cost and irreversibility thresholds for human confirmation. Sensitivity is unknown.
- **Per-belief regret ignores interactions.** Varying one belief while holding others at their means misses decisions that depend on two uncertain beliefs jointly.
- **Ruin probability estimates** depend on model-produced $\sigma$ and on correlations between goals that are rarely estimated.
- **Search complexity.** Effect lags make states time-indexed; generated actions grow the branching factor. Bounds on library size and search budget are needed.
- **Estimate consistency.** The same query to the model can return different numbers. Caching estimates per (action, state features) is likely required for stable planning.
- **No within-plan contingency.** Plans cannot branch on future observations; replanning compensates, but plans that only pay off as contingent strategies are invisible to the search.
- **Evaluation.** The method needs a business simulator with hidden dynamics, and a comparison against a plain LLM agent given the same goals and tools, measuring outcomes, calibration over time and decision cost.
- **Judge independence and human load.** A judge on the same model as the proposer may share its blind spots. Escalation thresholds must keep human review rare enough to be done carefully but frequent enough to catch drift.

## 12. Related work and References

Disclaimer the reference section is a first pass by Claude. I still need to run through it and double check it.

The method combines a symbolic planner, utility-based arbitration and LLM-supplied estimates. Closest relatives, and where this design departs from each:

| Work                                                                                          | Shared idea                                                       | Difference                                                                                                                            |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| STRIPS [Fikes & Nilsson 1971], GOAP [Orkin 2006]                                              | Actions with preconditions and effects; A\* over action sequences | Effects are stochastic, delayed, model-estimated and calibrated; goals are continuous utilities, not boolean states                   |
| Utility AI [Mark 2009]                                                                        | Utility curves drive arbitration                                  | Log-barrier product instead of weighted scores; utilities also score plans, not only choose behaviors                                 |
| HTN planning [Erol, Hendler & Nau 1994; Nau et al. 2003]                                      | Abstract goals decomposed into subgoals                           | Decomposition is model-generated and revisable on surprise; landmarks are measurable thresholds, not task methods                     |
| Landmark planning [Hoffmann, Porteous & Sebastia 2004; Richter & Westphal 2010]               | Intermediate states guide search                                  | Landmarks are proposed by a model, not derived logically, so the main goal is kept as an anchor                                       |
| BDI agents [Bratman 1987; Rao & Georgeff 1995]                                                | Commitment to intentions; when to reconsider                      | Reconsideration is quantified by switching cost and opportunity inside the log priority                                               |
| Options, feudal RL [Sutton, Precup & Singh 1999; Dayan & Hinton 1993; Vezhnevets et al. 2017] | Temporal abstraction; managers set subgoals                       | No policy is learned from reward; planning over calibrated effect models                                                              |
| Model predictive control [Mayne et al. 2000]                                                  | Plan, execute first step, replan                                  | Discrete actions, symbolic search, model-estimated dynamics                                                                           |
| ReAct [Yao et al. 2023a]                                                                      | LLM in an act–observe loop                                       | The LLM never picks actions; it supplies estimates to a planner                                                                       |
| LLM+P [Liu et al. 2023], LLM-Modulo [Kambhampati et al. 2024]                                 | LLM paired with a sound planner or verifiers                      | Goes beyond translating to PDDL: the LLM supplies quantitative, uncertain effect estimates that are calibrated over time              |
| RAP [Hao et al. 2023], Tree of Thoughts [Yao et al. 2023b]                                    | Search over LLM-predicted states                                  | Satisficing A\* with an explicit utility objective, ruin pruning and calibration, instead of MCTS or tree search over reasoning steps |
| Voyager [Wang et al. 2023]                                                                    | Growing library of generated skills                               | Actions carry effect models with falsifiers and are pruned by calibration                                                             |
| Generative Agents [Park et al. 2023]                                                          | LLM agents forming their own goals                                | Invented goals, fatal ones included, pass a judge agent and human confirmation, to prevent objective drift                            |

Disclaimer the reference section is a first pass by Claude. I still need to run through it and double check it.

- Bernoulli, D. (1738). Specimen theoriae novae de mensura sortis. English translation: Exposition of a new theory on the measurement of risk, *Econometrica* 22(1), 1954.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Bratman, M. (1987). *Intention, Plans, and Practical Reason*. Harvard University Press.
- Browne, C. et al. (2012). A survey of Monte Carlo tree search methods. *IEEE Trans. Computational Intelligence and AI in Games* 4(1).
- Charnes, A. & Cooper, W. W. (1959). Chance-constrained programming. *Management Science* 6(1).
- Dayan, P. & Hinton, G. (1993). Feudal reinforcement learning. *NIPS 5*.
- Efron, B. & Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *JASA* 70.
- Erol, K., Hendler, J. & Nau, D. (1994). HTN planning: complexity and expressivity. *AAAI-94*.
- Fikes, R. & Nilsson, N. (1971). STRIPS: a new approach to the application of theorem proving to problem solving. *Artificial Intelligence* 2(3–4).
- Frazier, P., Powell, W. & Dayanik, S. (2008). A knowledge-gradient policy for sequential information collection. *SIAM J. Control and Optimization* 47(5). (\*)
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
- Hao, S. et al. (2023). Reasoning with language model is planning with world model. *EMNLP 2023*.
- Hart, P., Nilsson, N. & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Trans. Systems Science and Cybernetics* 4(2).
- Hoffmann, J., Porteous, J. & Sebastia, L. (2004). Ordered landmarks in planning. *JAIR* 22.
- Howard, R. (1966). Information value theory. *IEEE Trans. Systems Science and Cybernetics* 2(1).
- Irving, G., Christiano, P. & Amodei, D. (2018). AI safety via debate. arXiv:1805.00899.
- Kadavath, S. et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221. (\*)
- Kambhampati, S. et al. (2024). LLMs can't plan, but can help planning in LLM-Modulo frameworks. *ICML 2024*. (\*)
- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal* 35(4).
- Kinny, D. & Georgeff, M. (1991). Commitment and effectiveness of situated agents. *IJCAI-91*.
- Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo planning. *ECML 2006*.
- Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*.
- Liu, B. et al. (2023). LLM+P: empowering large language models with optimal planning proficiency. arXiv:2304.11477.
- Manheim, D. & Garrabrant, S. (2018). Categorizing variants of Goodhart's law. arXiv:1803.04585. (\*)
- Mark, D. (2009). *Behavioral Mathematics for Game AI*. Course Technology. (\*)
- Mayne, D. et al. (2000). Constrained model predictive control: stability and optimality. *Automatica* 36(6).
- Nash, J. (1950). The bargaining problem. *Econometrica* 18(2).
- Nau, D. et al. (2003). SHOP2: an HTN planning system. *JAIR* 20.
- Ng, A., Harada, D. & Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. *ICML 1999*.
- Orkin, J. (2006). Three states and a plan: the A.I. of F.E.A.R. *Game Developers Conference*. (\*)
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41(1–2).
- Park, J. S. et al. (2023). Generative agents: interactive simulacra of human behavior. *UIST 2023*.
- Rao, A. & Georgeff, M. (1995). BDI agents: from theory to practice. *ICMAS-95*.
- Richter, S. & Westphal, M. (2010). The LAMA planner: guiding cost-based anytime planning with landmarks. *JAIR* 39.
- Skalse, J. et al. (2022). Defining and characterizing reward hacking. *NeurIPS 2022*. (\*)
- Smith, J. & Winkler, R. (2006). The optimizer's curse: skepticism and postdecision surprise in decision analysis. *Management Science* 52(3).
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- Sutton, R., Precup, D. & Singh, S. (1999). Between MDPs and semi-MDPs: a framework for temporal abstraction in reinforcement learning. *Artificial Intelligence* 112.
- Valmeekam, K. et al. (2023). On the planning abilities of large language models: a critical investigation. *NeurIPS 2023*. (\*)
- Vezhnevets, A. et al. (2017). FeUdal networks for hierarchical reinforcement learning. *ICML 2017*.
- Wang, G. et al. (2023). Voyager: an open-ended embodied agent with large language models. arXiv:2305.16291.
- Yao, S. et al. (2023a). ReAct: synergizing reasoning and acting in language models. *ICLR 2023*.
- Yao, S. et al. (2023b). Tree of thoughts: deliberate problem solving with large language models. *NeurIPS 2023*.
- Zheng, L. et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks. (\*)
