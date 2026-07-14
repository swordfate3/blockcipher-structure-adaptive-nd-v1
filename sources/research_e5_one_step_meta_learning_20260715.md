# E5 Literature Search: One-Step Meta-Learning

```text
date     = 2026-07-15
provider = Tavily via web-search-plus 2.8.6
depth    = advanced
query    = Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
           ICML 2017 Reptile arxiv 1803.02999 one gradient step
```

## Results

1. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**
   - URL: https://proceedings.mlr.press/v70/finn17a/finn17a.pdf
   - Tavily score: `0.840`
   - Search excerpt: explicitly optimizes loss after one or more target-task
     gradient steps; the one-step update is
     `theta' = theta - alpha * grad(L_task(theta))`.
2. **On First-Order Meta-Learning Algorithms**
   - URL: https://arxiv.org/abs/1803.02999
   - Search query also returned a secondary Reptile explainer:
     https://pub.aimind.so/reptile-a-first-order-model-agnostic-meta-learning-fo-maml-algorithm-serial-and-batched-version-4758f92bb02e
   - Search excerpt: Reptile/FOMAML approximate rapid-adaptation objectives
     without MAML's full second-order differentiation.
3. **Theoretical Convergence of Multi-Step Model-Agnostic Meta-Learning**
   - URL: https://jmlr2020.csail.mit.edu/papers/volume23/20-720/20-720.pdf
   - Tavily score: `0.776`
   - Search excerpt: analyzes multi-step MAML convergence and cites both MAML
     and Reptile as the core rapid-adaptation methods.
4. **Model-Agnostic Meta-Learning for Fast Adaptation (mirror)**
   - URL: https://www.crcv.ucf.edu/wp-content/uploads/2019/03/CAP6412_Spring2018_1703.03400.pdf
   - Tavily score: `0.765`
   - Search excerpt: mirror of the MAML paper with the same one-step objective.
5. **Exploring First Order Gradient Approximation Meta-Learning**
   - URL: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report145.pdf
   - Tavily score: `0.666`
   - Search excerpt: student report comparing first-order approximations; used
     only as a secondary pointer, not primary evidence.
6. **MetaLearn-MAML_Reptile implementation**
   - URL: https://github.com/hfahrudin/reptile_implement_tf2
   - Tavily score: `0.659`
   - Search excerpt: third-party implementation; not primary literature.

## E5 Use

MAML directly matches E4's exactly-one-epoch adaptation question. However,
Innovation 1 currently has only two comparable typed-SPN cipher tasks,
PRESENT and GIFT. Using one during meta-training and the other for evaluation
does not supply a credible distribution of meta-training tasks; using both in
meta-training leaks the target cipher. MAML/Reptile is therefore held until a
larger task family and a held-out target protocol exist.
