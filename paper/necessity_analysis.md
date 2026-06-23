# Necessity Analysis: Replacing Density with Gossip

## Is the change necessary?

Yes, but only under a narrow claim. Gossip should not be presented as a generic optimizer or as a new graph-learning framework. The useful claim is that class-level gossip gives a semantic alternative to density estimation for feature perturbation.

DDFP explores low-density feature regions with a normalizing-flow density model. That is meaningful, but the density gradient is class-agnostic and the flow adds an extra online model. In segmentation, many errors are class-boundary errors: road vs. sidewalk, person vs. rider, chair vs. sofa, and so on. A perturbation direction should therefore know which classes are currently close or confused.

The simplified GossipFP keeps only the pieces needed for that problem:

- an online prototype for each class;
- a normalized relation graph from prototype similarity and teacher confusion;
- a few lazy gossip diffusion rounds;
- the residual `b_c = h_c - p_c` as the perturbation direction;
- fixed-radius pseudo-label consistency on the perturbed feature.

This avoids the earlier over-engineered version with reliability scores, adaptive epsilon, risk weights, and mandatory separation loss.

## What problem does GossipFP solve?

For a pixel predicted as class `c`, density descent asks where the surrounding feature density decreases. GossipFP asks a more semantic question: after class `c` exchanges information with nearby and currently confused classes, where does its prototype disagree with the local consensus? The residual points toward a class-specific boundary field, and the student is trained to keep the original pseudo label under that perturbation.

This is not ordinary graph smoothing. Graph smoothing would use the consensus state `h_c` as the output. GossipFP uses the difference between the consensus and the original prototype. The main ablation must therefore compare residual perturbation against consensus-state perturbation under the same graph.

## Why this can be stronger than density descent

- No normalizing-flow optimizer is needed.
- The perturbation direction is class-conditional instead of class-agnostic.
- Teacher confusion directly identifies boundaries the current model struggles with.
- Row normalization prevents prototype similarity or confusion from dominating only because of scale.
- Multi-round lazy diffusion captures higher-order class neighborhoods, while the residual preserves boundary disagreement.
- Fixed epsilon keeps the method easy to interpret and avoids a large hyperparameter surface.

## Reviewer-risk fixes

- **"Gossip is only packaging."** The method now defines a row-stochastic class graph, applies repeated lazy diffusion, and perturbs with the diffusion residual rather than a nearest negative prototype.
- **"Graph diffusion is common."** The claim is not that diffusion alone is novel. The claim is that the residual of diffusion is a useful semantic perturbation field.
- **"The perturbation direction is mathematically unclear."** The implementation directly optimizes alignment between normalized features and the normalized residual vector. The paper no longer claims that a cosine-difference gradient is algebraically equal to `h_c - p_c`.
- **"Too many heuristic components."** Reliability gates, adaptive radius, risk weighting, and default separation loss have been removed from the main method.
- **"Graph construction is under-specified."** Valid neighbors are observed non-self classes; prototype similarity and teacher confusion are masked and row-standardized before top-K softmax.
- **"No convergence/scalability discussion."** The paper describes lazy diffusion as `P = 0.5I + 0.5A`, uses small `R` to avoid oversmoothing, and reports cost as `O(C^2 d + R C K d)`.

## What must be proven experimentally

- GossipFP improves over the reproduced teacher-student baseline.
- GossipFP improves over DDFP, or matches it with lower overhead.
- Removing teacher-confusion edges hurts performance.
- Removing prototype-similarity edges hurts performance.
- Removing row-normalization hurts performance.
- Using one gossip round is weaker than multi-round gossip.
- Consensus-state perturbation is weaker than residual perturbation.
- The gain is larger on low-label splits, boundary pixels, and rare classes.

## Recommended claim boundary

Do not claim that GossipFP is a general optimizer or a new GNN. The defensible claim is:

> Class-level gossip provides a semantic alternative to density estimation for feature perturbation in semi-supervised semantic segmentation, enabling boundary-aware consistency without an auxiliary generative density model.
