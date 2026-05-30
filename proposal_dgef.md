# Deep GeF (D-GeF): Hierarchical Probabilistic Circuits on Top of Forest Ensembles

**A Research Proposal for the Workshop on Tractable Probabilistic Modeling (TPM)**

---

## 1. Abstract

Generative Forests (GeFs) extend Random Forests (RFs) to full joint probabilistic models by converting each Decision Tree into a Probabilistic Circuit (PC). While this yields tractable marginalisation and principled handling of missing data, the resulting model remains structurally constrained by the *discriminative* tree-learning objective (Gini/Entropy splits). Consequently, the PC structure is not optimised for generative modelling.

We propose **Deep GeF (D-GeF)**, a hierarchical PC architecture that treats an ensemble of GeDTs not as a flat mixture, but as the *base leaves* of a higher-level **Meta-PC**. D-GeF learns a dependency structure *among trees* via Chow-Liu tree estimation and combines them through structured **Product nodes** that model context-specific independencies (CSI) at the ensemble level. This preserves the scalability of RFs while overcoming the systematic limitation of discriminative structure learning, and remains fully tractable for exact inference.

---

## 2. Motivation & Problem Statement

### 2.1 The Systematic Drawback of Current GeFs

A standard GeF is a flat mixture of $T$ GeDTs:

$$p_{\text{GeF}}(X, Y) = \frac{1}{T} \sum_{t=1}^{T} p_t(X, Y)$$

Each $p_t$ is a tree-structured PC induced by a Decision Tree (DT). The DT splits are chosen to maximise discriminative purity (e.g. Gini impurity), not to maximise the log-likelihood of a joint distribution. This creates a **structural mismatch**: the partition of the feature space is designed for classification, not for density estimation.

While GeF(LSPN) mitigates this by fitting complex leaf distributions (LearnSPN), it does **not** change the tree structure. If the hard splits are in the wrong places for generative modelling, even an expressive SPN at the leaf is restricted to a sub-optimal region of the feature space.

### 2.2 What is Missing?

1. **Flat mixture ignores tree-to-tree structure:** An RF is not just a bag of trees. Trees are trained on correlated bootstrap samples and overlapping feature subspaces. A flat average ignores potential context-specific independencies *between* trees.
2. **No generative optimisation at the ensemble level:** Once the forest is built, GeF simply averages. There is no mechanism to re-weight or re-combine trees based on how well they jointly explain the data.
3. **Rigid discriminative skeleton:** The hard splits are immutable after RF training. A purely discriminative skeleton permanently limits how well the joint distribution can be approximated.

### 2.3 D-GeF in a Nutshell

Instead of viewing a GeF as a single Sum node over $T$ GeDTs, D-GeF constructs a **multi-layer PC**:

- **Layer 0:** The original forest of $T$ GeDTs (kept unchanged as base learners).
- **Layer 1:** A **Meta-PC** that learns to combine subsets of trees via structured Sum and Product nodes, guided by their statistical dependencies.

This transforms the ensemble from a *flat mixture* into a *hierarchical, structured PC* that is learned with a **generative objective** at the top level.

---

## 3. Architecture of D-GeF

### 3.1 Layer 0: Base GeDT Ensemble

Given a trained Random Forest with $T$ trees, each tree $DT_t$ is converted into a GeDT $PC_t$ using Algorithm 1 from (Correia et al., 2020). We **do not modify** these base PCs. They serve as:
- Tractable base distributions $p_t(X,Y)$.
- Feature extractors that map an input $x$ to a leaf identifier $l_t(x)$.

### 3.2 Layer 1: The Meta-PC

The Meta-PC is a higher-level PC whose *scope* is the ensemble of trees. Its root is a **Sum node** over $K$ groups ($K \ll T$):

$$p_{\text{D-GeF}}(X, Y) = \sum_{k=1}^{K} w_k \cdot \phi_k(X, Y)$$

where each $\phi_k$ is a **Product node** over a subset of trees $\mathcal{C}_k \subseteq \{1, \dots, T\}$.

#### From Product of Trees to Product of Residuals

A naive product $\prod_{t \in \mathcal{C}_k} p_t(X,Y)$ is invalid because all $p_t$ share the same scope $(X,Y)$, violating decomposability. We instead use a **residual decomposition**:

1. Define a **base distribution** as the flat GeF mixture over the group:
   $$p_{0,k}(X,Y) = \frac{1}{|\mathcal{C}_k|} \sum_{t \in \mathcal{C}_k} p_t(X,Y)$$

2. For each tree $t \in \mathcal{C}_k$, define a **residual branch** $r_t(X,Y)$ that captures how $p_t$ differs from the base. For tractability, we restrict $r_t$ to a lightweight PC (e.g. a Chow-Liu tree or a fully factorised model) over a *subset* of features that tree $t$ specialises in.

3. The group factor is then:
   $$\phi_k(X, Y) = p_{0,k}(X,Y) \cdot \prod_{(i,j) \in \mathcal{E}_k} f_{ij}(X_{ij}, Y)$$
   where $\mathcal{E}_k$ are edges in the dependency graph of group $k$, and $f_{ij}$ are pairwise residual factors with disjoint scopes $X_{ij}$ (ensuring decomposability).

### 3.3 Illustrative Structure

```
[Sum] Root (weights w_1 ... w_K)
 |
 |-- [Product] Group 1: phi_1(X,Y)
 |    |-- [Sum] Base: (p_1 + p_2 + p_3)/3
 |    |-- [Prod] Residual Edge (1,2): f_{1,2}(X_1, X_3)
 |    |-- [Prod] Residual Edge (2,3): f_{2,3}(X_5)
 |
 |-- [Product] Group 2: phi_2(X,Y)
 |    |-- [Sum] Base: (p_4 + p_5)/2
 |    |-- [Prod] Residual Edge (4,5): f_{4,5}(X_2, X_7)
```

This is a valid PC because:
- Each Sum node is **smooth** (all children have the same scope).
- Each Product node is **decomposable** (children have disjoint scopes, by construction via feature-subset assignment in residual factors).

---

## 4. Structure Learning Algorithm for the Meta-PC

### Step 1: Extract Tree-Level Statistics

For each training sample $x^{(n)}$ and each tree $t$:
1. Compute the **leaf assignment** $l_t^{(n)} = \text{leaf}_t(x^{(n)})$.
2. Compute the **log-likelihood contribution** $\log p_t(x^{(n)})$.

This yields two matrices:
- $\mathbf{L} \in \{1, \dots, V\}^{N \times T}$ (leaf IDs, $V$ = max leaves per tree)
- $\mathbf{P} \in \mathbb{R}^{N \times T}$ (log-likelihoods)

### Step 2: Estimate Pairwise Tree Dependencies

We compute the **empirical Mutual Information (MI)** between leaf assignments of every tree pair $(t_i, t_j)$:

$$I(L_i; L_j) = \sum_{a=1}^{V_i} \sum_{b=1}^{V_j} P(L_i=a, L_j=b) \log \frac{P(L_i=a, L_j=b)}{P(L_i=a)P(L_j=b)}$$

High MI indicates that two trees partition the data in a correlated way, meaning they capture overlapping or complementary structure. Low MI suggests conditional independence.

### Step 3: Learn a Chow-Liu Tree over the Ensemble

We build a complete undirected graph with $T$ nodes (one per tree) and edge weights $I(L_i; L_j)$. Running the **Chow-Liu Maximum Spanning Tree** algorithm yields a tree-structured graphical model over the ensemble:

$$\mathcal{T}_{\text{CL}} = \arg\max_{\text{tree } \mathcal{T}} \sum_{(i,j) \in \mathcal{E}(\mathcal{T})} I(L_i; L_j)$$

This tree captures the *most informative* dependencies among trees while keeping the meta-structure tractable.

### Step 4: Cluster the Chow-Liu Tree into Groups

The Chow-Liu tree is decomposed into $K$ connected components (or clusters) using a lightweight graph clustering algorithm (e.g. spectral clustering or simply threshold-based edge cutting). Each cluster $k$ forms a group $\mathcal{C}_k$.

*Why clustering?* 
- A single Product node over all $T$ trees would require too many residual factors.
- Clustering isolates strongly coupled trees into groups, where Product nodes are most meaningful.

### Step 5: Construct Residual Factors

For each edge $(i,j)$ within a group $\mathcal{C}_k$:
1. Identify the **feature subsets** that trees $i$ and $j$ primarily split on (available from the tree structures).
2. If their primary feature sets are disjoint, $f_{ij}$ can be a simple factorised distribution over the union.
3. If they overlap, $f_{ij}$ is learned as a small PC (e.g. Chow-Liu tree or Gaussian Copula) over the overlapping features, trained to maximise the likelihood of the *ratio* $p_i / p_{0,k}$.

### Step 6: Assemble the Meta-PC

1. Create a **Sum node** root.
2. For each group $k$:
   - Create a **Product node** $\phi_k$.
   - Add the base mixture $p_{0,k}$ as a child.
   - Add all residual factors $f_{ij}$ for $(i,j) \in \mathcal{E}_k$ as children (ensuring disjoint scopes via feature assignment).
   - Add $\phi_k$ to the root Sum node with weight $w_k$ proportional to the group's training likelihood.

---

## 5. Pseudocode

```python
def learn_dgef(X_train, y_train, n_estimators=100, K=10):
    # Stage 1: Train RF and convert to standard GeF
    rf = RandomForest(n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    gef = rf.topc()  # Flat mixture of GeDTs

    T = n_estimators
    N = X_train.shape[0]

    # Stage 2: Extract leaf assignments and log-probs
    L = np.zeros((N, T), dtype=int)
    log_probs = np.zeros((N, T))
    for t in range(T):
        pc_t = gef.root.children[t]
        for n in range(N):
            x_n = np.concatenate([X_train[n], [y_train[n]]])
            L[n, t] = get_leaf_assignment(pc_t, x_n)
            log_probs[n, t] = evaluate(pc_t, x_n)

    # Stage 3: Compute MI matrix between trees
    MI = np.zeros((T, T))
    for i in range(T):
        for j in range(i+1, T):
            MI[i, j] = compute_mutual_information(L[:, i], L[:, j])
            MI[j, i] = MI[i, j]

    # Stage 4: Chow-Liu maximum spanning tree
    cl_tree = chow_liu_maximum_spanning_tree(MI)

    # Stage 5: Cluster into K groups
    groups = spectral_cluster_tree(cl_tree, n_clusters=K)

    # Stage 6: Build Meta-PC
    root = SumNode(scope=scope_all, n=N)

    for group in groups:
        # Base distribution: flat mixture over trees in group
        base_sum = SumNode(scope=scope_all, n=len(group))
        for t in group:
            base_sum.add_child(gef.root.children[t], weight=1.0/len(group))

        # Product node for this group
        prod = ProdNode(scope=scope_all, n=N)
        prod.add_child(base_sum)

        # Add residual factors for edges within this group
        for (i, j) in get_edges(group, cl_tree):
            overlap_features = get_overlap_features(gef.root.children[i],
                                                    gef.root.children[j])
            if len(overlap_features) == 0:
                # Disjoint scope -> simple factorised residual
                res = build_factorised_residual([i, j], X_train)
            else:
                # Overlapping scope -> small Chow-Liu PC on overlap
                res = build_chow_liu_residual([i, j], overlap_features, X_train)
            prod.add_child(res)

        # Weight of this group from training LL
        group_ll = np.mean(np.logsumexp(log_probs[:, group], axis=1))
        root.add_child(prod, weight=np.exp(group_ll))

    # Normalise root weights
    root.normalise_weights()

    dgef = PC()
    dgef.root = root
    dgef.ncat = gef.ncat
    return dgef
```

---

## 6. Tractability Analysis

D-GeF is a Probabilistic Circuit by construction. We verify the structural properties required for tractability:

| Property | Verification in D-GeF |
|----------|----------------------|
| **Smoothness** | Every Sum node (root and base mixtures) combines children with identical scope $(X, Y)$. Residual factors may have smaller scopes, but they are always placed under Product nodes, not Sum nodes. Thus all Sum nodes are smooth. |
| **Decomposability** | Product nodes combine: (a) a base mixture with scope $(X,Y)$, and (b) residual factors with **disjoint sub-scopes** $\{X_{ij}\}$. To ensure strict decomposability, we constrain each residual factor $f_{ij}$ to use only features *not used* by any other sibling residual factor in the same Product node. This is achievable because trees naturally split on feature subsets. |
| **Determinism** | The root Sum and base mixtures are deterministic if tree weights are non-overlapping (hard in general). However, **determinism is not required** for tractable marginals — only smoothness and decomposability are. For MPE inference, we can optionally enforce determinism via argmax routing. |
| **Marginal Complexity** | Marginalisation of any subset $X_o \subseteq X$ is linear in the size of the circuit. The Meta-PC adds at most $O(T^2)$ nodes (for residual edges), which is small compared to the base forest. Thus inference remains tractable and efficient. |

### Complexity
- **Learning:** Dominated by base RF training ($O(T \cdot N \log N)$) and MI estimation ($O(T^2 \cdot N)$). Meta-PC construction is $O(T^2)$.
- **Inference:** Linear in circuit size: $O(|\text{D-GeF}|) = O(T \cdot |\text{tree}| + T^2)$, which is comparable to GeF for moderate $T$.

---

## 7. Why This Overcomes the Discriminative Limitation

| Aspect | Standard GeF | GeF(LSPN) | D-GeF |
|--------|-------------|-----------|-------|
| **Split structure** | Hard splits from discriminative DT | Same hard splits | Same hard splits at base, **but** supplemented by generative structure at meta-layer |
| **Leaf distribution** | Fully factorised | LearnSPN (complex PC) | Base: GeDT leaves; Meta: structured combinations |
| **Ensemble model** | Flat mixture $\frac{1}{T}\sum p_t$ | Flat mixture | Hierarchical PC with learned dependencies |
| **Generative objective at ensemble level** | None | None | **Yes** — Meta-PC weights and residual factors are learned via log-likelihood and MI |
| **Tree-to-tree dependencies** | Ignored | Ignored | Explicitly modelled via Chow-Liu tree |

**The key insight:** D-GeF does not *replace* the discriminative skeleton; it **wraps** it inside a generative architecture. The hard splits remain, but their influence on the final joint distribution is mediated through a learned generative layer that can up-weight, down-weight, and multiplicatively combine trees based on how they jointly explain the data.

---

## 8. Connection to Tractable Probabilistic Modeling (TPM)

### 8.1 Relevance to TPM Themes

1. **Probabilistic Circuits as unifying framework:** D-GeF literally builds a PC on top of another PC, demonstrating how PCs can be composed hierarchically.
2. **Structure Learning:** The Meta-PC is learned via classical structure learning techniques (Chow-Liu tree, MI estimation) applied at the *ensemble level* — a novel form of structure learning for "PCs over PCs".
3. **Context-Specific Independence (CSI):** D-GeF exploits CSI at two levels:
   - **Intra-tree:** The original DT splits create context-specific regions.
   - **Inter-tree:** The Chow-Liu tree identifies which trees are conditionally independent given the data context (i.e., in which regions of the feature space their leaf assignments decouple).
4. **Bridging discriminative and generative worlds:** A core challenge in TPM is making tractable models competitive with discriminative models. D-GeF shows a principled way to *retrofit* discriminative ensembles with generative structure without re-training from scratch.

### 8.2 Novelty

To the best of our knowledge, no prior work has proposed:
- Learning a **dependency structure among trees** in a Random Forest via a Chow-Liu tree.
- Constructing a **hierarchical PC where the base leaves are themselves complex PCs** (GeDTs).
- Using **residual decomposition** to maintain decomposability when combining multiple full-joint distributions.

---

## 9. Empirical Plan & Future Work

### 9.1 Baselines
1. Standard Random Forest (scikit-learn)
2. Standard GeF (fully factorised leaves)
3. GeF(LSPN) (LearnSPN leaves)
4. XGBoost / LightGBM (for discriminative reference)

### 9.2 Datasets
- OpenML-CC18 benchmark (used in the original GeF paper)
- Real-world tabular datasets with mixed variable types (continuous + categorical)

### 9.3 Evaluation Metrics
- **Classification accuracy** with missing features at test time (MAR)
- **Log-likelihood** on held-out data (generative quality)
- **Robustness values** (from Correia et al., 2020)
- **Out-of-distribution detection** (AUC-ROC using log-density)

### 9.4 Research Questions
1. Does D-GeF achieve higher log-likelihood than flat GeF/GeF(LSPN) while maintaining comparable classification accuracy?
2. How sensitive is D-GeF to the choice of $K$ (number of groups)?
3. Can the Meta-PC structure be interpreted to reveal which trees specialise on which data regions?

### 9.5 Future Extensions
- **Gradient-based refinement:** Use gradient descent to fine-tune Meta-PC weights and residual factor parameters after structure learning.
- **Deep stacking:** Add more than one Meta-PC layer (e.g. a third layer combining groups of groups).
- **Online learning:** Update the Chow-Liu tree structure incrementally as new data arrives.

---

## 10. References

1. Correia, A. H. C., Peharz, R., & de Campos, C. P. (2020). *Joints in Random Forests*. NeurIPS 2020.
2. Correia, A. H. C., Peharz, R., & de Campos, C. P. (2020). *Towards Robust Classification with Deep Generative Forests*. ICML 2020 Workshop on Uncertainty and Robustness in Deep Learning.
3. Gens, R., & Domingos, P. (2013). *Learning the Structure of Sum-Product Networks*. ICML 2013.
4. Chow, C., & Liu, C. (1968). *Approximating Discrete Probability Distributions with Dependence Trees*. IEEE Transactions on Information Theory.
5. Poon, H., & Domingos, P. (2011). *Sum-Product Networks: A New Deep Architecture*. UAI 2011.
6. Van den Broeck, G., et al. (2019). *Tractable Probabilistic Models: Representations, Algorithms, Learning, and Applications*. Tutorial at UAI 2019.
7. Peharz, R., et al. (2015). *On Theoretical Properties of Sum-Product Networks*. AISTATS 2015.

---

## 11. Conclusion

D-GeF proposes a principled way to overcome the systematic limitation of discriminative structure in GeFs. By constructing a **Meta-PC** over an ensemble of base GeDTs, we retain the scalability and predictive power of Random Forests while unlocking a more expressive, generative structure at the ensemble level. The use of **Chow-Liu tree learning**, **mutual information analysis**, and **residual factorisation** ensures that the resulting model remains a valid, tractable Probabilistic Circuit suitable for exact marginalisation and robust classification under missing data.

This work is directly aligned with the mission of the **Tractable Probabilistic Modeling (TPM)** workshop: advancing the theory and application of models that guarantee efficient, reliable, and exact probabilistic reasoning.

---

*Proposal prepared for the Workshop on Tractable Probabilistic Modeling (TPM).*
