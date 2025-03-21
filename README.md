# KMSnet
This repository containes the code for Kubo-Martin-Schwinger (KMS) states on the $C^*$-algebra of a directed network originally developped for the papers

E. Moutuou and H. Benali - *KMS states of Information Flow in Directed Brain Synaptic Networks*, 2024, arXiv:2410.18222

and 

E. Moutuou and H. Benali - *Brain functions emerge as thermal equilibrium states of the connectome*, 2024, arXiv:2408.14221


The KMS states of the graph algebra of a directed multigraph $G$ are represented by probability distributions on the nodes. They are generated by pure KMS state vectors $x^{j|\beta}$ where $j$ is a node and $\beta$ is a paremeter representing inverse temperature. These pure KMS states describe the **interaction profile** of each nodes at inverse temperature $\beta$. The graph $G$ is supposed to be a `networkx.MultiDiGraph()` object. 

This code can also be used to produce publishable visualizations of directed multigraphs.

## Critical inverse temperature
The parameter $\beta$ must be larger than the critical value $\beta_c$ which is computed via:

* `states.critical_inverse_temperature(G)`

## Interaction profiles
The interaction profile of a node $j$ at inverse temperature $\beta > \beta_c$ is computed via the function:

* `states.get_interaction_profile(G, j, beta)` which returns the labelled list of nodes and the 1-array representing the distribution $x^{j|\beta}$. 

## KMS matrices
To get the KMS matrix whose columns are the interaction profiles $x^{j|\beta}$, use the function 

* `states.kms_matrix(G, beta, with_feedback=True)`. (or `with_feedback=False` if you want to remove trivial loops and self-interactions, which basically removes the diagonal entries and then renormalizes the columns).




