# ASG-RNN-like Benchmark Run

This is an ASG-RNN-like adaptation, not a verified reproduction of the ASG-RNN paper.
Local search did not find a direct ASG-RNN implementation. The experiment uses the existing M5 protocol and an engineered hybrid graph.

Protocol: max_days=365, context_length=28, val_days=28, test_days=28, seed=42.
Benchmark products: FOODS_3_228_CA_1_validation, FOODS_2_044_CA_3_validation, HOBBIES_1_133_CA_4_validation.
Graph construction: top-k hybrid similarity using log-sales correlation, price correlation, M5 metadata similarity, and demand-profile similarity.
Model: GRU temporal encoder + static node feature encoder + adjacency-weighted neighbor aggregation + point forecast head.
Expanded panel rule: reproducible balanced sampling from CA M5 products across stable/high, intermittent, and low-volume demand classes, while forcing the benchmark trio to remain in the panel.