# Benchmark Product Selection

Official reference products selected from the same 16 M5 items used in `notebooks/29_graph_neural_network.ipynb`.

## Final Selection

1. High-demand / stable
   - `FOODS_3_228_CA_1_validation`
   - justification: highest mean demand in the 16-product panel (`mean_sales ~= 7.01`), almost no zeros (`zero_rate ~= 0.011`), very high activity (`361` non-zero days), and low relative variability (`cv ~= 0.50`). This is the clearest smooth high-demand benchmark.

2. Intermittent
   - `FOODS_2_044_CA_3_validation`
   - justification: clearly intermittent without collapsing into the extreme low-volume corner. It has moderate sparsity (`zero_rate ~= 0.608`), only `143` non-zero days, and intermittent-demand descriptors consistent with that regime (`ADI ~= 2.55`, `CV2 ~= 0.39`).

3. Low-volume
   - `HOBBIES_1_133_CA_4_validation`
   - justification: lowest average demand in the selected panel (`mean_sales ~= 0.055`), the fewest active days (`19` non-zero days), and the highest sparsity (`zero_rate ~= 0.948`). This is the strongest low-volume stress-test case.

## Rationale

This trio gives a clean benchmark spectrum for later comparisons:

- one smooth and consistently active product
- one genuinely intermittent product with usable recurring demand
- one extremely sparse low-volume product for stress testing
