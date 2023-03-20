# Recommender system
## Dataset
- MovieLens Latest Datasets: These datasets will change over time, and are not appropriate for reporting research results. We will keep the download links stable for automated downloads. We will not archive or make available previously released versions.
- Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
## About the project:
- Apply the LightGCN model for recommending the movies which the users may like based on the moving ratings
- 3.5 as the threshold for positive rating. Any interaction is considered an edge
- Self-supervised-based pre-training on graphs
- Metrics: Precision and Recall, Normalized Dicounted Cumulative Gain (NDCG)
## References:
- [LightGCN: Simplifying and Powering Graph ConvolutionNetwork for Recommendation](https://arxiv.org/pdf/2002.02126.pdf)
- [Recommender system using Bayesian personalized ranking](https://towardsdatascience.com/recommender-system-using-bayesian-personalized-ranking-d30e98bba0b9)
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907v4)
- [Self-supervised Learning on Graphs: Contrastive, Generative,or Predictive](https://arxiv.org/pdf/2105.07342v4.pdf)