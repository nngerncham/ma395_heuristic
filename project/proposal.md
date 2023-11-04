# Project Proposal (Nawat)

Topic: Multi-objective Optimization on Hyper-parameters for Constructing Graphs for ANNS

## Background

My senior project is on supporting removing points from a graph that is constructed to support fast and accurate queries to answer approximate nearest neighbor search problems. The first step to this is to construct the index of this graph, i.e., figuring out which vertices should be connected to other vertices. For context, each vertex represents a vector or a point in $\mathbb{R}^d$ so the distance between each vertex is simply the Euclidean distance.

The goal is to make sure that the index construction does not take too long while making sure that the resulting graph allows for fast and accurate queries. That is, we have a few objectives that we are optimizing for:

1. Minimizing the index construction time
2. Maximizing recall (how accurate it is compared to ground truth)
3. Minimizing query time (how long it takes to get the answer)
4. Minimizing how much memory is required for storing the graph

The algorithms used for index construction that I'm interested in are called Vamana and HNSW and both have a few parameters I can tune:

1. Maximum degree of a vertex
2. Candidate collection size during index construction
   - During construction, both algorithm will need to store a number of neighbor candidates when deciding which points should be connected to another
3. Candidate collection size during search
4. (Vamana only) Closeness threshold
   - Vamana aims to make sure that the proximity graph that it constructs is one where close points are connected and far points are only connected if nothing near it is connected.
   - Let's say we have points $x, y, z$ where $y$ and $z$ are close enough---within the threshold---to each other but both are far from $x$. Then, $x$ is connected to either $y$ or $z$, not both
5. (HNSW only) Probability normalizing factor
   - HNSW constructs a multi-layer proximity graph where we use this parameter to decide the probability of a point showing up in the higher layers

The dataset used to evaluate ANNS are rather big, usually with around a million points. Running the index construction algorithm once could take a really long time (I've seen times from 5 minutes to an hour). So, I want to

## Idea

I'll split this into 2 phases so that I have an achievable minimum point and something I can investigate further if I have time (not very likely):

1. Start by running GA to optimize the hyper-parameters on a toy data set using high number of iterations. I will use a data set with 10000 points for this because---from my experience---the index for this many data points runs in just a few seconds or less. Then, I will optimize the hyper-parameters further on the bigger data set but use very low number of iterations by using the values from the toy problems as a starting point.
2. Look into surrogate-based optimization to try to do this optimization _properly_.

## Reference

- [Vamana](https://papers.nips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
- [HNSW](https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf)
