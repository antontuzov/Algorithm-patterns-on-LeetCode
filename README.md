# 🚀 Algorithm Patterns in Python

Welcome to the **Algorithm Patterns in Python** repository! 🐍 This repo includes Python implementations of some of the most common and powerful algorithm patterns frequently used in coding interviews, especially on LeetCode. 📝

Each algorithm pattern is demonstrated with a real problem and Python solution, making it easier to understand and apply.

## 📂 Table of Contents
- [Patterns Overview](#-patterns-overview)
- [Algorithm Patterns](#-algorithm-patterns)
  - [Two Pointers](#1-two-pointers)
  - [Sliding Window](#2-sliding-window)
  - [Binary Search](#3-binary-search)
  - [Fast & Slow Pointers](#4-fast--slow-pointers)
  - [Depth-First Search (DFS)](#5-depth-first-search-dfs)
  - [Breadth-First Search (BFS)](#6-breadth-first-search-bfs)
  - [Merge Intervals](#7-merge-intervals)
  - [Topological Sort](#8-topological-sort)
  - [Dynamic Programming (DP)](#9-dynamic-programming-dp)
  - [Union-Find (Disjoint Set)](#10-union-find-disjoint-set)
- [Getting Started](#-getting-started)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📋 Patterns Overview

Below is a brief description of each algorithm pattern and its applications:

| Pattern                  | Use Cases                                                                                  |
|--------------------------|--------------------------------------------------------------------------------------------|
| **Two Pointers**         | Problems involving pairs or sequence search                                                |
| **Sliding Window**       | Substring or contiguous subarray problems                                                  |
| **Binary Search**        | Efficient search in sorted arrays                                                          |
| **Fast & Slow Pointers** | Detecting cycles, finding middle elements                                                  |
| **DFS**                  | Tree/graph traversal, backtracking                                                         |
| **BFS**                  | Shortest path in unweighted graphs                                                         |
| **Merge Intervals**      | Merging overlapping intervals                                                              |
| **Topological Sort**     | Dependency resolution, ordering in DAGs                                                   |
| **Dynamic Programming**  | Optimization problems, sequence alignment, knapsack problems                              |
| **Union-Find**           | Connected components, cycle detection                                                     |

## 📖 Algorithm Patterns

### 1. Two Pointers
🚦 Efficiently find pairs, palindromes, and handle sorted arrays.

#### Example: [Two Sum II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
```python
def two_sum(numbers, target):
    left, right = 0, len(numbers) - 1
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
