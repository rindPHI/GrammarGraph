# GrammarGraph

Creating graphs from context-free grammars for fun and profit.

## Features

* Creating sub graphs
* Export back to grammars
* Reachability
* Filter abstraction
* Dijkstra's algorithm for shortest paths between nodes
* Checking whether a (sub) graph represents a tree
* Computing k-paths (paths of exactly length k) in grammars and derivation trees, and a 
  k-path coverage measure ([see this paper](https://ieeexplore.ieee.org/document/8952419)) of 
  derivation trees based on that.

Have a look at our [feature demo](DEMO/DEMO.md)!

## Install

GrammarGraph requires at least Python 3.9.

We recommend to install GrammarGraph in a virtual environment. Example usage (inside project directory):

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Run tests
python3 -m pytest -n 16
```

Author: [Dominic Steinhöfel](https://www.dominic-steinhoefel.de).
