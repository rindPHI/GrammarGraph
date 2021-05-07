# GrammarGraph

Creating graphs from context-free grammars for fun and profit.

## Features

* Creating sub graphs
* Export back to grammars
* Reachability
* Filter abstraction
* Dijkstra's algorithm for shortest paths between nodes
* Checking whether a (sub) graph represents a tree

Have a look at our [feature demo](DEMO/DEMO.md)!

## Install

### Globally

Run `python3 setup.py install`.

### Build & Use in Pipenv

Run `python3 setup.py sdist`. Then, you can include it into another project by adding
the following to your `Pipfile` (and running `pipenv update` afterward):

```toml
[[source]]
# ...

[packages]
# ...
grammargraph = {path = "path/to/GrammarGraph/dist/grammargraph-0.0.1.tar.gz"}

# ...
```

### Use w/o Installation

The project comes with a `Pipfile` from which you can create a pipenv environment to
use GrammarGraph without building it. For this, run

```
pipenv install
pipenv shell
```

Author: [Dominic Steinhoefel](mailto:dominic.steinhoefel@cispa.de)
