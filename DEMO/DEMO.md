GrammarGraph lets you build a directed graph from a context-free grammar (CFG).
A context-free grammar is a mapping from nonterminals to lists of expansion alternatives.
For instance, the following grammar represents a CSV file:


```python
import string
from typing import List, Dict

CSV_GRAMMAR: Dict[str, List[str]] = {
    '<start>': ['<csvline>'],
    '<csvline>': ['<items>'],
    '<items>': ['<item>,<items>', '<item>'],
    '<item>': ['<letters>'],
    '<letters>': ['<letter><letters>', '<letter>'],
    '<letter>': list(string.ascii_letters + string.digits + string.punctuation + ' \t\n')
}
```

This particular representation of a grammar is based on the [Fuzzing Book](https://www.fuzzingbook.org/).
Such a grammar can, e.g., be used to create random CSV files (based on Fuzzing Book implementations,
this is independent from GrammarGraph):


```python
from fuzzingbook.GrammarCoverageFuzzer import GrammarCoverageFuzzer

fuzzer = GrammarCoverageFuzzer(CSV_GRAMMAR)

for _ in range(10):
    print(fuzzer.fuzz())
```

    g ,V
    O
    /
    a+,P9d(
    &\?
    -{
    Y=
    5m
    j
    w@,E


Using GrammarGraph, we can visualize this grammar based on GraphViz. To make sure everything
fits on the screen, we reduce the number of alternatives for the `<letter>` nonterminal.


```python
CSV_GRAMMAR["<letter>"] = ['a', 'b', 'c', '1', '2', '3']

from grammar_graph.gg import GrammarGraph

graph = GrammarGraph.from_grammar(CSV_GRAMMAR)
graph.to_dot()
```




    
![svg](output_5_0.svg)
    



We can create and visualize sub graphs:


```python
graph.subgraph("<letters>").to_dot()
```




    
![svg](output_7_0.svg)
    



We can also check whether a subgraph is a tree structure (and can thus, e.g., be trivially
converted to a regular expression), or whether a node is reachable from another one:


```python
graph.subgraph("<letter>").is_tree()
```




    True




```python
graph.subgraph("<letters>").is_tree()
```




    False




```python
letters = graph.get_node("<letters>")
letters.reachable(letters)
```




    True



You can also create a sub grammar from a sub graph.


```python
graph.subgraph("<letters>").to_grammar()
```




    {'<start>': ['<letters>'],
     '<letters>': ['<letter><letters>', '<letter>'],
     '<letter>': ['a', 'b', 'c', '1', '2', '3']}



GrammarGraph features implementations of breadth-first search, filtering, and shortest path
discovery (based on Dijkstra's algorithm with Fibonacci heaps). This can, e.g., be used to
embed a subtree into a bigger context. For instance, the shortest path from `<items>` to `<letter>` is:


```python
[node.symbol for node in graph.shortest_path(graph.get_node("<items>"), graph.get_node("<letter>"))]
```




    ['<items>', '<item>', '<letters>', '<letter>']



Let us assume we have CSV item `"abc"` from which we want to create a (random) CSV file. We can accomplish
this by finding the shortest path from `<start>` to `<item>` and follow this path, choosing an appropriate
grammar production rule along the way.


```python
from fuzzingbook.Grammars import unreachable_nonterminals, is_nonterminal
from fuzzingbook.Parser import EarleyParser
import copy

item_string = "abc"

# We massage the grammar a little bit to get a tree starting at `<item>`
item_grammar = copy.deepcopy(CSV_GRAMMAR)
item_grammar["<start>"] = ["<item>"]
for unreachable in unreachable_nonterminals(item_grammar):
    del item_grammar[unreachable]

item_tree = next(EarleyParser(item_grammar).parse(item_string))[1][0]
item_tree
```




    ('<item>',
     [('<letters>',
       [('<letter>', [('a', [])]),
        ('<letters>',
         [('<letter>', [('b', [])]),
          ('<letters>', [('<letter>', [('c', [])])])])])])



The path we need to follow for creating a complete file embedding `item` is:


```python
item_node = graph.get_node("<item>")

path = [node.symbol for node in graph.shortest_path(graph.root, item_node)]
path
```




    ['<start>', '<csvline>', '<items>', '<item>']



So let's create a derivation tree. In principle, such a tree will be incomplete, namely if an
expansion alternative contains nonterminals which we do not have to follow. Such incomplete
nodes have `None` as children and can later on be instantiated, e.g., by a fuzzer. For our simple
CSV grammar, however, this is not the case. In the algorithm sketched below we still account for this.

We use a "canonical" grammar representation for simplicity.


```python
from fuzzingbook.Parser import canonical

canonical_grammar = canonical(CSV_GRAMMAR)
canonical_grammar
```




    {'<start>': [['<csvline>']],
     '<csvline>': [['<items>']],
     '<items>': [['<item>', ',', '<items>'], ['<item>']],
     '<item>': [['<letters>']],
     '<letters>': [['<letter>', '<letters>'], ['<letter>']],
     '<letter>': [['a'], ['b'], ['c'], ['1'], ['2'], ['3']]}




```python
assert graph.root.reachable(item_node)


def wrap_in_tree_starting_in(start_nonterminal: str, tree, grammar, graph: GrammarGraph):
    start_node = graph.get_node(start_nonterminal)
    end_node = graph.get_node(tree[0])
    assert start_node.reachable(end_node)

    derivation_path = [n.symbol for n in graph.shortest_non_trivial_path(start_node, end_node)]

    result = (start_nonterminal, [])
    curr_tree = result
    for path_idx in range(len(derivation_path) - 1):
        path_nonterminal = derivation_path[path_idx]
        next_nonterminal = derivation_path[path_idx + 1]
        alternatives_for_path_nonterminal = [a for a in grammar[path_nonterminal]
                                             if next_nonterminal in a]
        shortest_alt_for_path_nonterminal = \
            [a for a in alternatives_for_path_nonterminal
             if not any(a_ for a_ in alternatives_for_path_nonterminal
                        if len(a_) < len(a))][0]
        idx_of_next_nonterminal = shortest_alt_for_path_nonterminal.index(next_nonterminal)
        for alt_idx, alt_symbol in enumerate(shortest_alt_for_path_nonterminal):
            if alt_idx == idx_of_next_nonterminal:
                if path_idx == len(derivation_path) - 2:
                    curr_tree[1].append(tree)
                else:
                    curr_tree[1].append((alt_symbol, []))
            else:
                curr_tree[1].append((alt_symbol, None if is_nonterminal(alt_symbol) else []))

        curr_tree = curr_tree[1][idx_of_next_nonterminal]

    return result


wrapped_tree = wrap_in_tree_starting_in("<start>", item_tree, canonical_grammar, graph)
wrapped_tree
```




    ('<start>',
     [('<csvline>',
       [('<items>',
         [('<item>',
           [('<letters>',
             [('<letter>', [('a', [])]),
              ('<letters>',
               [('<letter>', [('b', [])]),
                ('<letters>', [('<letter>', [('c', [])])])])])])])])])




```python
from fuzzingbook.GrammarFuzzer import tree_to_string

tree_to_string(wrapped_tree)
```




    'abc'



Now let's assume that we want to add another item into that tree. We can do so, e.g., by
replacing the top-level `<items>` node with another `<items>` node expanded once more.
The shortest node from `<items>` to `<items>`, however, is trivial:


```python
items = graph.get_node("<items>")
[node.symbol for node in graph.shortest_path(items, items)]
```




    ['<items>']



The function `get_shortest_non_trivial_path` returns a nontrivial path which is useful for
our purposes:


```python
[node.symbol for node in graph.shortest_non_trivial_path(items, items)]
```




    ['<items>', '<items>']



The astute reader may have discovered that this method is also used in the above
declaration of `wrap_in_tree_starting_in`, which is why we can use this routine
also for mapping a tree into one starting with the same nonterminal!


```python
items_tree = ('<items>',
              [('<item>',
                [('<letters>',
                  [('<letter>', [('a', [])]),
                   ('<letters>',
                    [('<letter>', [('b', [])]),
                     ('<letters>', [('<letter>', [('c', [])])])])])])])

wrapped_tree = wrap_in_tree_starting_in("<items>", items_tree, canonical_grammar, graph)
wrapped_tree
```




    ('<items>',
     [('<item>', None),
      (',', []),
      ('<items>',
       [('<item>',
         [('<letters>',
           [('<letter>', [('a', [])]),
            ('<letters>',
             [('<letter>', [('b', [])]),
              ('<letters>', [('<letter>', [('c', [])])])])])])])])



If we now embed this tree into a one starting in `<start>`, we have an incomplete derivation tree
with a "hole" for another item.


```python
wrapped_tree = wrap_in_tree_starting_in("<start>", wrapped_tree, canonical_grammar, graph)
wrapped_tree
```




    ('<start>',
     [('<csvline>',
       [('<items>',
         [('<item>', None),
          (',', []),
          ('<items>',
           [('<item>',
             [('<letters>',
               [('<letter>', [('a', [])]),
                ('<letters>',
                 [('<letter>', [('b', [])]),
                  ('<letters>', [('<letter>', [('c', [])])])])])])])])])])




```python
tree_to_string(wrapped_tree)
```




    ',abc'



Let's complete this tree using the fuzzer:


```python
complete_tree = fuzzer.expand_tree(wrapped_tree)
complete_tree
```




    ('<start>',
     [('<csvline>',
       [('<items>',
         [('<item>', [('<letters>', [('<letter>', [('1', [])])])]),
          (',', []),
          ('<items>',
           [('<item>',
             [('<letters>',
               [('<letter>', [('a', [])]),
                ('<letters>',
                 [('<letter>', [('b', [])]),
                  ('<letters>', [('<letter>', [('c', [])])])])])])])])])])




```python
tree_to_string(complete_tree)
```




    '1,abc'



We think that representing CFGs as graphs is useful for a number of purposes such as the one
we looked into just now. Hopefully it can help you too!
