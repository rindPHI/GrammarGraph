import html
import re
from functools import lru_cache
from typing import List, Dict, Callable, Union, Optional, Tuple, cast

import fibheap as fh
import sys
from fuzzingbook.Grammars import is_nonterminal, RE_NONTERMINAL
from fuzzingbook.fuzzingbook_utils import unicode_escape
from graphviz import Digraph
from orderedset import OrderedSet

NonterminalType = str
Grammar = Dict[NonterminalType, List[str]]


def split_expansion(expansion: str) -> List[str]:
    return [token for token in re.split(RE_NONTERMINAL, expansion) if token]


class Node:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def __hash__(self):
        return hash(self.symbol)

    def __lt__(self, other):
        # Needed for fibheap
        return self.symbol < other.symbol

    def quote_symbol(self):
        return '"' + self.symbol.translate(str.maketrans({'"': r"\""})) + '"'

    @lru_cache
    def reachable(self, to_node: 'Node') -> bool:
        # Note: Reachability is not reflexive!
        graph = GrammarGraph(self)
        f = lambda node: isinstance(node, NonterminalNode) and to_node in node.children
        sources = graph.filter(f, f)
        return len(sources) > 0


class NonterminalNode(Node):
    def __init__(self, symbol: str, children: List[Node]):
        super().__init__(symbol)
        self.children = children  # in fact, all children will be ChoiceNode instances.

    def __repr__(self):
        return f"NonterminalNode({self.quote_symbol()}, {repr(self.children)})"


class ChoiceNode(NonterminalNode):
    def __init__(self, symbol: str, children: List[Node]):
        super().__init__(symbol, children)

    def __repr__(self):
        return f"ChoiceNode({self.symbol}, {repr(self.children)})"


class TerminalNode(Node):
    def __init__(self, symbol: str, id: int):
        super().__init__(symbol)
        self.id = id

    def __repr__(self):
        return f"TerminalNode({self.quote_symbol()}, {self.id})"

    def __hash__(self):
        return hash((self.symbol, self.id))

    def quote_symbol(self):
        return Node(f"{self.symbol}-{self.id}").quote_symbol()


class GrammarGraph:
    def __init__(self, root):
        self.root = root
        self.__all_nodes = None
        self.__all_edges = None

    def __repr__(self):
        return f"GrammarGraph({repr(self.root)})"

    def bfs(self, action: Callable[[Node], Union[None, bool]], start_node: Union[None, Node] = None):
        if start_node is None:
            start_node = self.root

        visited = [start_node]
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            if action(node):
                return

            if issubclass(type(node), NonterminalNode):
                for child in node.children:
                    if child not in visited:
                        visited.append(child)
                        queue.append(child)

    @property
    def all_nodes(self) -> OrderedSet[Node]:
        if self.__all_nodes is not None:
            return self.__all_nodes

        nodes: OrderedSet[Node] = OrderedSet([])

        def action(node: Node):
            nonlocal nodes
            nodes.add(node)

        self.bfs(action)
        self.__all_nodes = nodes
        return nodes

    @all_nodes.setter
    def all_nodes(self, val: OrderedSet[Node]) -> None:
        self.__all_nodes = val

    @property
    def all_edges(self) -> OrderedSet[Tuple[Node, Node]]:
        if self.__all_edges is not None:
            return self.__all_edges

        result = OrderedSet()
        for node in self.all_nodes:
            if not isinstance(node, NonterminalNode):
                continue
            result.update({(node, child) for child in node.children})

        self.__all_edges = result
        return result

    @all_edges.setter
    def all_edges(self, val: OrderedSet[Tuple[Node, Node]]) -> None:
        self.__all_edges = val

    def shortest_non_trivial_path(self, source: Node, target: Node,
                                  nodes_filter: Optional[Callable[[Node], bool]] =
                                  lambda n: type(n) is NonterminalNode) -> List[Node]:
        if nodes_filter is None:
            def nodes_filter(n):
                return True

        if issubclass(type(source), NonterminalNode):
            source: NonterminalNode

            paths: List[List[Node]] = [self.shortest_path(child, target, nodes_filter) for child in source.children]
            paths = [path for path in paths if path]

            sorted(paths, key=len)
            result: List[Node]
            if not nodes_filter(source):
                result = paths[0]
            else:
                result = [source] + paths[0]

            assert not nodes_filter(target) or result[-1] == target
            return result

        return []

    def shortest_path(self, source: Node, target: Node,
                      nodes_filter: Optional[Callable[[Node], bool]] = lambda n: type(n) is NonterminalNode) \
            -> List[Node]:
        dist, prev = self.dijkstra(source, target)
        s = []
        u = target
        if u == source or prev[u] is not None:
            while u is not None:
                s = [u] + s
                u = None if u == source else prev[u]

        return s if nodes_filter is None else list([n for n in s if nodes_filter(n)])

    def shortest_distances(self, infinity: int = sys.maxsize) -> Dict[Node, Dict[Node, int]]:
        """Implementation of the Floyd-Warshall algorithm for finding shortest distances between all paths"""
        dist: Dict[Node, Dict[Node, int]] = {u: {v: infinity for v in self.all_nodes} for u in self.all_nodes}

        for (u, v) in self.all_edges:
            dist.setdefault(u, {})[v] = 1

        # for v in self.all_nodes:
        #     dist.setdefault(v, {})[v] = 0

        for k in self.all_nodes:
            for i in self.all_nodes:
                for j in self.all_nodes:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        return dist

    def dijkstra(self, source: Node, target: Optional[Node] = None) -> Tuple[
        Dict[Node, int], Dict[Node, Optional[Node]]]:
        """Implementation of Dijkstra's algorithm with Fibonacci heap"""
        nodes = self.all_nodes
        fh_node_map: Dict[Node, fh.Node] = {}
        fh_rev_node_map: Dict[fh.Node, Node] = {}

        dist: Dict[Node, int] = {source: 0}
        prev: Dict[Node, Optional[Node]] = {}

        q: fh.Fheap = fh.makefheap()

        for v in nodes:
            if v != source:
                dist[v] = sys.maxsize
                prev[v] = None

            fh_node = fh.Node(dist[v])
            fh_node_map[v] = fh_node
            fh_rev_node_map[fh_node] = v
            q.insert(fh_node)

        while q.num_nodes:
            u: Node = fh_rev_node_map[q.extract_min()]

            if u == target:
                break

            if issubclass(type(u), NonterminalNode):
                u: NonterminalNode
                for v in u.children:
                    alt = dist[u] + 1
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
                        q.decrease_key(fh_node_map[v], alt)

        return dist, prev

    def to_grammar(self):
        result: Grammar = {}

        def action(node: Node):
            nonlocal result
            if type(node) is TerminalNode or type(node) is ChoiceNode:
                return

            node: NonterminalNode
            choice_node: ChoiceNode
            productions = []
            for choice_node in node.children:
                productions.append("".join([child.symbol for child in choice_node.children]))

            result[node.symbol] = productions

        self.bfs(action)
        return result

    def subgraph(self, nonterminal: Union[NonterminalNode, str]):
        if type(nonterminal) is NonterminalNode:
            start_node = nonterminal
        else:
            start_node = self.get_node(nonterminal)

        if start_node.symbol == "<start>":
            return self

        root_node = NonterminalNode("<start>", [ChoiceNode("<start>-choice-1", [start_node])])
        return GrammarGraph(root_node)

    def parents(self, node: Node) -> List[Node]:
        result = []

        def action(maybe_parent: Node) -> bool:
            if issubclass(type(maybe_parent), NonterminalNode) and \
                    node in cast(NonterminalNode, maybe_parent).children and \
                    maybe_parent not in result:
                result.append(maybe_parent)

            return False

        self.bfs(action)

        return result

    def is_tree(self):
        # We cannot simply perform a BFS and return False if any child has already been
        # seen since we re-use nodes for the same nonterminal (they could be copied, but
        # I'd rather not do this).
        result = True

        def action(node: Node):
            nonlocal result
            if issubclass(type(node), NonterminalNode):
                if node.reachable(node):
                    result = False
                    return True

        self.bfs(action)
        return result

    def get_node(self, nonterminal: str) -> Union[None, NonterminalNode]:
        assert is_nonterminal(nonterminal)

        candidates = [node for node in self.all_nodes
                      if isinstance(node, NonterminalNode)
                      and node.symbol == nonterminal]

        if not candidates:
            return None

        assert len(candidates) == 1
        return candidates[0]

    def filter(self, f: Callable[[Node], bool], abort: Callable[[Node], bool] = lambda n: False) -> List[Node]:
        result: List[Node] = []

        def action(node: Node) -> bool:
            nonlocal result
            if f(node):
                result.append(node)

            return abort(node)

        self.bfs(action)
        return result

    def to_dot(self) -> Digraph:
        def node_attr(dot: Digraph, symbol: str, **attr):
            dot.node(dot_escape(unicode_escape(symbol)), **attr)

        def edge_attr(dot: Digraph, start_node: str, stop_node: str, **attr):
            dot.edge(dot_escape(unicode_escape(start_node)), dot_escape(unicode_escape(stop_node)), **attr)

        def dot_escape(s):
            return f'<{html.escape(s)}>'.replace(":", "&#58;")

        graph = Digraph(comment="GrammarGraph")

        def action(node: Node):
            if type(node) is TerminalNode:
                # graph.node(node.symbol, label=Node(node.symbol).quote_symbol(), shape="box")
                node_attr(graph, node.symbol, shape="box")
            elif type(node) is ChoiceNode:
                # graph.node(node.symbol, shape="diamond")
                node_attr(graph, node.symbol, shape="diamond")
            else:
                # graph.node(node.symbol, shape="circle")
                node_attr(graph, node.symbol, shape="circle")

            if issubclass(type(node), NonterminalNode):
                node: NonterminalNode
                for nr, child in enumerate(node.children):
                    # graph.edge(node.symbol, child.symbol, label=f"<{nr + 1}>")
                    edge_attr(graph, node.symbol, child.symbol, label=f"<{nr + 1}>")

        self.bfs(action)
        return graph

    @staticmethod
    def from_grammar(grammar: Grammar):
        nonterminal_nodes: Dict[str, NonterminalNode] = {}
        terminal_ids: Dict[str, int] = {}

        def recurse(label: str) -> Node:
            nonlocal nonterminal_nodes, terminal_ids

            if not is_nonterminal(label):
                terminal_id = terminal_ids.setdefault(label, 1)
                terminal_ids[label] = terminal_id + 1
                return TerminalNode(label, terminal_id)

            if label in nonterminal_nodes:
                return nonterminal_nodes[label]

            children_nodes = []
            new_node = NonterminalNode(label, children_nodes)
            nonterminal_nodes[label] = new_node

            assert grammar[label], f"Grammar has no rules for {label}"
            for nr, expansion in enumerate(grammar[label]):
                expansion_children_nodes = []
                if len(expansion) == 0:
                    # Expansion is the empty string
                    expansion_children_nodes.append(recurse(expansion))

                expansion_elements = split_expansion(expansion)
                for elem in expansion_elements:
                    if elem == label:
                        expansion_children_nodes.append(new_node)
                    else:
                        expansion_children_nodes.append(recurse(elem))
                children_nodes.append(ChoiceNode(f"{label}-choice-{nr + 1}", expansion_children_nodes))

            return new_node

        return GrammarGraph(recurse("<start>"))
