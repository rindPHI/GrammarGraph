import copy
import html
import json
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
ParseTree = Tuple[str, Optional[List['ParseTree']]]


def split_expansion(expansion: str) -> List[str]:
    return [token for token in re.split(RE_NONTERMINAL, expansion) if token]


class Node:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other):
        return type(other) == type(self) and self.symbol == other.symbol

    def to_json(self):
        raise NotImplemented()

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
        self.__hash = None

    def __repr__(self):
        return f"NonterminalNode({self.quote_symbol()}, {repr(self.children)})"

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash((self.symbol, tuple([child.symbol for child in self.children])))
        return self.__hash

    def __eq__(self, other):
        return (isinstance(other, NonterminalNode) and
                self.symbol == other.symbol and
                [child.symbol for child in self.children] == [child.symbol for child in other.children])


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

    def __eq__(self, other):
        return isinstance(other, TerminalNode) and self.id == other.id and super().__eq__(other)

    def quote_symbol(self):
        return Node(f"{self.symbol}-{self.id}").quote_symbol()


class GrammarGraph:
    def __init__(self, root):
        self.root = root
        self.__all_nodes = None
        self.__all_edges = None
        self.__hash = None

    def __repr__(self):
        return f"GrammarGraph({repr(self.root)})"

    def __eq__(self, other):
        return isinstance(other, GrammarGraph) and self.to_grammar() == other.to_grammar()

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash(json.dumps(self.to_grammar()))
        return self.__hash

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
        """
        This method is limited to nonterminals since for terminal nodes, there might be multiple
        results with different IDs. Use `filter` instead.

        :param nonterminal: The nonterminal for which to get a node.
        :return: The corresponding `NonterminalNode` of none if none found.
        """
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

    @lru_cache(maxsize=None)
    def nonterminal_kpaths(self, node: Union[NonterminalNode, str], k: int) -> List[Tuple[Node, ...]]:
        if isinstance(node, str):
            assert is_nonterminal(node)
            node = self.get_node(node)

        subgraph = self.subgraph(node)
        return [p for p in subgraph.k_paths(k) if p[0] != subgraph.root]

    def graph_paths_from_tree(self, tree: ParseTree) -> OrderedSet[Tuple[Optional[Node], ...]]:
        node, children = tree
        assert is_nonterminal(node), "Terminal nodes are ambiguous, have to be obtained from parents"

        g_node = self.get_node(node)

        if children is None:
            return OrderedSet([(g_node, None)])

        # Find suitable choice node
        choice_nodes: List[ChoiceNode] = cast(List[ChoiceNode], g_node.children)

        if not children:
            # Epsilon transitions
            choice_node: ChoiceNode = next(
                choice_node for choice_node in choice_nodes
                if len(choice_node.children) == 1 and not choice_node.children[0].symbol
            )
        else:
            tree_children_symbols = [child[0] for child in children]

            # Note: Sometimes, the parser seems to "skip" a nonterminal with a possible epsilon-production,
            # as in "<maybe_nuls>: <epsilon> | <nuls>", for which only "<nuls>" appears in the tree.
            # For this reason, the reachability alternative below was added.
            matching_choice_nodes: List[ChoiceNode] = [
                choice_node for choice_node in choice_nodes
                if len(choice_node.children) == len(tree_children_symbols) and
                   all(
                       choice_node.children[idx].symbol == c or
                       # choice_node has a child `choice_node_child` (e.g., "<maybe_nuls>"), which in turn has a
                       # (choice node) child `choice_node_grandchild` (e.g., the non-epsilon production),
                       # which has c (e.g., "<nuls>") as only alternative
                       (choice_node_child := choice_node.children[idx],
                        isinstance(choice_node_child, NonterminalNode) and
                        any(
                            isinstance(choice_node_grandchild, ChoiceNode) and
                            len(choice_node_grandchild.children) == 1 and
                            choice_node_grandchild.children[0].symbol == c
                            for choice_node_grandchild in choice_node_child.children))[-1]
                       for idx, c in enumerate(tree_children_symbols))
            ]
            assert len(matching_choice_nodes) == 1

            choice_node: ChoiceNode = matching_choice_nodes[0]

        result: OrderedSet[Tuple[Node, ...]] = OrderedSet([])

        for child_idx, child in enumerate(children):
            # Nonterminal children
            if not is_nonterminal(child[0]):
                result.add((g_node, choice_node, choice_node.children[child_idx]))
                continue

            result.update([(g_node, choice_node) + path for path in self.graph_paths_from_tree(child)])

        return result

    def k_paths_in_tree(self, tree: ParseTree, k: int) -> OrderedSet[Tuple[Node, ...]]:
        assert k > 0
        orig_k = k
        k += k - 1  # Each path of k terminal/nonterminal nodes includes k-1 choice nodes

        # For open trees: Extend all paths ending with None with the possible k-paths for the last nonterminal.
        all_paths = OrderedSet([path for l in [
            [path] if path[-1] is not None
            else [path[:-2] + possible_kpath
                  for possible_kpath in
                  self.nonterminal_kpaths(path[-2], orig_k)]
            for path in self.graph_paths_from_tree(tree)
        ] for path in l])

        kpath: Tuple[Node, ...]
        result = OrderedSet([
            kpath
            for path in all_paths
            for kpath in [path[i:i + k] for i in range(0, len(path), 1)]
            if (len(kpath) == k and
                not isinstance(kpath[0], ChoiceNode) and
                not isinstance(kpath[-1], ChoiceNode))
        ]).intersection(self.k_paths(orig_k))

        # def path_to_string(p) -> str:
        #     return " ".join([n.symbol for n in p])
        #
        # collisions = [
        #     path for path in result
        #     if len([p for p in result if path_to_string(p) == path_to_string(path)]) > 1]
        # assert not collisions

        return result

    @lru_cache(maxsize=None)
    def k_paths(self, k: int) -> List[Tuple[Node, ...]]:
        assert k > 0
        k += k - 1  # Each path of k terminal/nonterminal nodes includes k-1 choice nodes
        result: List[Tuple[Node, ...]] = []

        for node in self.all_nodes:
            if not isinstance(node, NonterminalNode):
                continue

            node_result: List[Tuple[Node, ...]] = [(node,)]
            for _ in range(k - 1):
                new_node_result: List[Tuple[Node, ...]] = []
                path: Tuple[Node, ...]
                for path in node_result:
                    last_node = path[-1]
                    if isinstance(last_node, TerminalNode):
                        continue

                    new_node_result.extend([path + (child,) for child in cast(NonterminalNode, last_node).children])

                node_result = new_node_result

            result.extend(node_result)

        return [
            kpath for kpath in result
            if (len(kpath) == k and
                not isinstance(kpath[0], ChoiceNode) and
                not isinstance(kpath[-1], ChoiceNode))]

    def k_path_coverage(self, tree: ParseTree, k: int) -> float:
        return len(self.k_paths_in_tree(tree, k)) / len(self.k_paths(k))

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
