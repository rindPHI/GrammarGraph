import html
import json
import re
import sys
from functools import lru_cache
from typing import List, Dict, Callable, Union, Optional, Tuple, cast, Set

import fibheap as fh
from graphviz import Digraph

NonterminalType = str
Grammar = Dict[NonterminalType, List[str]]
ParseTree = Tuple[str, Optional[List['ParseTree']]]

RE_NONTERMINAL = re.compile(r'(<[^<> ]*>)')


@lru_cache(maxsize=None)
def is_nonterminal(s):
    return RE_NONTERMINAL.match(s)


def split_expansion(expansion: str) -> List[str]:
    return [token for token in re.split(RE_NONTERMINAL, expansion) if token]


def unicode_escape(s: str, error: str = 'backslashreplace') -> str:
    def ascii_chr(byte: int) -> str:
        if 0 <= byte <= 127:
            return chr(byte)
        return r"\x%02x" % byte

    bytes_ = s.encode('utf-8', error)
    return "".join(map(ascii_chr, bytes_))


class Node:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.__hash = None

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash(self.symbol)
        return self.__hash

    def __eq__(self, other):
        return type(other) == type(self) and self.symbol == other.symbol

    def to_json(self):
        raise NotImplemented()

    def __lt__(self, other):
        # Needed for fibheap
        return self.symbol < other.symbol

    def quote_symbol(self):
        return '"' + self.symbol.translate(str.maketrans({'"': r"\""})) + '"'


class NonterminalNode(Node):
    # NOTE: We do not override __eq__ and __hash__, since in each grammar graph, there
    #       should be only one nonterminal node for a given nonterminal symbol. Thus,
    #       including children in comparisons is not necessary, so we abstain from that
    #       for performance reasons.
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

    def __eq__(self, other):
        return isinstance(other, TerminalNode) and self.id == other.id and super().__eq__(other)

    def quote_symbol(self):
        return Node(f"{self.symbol}-{self.id}").quote_symbol()


class GrammarGraph:
    def __init__(self, root: Node, grammar: Optional[Grammar] = None):
        assert isinstance(root, Node)
        self.root = root
        self._grammar = grammar
        self.__all_nodes = None
        self.__all_edges = None
        self.__hash = None

    @property
    def grammar(self):
        if self._grammar is None:
            self._grammar = self._compute_grammar()

        return self._grammar

    @grammar.setter
    def grammar(self, grammar: Grammar):
        raise NotImplementedError()

    def __repr__(self):
        return f"GrammarGraph({repr(self.root)})"

    def __eq__(self, other):
        return isinstance(other, GrammarGraph) and self.grammar == other.grammar

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
    def all_nodes(self) -> Set[Node]:
        if self.__all_nodes is not None:
            return self.__all_nodes

        nodes: Set[Node] = set([])

        def action(node: Node):
            nonlocal nodes
            nodes.add(node)

        self.bfs(action)
        self.__all_nodes = nodes
        return nodes

    @all_nodes.setter
    def all_nodes(self, val: Set[Node]) -> None:
        self.__all_nodes = val

    @property
    def all_edges(self) -> Set[Tuple[Node, Node]]:
        if self.__all_edges is not None:
            return self.__all_edges

        result = set([])
        for node in self.all_nodes:
            if not isinstance(node, NonterminalNode):
                continue
            result.update({(node, child) for child in node.children})

        self.__all_edges = result
        return result

    @all_edges.setter
    def all_edges(self, val: Set[Tuple[Node, Node]]) -> None:
        self.__all_edges = val

    @lru_cache
    def reachable(self, from_node: Union[str, Node], to_node: Union[str, Node]) -> bool:
        # Note: Reachability is not reflexive!
        def node_in_children(node: Node) -> bool:
            return isinstance(node, NonterminalNode) and to_node in node.children

        if isinstance(from_node, str):
            from_node = self.get_node(from_node)
        if isinstance(to_node, str):
            to_node = self.get_node(to_node)

        sources = self.filter(node_in_children, node_in_children, from_node=from_node)
        return len(sources) > 0

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

    @lru_cache(maxsize=None)
    def __shortest_path(self, source: Node, target: Node) -> List[Node]:
        dist, prev = self.dijkstra(source, target)
        s = []
        u = target
        if u == source or prev[u] is not None:
            while u is not None:
                s = [u] + s
                u = None if u == source else prev[u]

        return s

    def shortest_path(self, source: Node, target: Node,
                      nodes_filter: Optional[Callable[[Node], bool]] = lambda n: type(n) is NonterminalNode) \
            -> List[Node]:
        result = self.__shortest_path(source, target)
        return result if nodes_filter is None else list([n for n in result if nodes_filter(n)])

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
        """Deprecated; use the `grammar` property."""
        return self.grammar

    def _compute_grammar(self):
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
                if self.reachable(node, node):
                    result = False
                    return True

        self.bfs(action)
        return result

    @lru_cache(maxsize=None)
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

    def filter(
            self,
            f: Callable[[Node], bool],
            abort: Callable[[Node], bool] = lambda n: False,
            from_node: Optional[Node] = None) -> List[Node]:
        result: List[Node] = []

        def action(node: Node) -> bool:
            nonlocal result
            if f(node):
                result.append(node)

            return abort(node)

        self.bfs(action, start_node=from_node)
        return result

    @lru_cache(maxsize=None)
    def nonterminal_kpaths(
            self, node: Union[NonterminalNode, str], k: int, up_to: bool = False) -> Set[Tuple[Node, ...]]:
        if isinstance(node, str):
            assert is_nonterminal(node)
            node = self.get_node(node)

        return self.k_paths(k, up_to=up_to, start_node=node)

    def tree_is_valid(self, tree: ParseTree) -> bool:
        try:
            self.graph_paths_from_tree(tree)
            return True
        except RuntimeError:
            return False

    def graph_paths_from_tree(self, tree: ParseTree) -> Set[Tuple[Optional[Node], ...]]:
        # We first compute a list, and then an ordered set with unique elements. This is
        # *much* more performant!

        def helper(tree_: ParseTree) -> List[Tuple[Optional[Node], ...]]:
            node, children = tree_
            assert is_nonterminal(node), "Terminal nodes are ambiguous, have to be obtained from parents"

            g_node = self.get_node(node)

            if children is None:
                return [(g_node, None)]

            # Find suitable choice node
            choice_node = self.find_choice_node_for_children(g_node, [child[0] for child in children])

            result: List[Tuple[Node, ...]] = []

            for child_idx, child in enumerate(choice_node.children):
                # Terminal children
                if not is_nonterminal(child.symbol):
                    result.append((g_node, choice_node, child))
                    continue

                result.extend([(g_node, choice_node) + path for path in helper(children[child_idx])])

            return result

        return set(helper(tree))

    def find_choice_node_for_children(self, parent_node: Union[str, NonterminalNode], child_symbols: List[str]):
        if isinstance(parent_node, str):
            parent_node = self.get_node(parent_node)

        choice_nodes: List[ChoiceNode] = cast(List[ChoiceNode], parent_node.children)

        if not child_symbols:
            # Epsilon transitions
            try:
                return next(
                    choice_node for choice_node in choice_nodes
                    if len(choice_node.children) == 1 and not choice_node.children[0].symbol
                )
            except StopIteration:
                raise SyntaxError(f"Could not find a choice node for epsilon transition from {parent_node.symbol}.")

        # Note: Sometimes, the parser seems to "skip" a nonterminal with a possible epsilon-production,
        # as in "<maybe_nuls>: <epsilon> | <nuls>", for which only "<nuls>" appears in the tree.
        # For this reason, the reachability alternative below was added.
        matching_choice_nodes: List[ChoiceNode] = [
            choice_node for choice_node in choice_nodes
            if (len(choice_node.children) == len(child_symbols) and
                all(choice_node.children[idx].symbol == c or
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
                    for idx, c in enumerate(child_symbols)))
        ]

        if len(matching_choice_nodes) != 1:
            raise RuntimeError(f"Child symbols {child_symbols} seem to be incorrect for parent {parent_node.symbol}")

        return matching_choice_nodes[0]

    def k_paths_in_tree(self, tree: ParseTree, k: int) -> Set[Tuple[Node, ...]]:
        assert k > 0
        orig_k = k
        k += k - 1  # Each path of k terminal/nonterminal nodes includes k-1 choice nodes

        # For open trees: Extend all paths ending with None with the possible k-paths for the last nonterminal.
        all_paths = self.graph_paths_from_tree(tree)

        concrete_k_paths: List[Tuple[Node, ...]] = [
            kpath
            for path in all_paths
            for kpath in [
                path[i:i + k]
                for i in range(0, len(path) - k + 1, 1)
                if path[i + k - 1] is not None]
            if (len(kpath) == k and
                not isinstance(kpath[0], ChoiceNode) and
                not isinstance(kpath[-1], ChoiceNode))
        ]

        assert all(p[-1] is not None for p in concrete_k_paths)
        assert all(any(p[-1] == kpath[-1] for kpath in concrete_k_paths) for p in all_paths if p[-1] and len(p) >= k)

        # For open trees: Extend all paths ending with None with the possible k-paths for the last nonterminal.
        potential_k_paths: List[Tuple[Node, ...]] = []

        for prefix in [p[-(k + 1):-1] for p in all_paths if p[-1] is None]:
            assert prefix
            nonterminal_kpaths = self.nonterminal_kpaths(prefix[-1], orig_k, up_to=True)
            potential_k_paths.extend([p for p in nonterminal_kpaths if len(p) == k])
            for postfix in [p for p in nonterminal_kpaths if p[0] == prefix[-1]]:
                path = prefix[:-1] + postfix
                potential_k_paths.extend([path[i:i + k] for i in range(0, len(path) - k + 1, 2)])

        potential_k_paths_set = set(potential_k_paths)
        assert not potential_k_paths_set or potential_k_paths_set.intersection(self.k_paths(orig_k))

        return set(concrete_k_paths) | potential_k_paths_set

    @lru_cache(maxsize=None)
    def k_paths(self, k: int, up_to: bool = False, start_node: Optional[Node] = None) -> set[Tuple[Node, ...]]:
        assert k > 0
        k += k - 1  # Each path of k terminal/nonterminal nodes includes k-1 choice nodes
        result: set[Tuple[Node, ...]] = set([])

        if not start_node:
            all_nodes = self.all_nodes
        else:
            all_nodes = [n for n, dist in self.dijkstra(start_node)[0].items() if dist < sys.maxsize]

        for node in all_nodes:
            node_result: List[Tuple[Node, ...]] = [(node,)]
            for _ in range(k - 1):
                new_node_result: List[Tuple[Node, ...]] = []
                path: Tuple[Node, ...]
                for path in node_result:
                    last_node = path[-1]
                    if up_to:
                        new_node_result.append(path)
                    if isinstance(last_node, TerminalNode):
                        continue

                    new_node_result.extend([path + (child,) for child in cast(NonterminalNode, last_node).children])

                node_result = new_node_result

            result.update(node_result)

        return {
            kpath for kpath in result
            if ((len(kpath) <= k if up_to else len(kpath) == k) and
                not isinstance(kpath[0], ChoiceNode) and
                not isinstance(kpath[-1], ChoiceNode))}

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

        return GrammarGraph(recurse("<start>"), grammar=grammar)
