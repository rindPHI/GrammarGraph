import random
import string
import sys
import unittest
from typing import Dict

from fuzzingbook.Grammars import JSON_GRAMMAR, US_PHONE_GRAMMAR, is_nonterminal, srange
from fuzzingbook.Parser import CSV_GRAMMAR

from grammar_graph.gg import GrammarGraph, Node, NonterminalNode, ChoiceNode


class TestGrammarGraph(unittest.TestCase):

    # def test_todot(self):
    #     graph = GrammarGraph.from_grammar(JSON_GRAMMAR)
    #     print(graph.to_dot())

    def test_reachability_and_filter(self):
        graph = GrammarGraph.from_grammar(JSON_GRAMMAR)

        element_node = graph.get_node("<element>")
        self.assertTrue(element_node.reachable(element_node))

        value_node = graph.get_node("<value>")
        self.assertTrue(value_node.reachable(value_node))
        self.assertTrue(element_node.reachable(value_node))

        int_node = graph.get_node("<int>")
        self.assertTrue(value_node.reachable(int_node))
        self.assertTrue(element_node.reachable(int_node))
        self.assertFalse(int_node.reachable(int_node))

    def test_parents(self):
        graph = GrammarGraph.from_grammar(JSON_GRAMMAR)

        def action(node: Node) -> bool:
            parents = graph.parents(node)
            parent: NonterminalNode
            for parent in parents:
                self.assertIn(node, parent.children)

            if issubclass(type(Node), NonterminalNode):
                node: NonterminalNode
                for child in node.children:
                    self.assertIn(node, graph.parents(child))

            return False

        graph.bfs(action)

    def test_to_grammar(self):
        graph = GrammarGraph.from_grammar(JSON_GRAMMAR)
        self.assertEqual(JSON_GRAMMAR, graph.to_grammar())

    def test_is_tree(self):
        graph = GrammarGraph.from_grammar(JSON_GRAMMAR)
        self.assertFalse(graph.is_tree())
        self.assertTrue(graph.subgraph("<character>").is_tree())
        self.assertTrue(graph.subgraph("<digit>").is_tree())
        self.assertFalse(graph.subgraph("<digits>").is_tree())
        self.assertTrue(graph.subgraph("<sign>").is_tree())
        self.assertFalse(graph.subgraph("<start>").is_tree())

    def test_is_tree_2(self):
        graph = GrammarGraph.from_grammar(US_PHONE_GRAMMAR)
        self.assertTrue(graph.subgraph("<exchange>").is_tree())
        self.assertTrue(graph.subgraph("<start>").is_tree())

    def test_dijkstra(self):
        graph = GrammarGraph.from_grammar(JSON_GRAMMAR)
        value = graph.get_node("<value>")
        member = graph.get_node("<member>")
        path = list(map(lambda node: node.symbol,
                        [node for node in graph.shortest_path(value, member)]))
        self.assertEqual(['<value>', '<object>', '<members>', '<member>'], path)

    def test_floyd_warshall(self):
        graph = GrammarGraph.from_grammar(CSV_GRAMMAR)
        distances = graph.shortest_distances()
        self.assertEqual(4, distances[graph.get_node("<csvline>")][graph.get_node("<item>")])
        self.assertEqual(10, distances[graph.get_node("<start>")][graph.get_node("<letter>")])
        self.assertEqual(sys.maxsize, distances[graph.get_node("<letters>")][graph.get_node("<item>")])
        self.assertEqual(2, distances[graph.get_node("<letters>")][graph.get_node("<letters>")])

    def test_reachability_via_floyd_warshall(self):
        graph = GrammarGraph.from_grammar(CSV_GRAMMAR)
        node_distances = graph.shortest_distances()

        str_node_distances: [Dict[str, Dict[str, int]]] = {
            u.symbol: {
                v.symbol: dist
                for v, dist in node_distances[u].items()
                if not isinstance(v, ChoiceNode)
            }
            for u in graph.all_nodes
            if type(u) is NonterminalNode
        }

        self.assertTrue(all(is_nonterminal(u) for u in str_node_distances))

        for _ in range(500):
            u = random.choice(list(CSV_GRAMMAR.keys()))
            v = random.choice(list(CSV_GRAMMAR.keys()))

            self.assertEqual(
                graph.get_node(u).reachable(graph.get_node(v)),
                str_node_distances[u][v] < sys.maxsize,
                f"Differing result for {u} and {v}"
            )

    def test_nontrivial_path_json(self):
        graph = GrammarGraph.from_grammar(JSON_GRAMMAR)
        members = graph.get_node("<members>")
        path = list(map(lambda node: node.symbol,
                        [node for node in graph.shortest_non_trivial_path(members, members)]))
        self.assertEqual(['<members>', '<symbol-2>', '<symbol>', '<members>'], path)

    def test_nontrivial_path_csv(self):
        graph = GrammarGraph.from_grammar(CSV_GRAMMAR)
        items = graph.get_node("<items>")
        self.assertEqual(['<items>', '<items>'],
                         [node.symbol for node in graph.shortest_non_trivial_path(items, items)])

    def test_tinyc_shortest_path(self):
        graph = GrammarGraph.from_grammar(TINYC_GRAMMAR)
        source = graph.get_node("<term>")
        target = graph.get_node("<expr>")

        sh_nt_path = [node.symbol for node in graph.shortest_non_trivial_path(source, target)]
        self.assertEqual("<term>", sh_nt_path[0])
        self.assertEqual("<expr>", sh_nt_path[-1])

        sh_path = [node.symbol for node in graph.shortest_path(source, target)]
        self.assertEqual("<term>", sh_path[0])
        self.assertEqual("<expr>", sh_path[-1])


TINYC_GRAMMAR = {
    "<start>": ["<mwss><statement><mwss>"],
    "<statement>": [
        "if<mwss><paren_expr><mwss><statement>",
        "if<mwss><paren_expr><mwss><statement><mwss>else<wss><statement>",
        "while<mwss><paren_expr><mwss><statement>",
        "do<wss><statement>while<mwss><paren_expr><mwss>;",
        "{<mwss><statements><mwss>}",
        "<mwss><expr><mwss>;",
        ";"
    ],
    "<statements>": ["", "<statement>", "<statement><mwss><statements>"],
    "<paren_expr>": ["(<mwss><expr><mwss>)"],
    "<expr>": [
        "<test>",
        "<id><mwss>=<mwss><expr>"
    ],
    "<test>": [
        "<sum>",
        "<sum><mwss><<mwss><sum>"
    ],
    "<sum>": [
        "<term>",
        "<sum><mwss>+<mwss><term>",
        "<sum><mwss>-<mwss><term>"
    ],
    "<term>": [
        "<id>",
        "<int>",
        "<paren_expr>"
    ],
    "<id>": srange(string.ascii_lowercase),
    "<int>": [
        "<digit>",
        "<digit_nonzero><digits>"
    ],
    "<digits>": [
        "<digit>",
        "<digit><int>"
    ],
    "<digit>": srange(string.digits),
    "<digit_nonzero>": list(set(srange(string.digits)) - {"0"}),
    "<mwss>": ["", "<wss>"],
    "<wss>": ["<ws>", "<ws><wss>"],
    "<ws>": srange(" \n\t"),
}
if __name__ == '__main__':
    unittest.main()
