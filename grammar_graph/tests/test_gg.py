import unittest

from fuzzingbook.Grammars import JSON_GRAMMAR, US_PHONE_GRAMMAR

from grammar_graph.gg import GrammarGraph, Node, NonterminalNode


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
        path = list(map(lambda node: node.symbol,
                        [node for node in graph.shortest_path(graph.get_node("<value>"), graph.get_node("<member>"))]))
        self.assertEqual(['<value>', '<object>', '<members>', '<member>'], path)


if __name__ == '__main__':
    unittest.main()
