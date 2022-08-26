import copy
import logging
import random
import string
import sys
import unittest
from typing import Dict

from fuzzingbook.GrammarCoverageFuzzer import GrammarCoverageFuzzer
from fuzzingbook.Grammars import JSON_GRAMMAR, US_PHONE_GRAMMAR, is_nonterminal, srange
from fuzzingbook.Parser import CSV_GRAMMAR, EarleyParser

from grammar_graph.gg import GrammarGraph, Node, NonterminalNode, ChoiceNode, TerminalNode, path_to_string


def path_to_string_no_choice(p) -> str:
    return path_to_string(p, False)


class TestGrammarGraph(unittest.TestCase):

    # def test_todot(self):
    #     graph = GrammarGraph.from_grammar(JSON_GRAMMAR)
    #     print(graph.to_dot())

    def test_reachability_and_filter(self):
        graph = GrammarGraph.from_grammar(JSON_GRAMMAR)

        element_node = graph.get_node("<element>")
        self.assertTrue(graph.reachable(element_node, element_node))

        value_node = graph.get_node("<value>")
        self.assertTrue(graph.reachable(value_node, value_node))
        self.assertTrue(graph.reachable(element_node, value_node))

        int_node = graph.get_node("<int>")
        self.assertTrue(graph.reachable(value_node, int_node))
        self.assertTrue(graph.reachable(element_node, int_node))
        self.assertFalse(graph.reachable(int_node, int_node))

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

    def test_get_terminal_node(self):
        graph = GrammarGraph.from_grammar(JSON_GRAMMAR)
        try:
            graph.get_node("1")
            self.fail()
        except AssertionError:
            pass

        one_nodes = graph.filter(lambda n: n.symbol == "1")
        self.assertTrue(all(isinstance(one_node, TerminalNode) for one_node in one_nodes))
        self.assertEqual(2, len(one_nodes))

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
                graph.reachable(graph.get_node(u), graph.get_node(v)),
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
        graph = GrammarGraph.from_grammar(SCRIPTSIZE_C_GRAMMAR)
        source = graph.get_node("<term>")
        target = graph.get_node("<expr>")

        sh_nt_path = [node.symbol for node in graph.shortest_non_trivial_path(source, target)]
        self.assertEqual("<term>", sh_nt_path[0])
        self.assertEqual("<expr>", sh_nt_path[-1])

        sh_path = [node.symbol for node in graph.shortest_path(source, target)]
        self.assertEqual("<term>", sh_path[0])
        self.assertEqual("<expr>", sh_path[-1])

    def test_grammar_k_paths(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        str_paths = [", ".join([n.symbol for n in p if not isinstance(n, ChoiceNode)]) for p in graph.k_paths(3)]
        self.assertEqual(85, len(str_paths))

    def test_graph_paths_from_tree_no_terminals(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        # x + y * z
        tree = (
            '<start>', [
                ('<add_expr>', [
                    ('<add_expr>', [
                        ('<mult_expr>', [('<unary_expr>', [('<id>', [('x', [])])])])
                    ]),
                    (' ', []),
                    ('<add_symbol>', [('+', [])]),
                    (' ', []),
                    ('<mult_expr>', [
                        ('<mult_expr>', [('<unary_expr>', [('<id>', [('x', [])])])]),
                        (' ', []),
                        ('<mult_symbol>', [('*', [])]),
                        (' ', []),
                        ('<unary_expr>', [('<id>', [('z', [])])]),
                    ])])])

        str_results = set(map(path_to_string_no_choice, graph.graph_paths_from_tree(tree, include_terminals=False)))

        self.assertEqual({
            '<start> <add_expr> <add_symbol>',
            '<start> <add_expr> <mult_expr> <unary_expr> <id>',
            '<start> <add_expr> <mult_expr> <mult_symbol>',
            '<start> <add_expr> <add_expr> <mult_expr> <unary_expr> <id>',
            '<start> <add_expr> <mult_expr> <mult_expr> <unary_expr> <id>'
        }, str_results)

    def test_graph_k_paths_from_tree_no_terminals(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        # x + y * z
        tree = (
            '<start>', [
                ('<add_expr>', [
                    ('<add_expr>', [
                        ('<mult_expr>', [('<unary_expr>', [('<id>', [('x', [])])])])
                    ]),
                    (' ', []),
                    ('<add_symbol>', [('+', [])]),
                    (' ', []),
                    ('<mult_expr>', [
                        ('<mult_expr>', [('<unary_expr>', [('<id>', [('x', [])])])]),
                        (' ', []),
                        ('<mult_symbol>', [('*', [])]),
                        (' ', []),
                        ('<unary_expr>', [('<id>', [('z', [])])]),
                    ])])])

        str_results = set(map(path_to_string_no_choice, graph.k_paths_in_tree(
            tree, 3, include_terminals=False, include_potential_paths=False)))

        self.assertEqual({
            '<start> <add_expr> <add_symbol>',
            '<start> <add_expr> <mult_expr>',
            '<start> <add_expr> <mult_expr>',
            '<start> <add_expr> <add_expr>',
            '<start> <add_expr> <mult_expr>',
            '<add_expr> <add_expr> <mult_expr>',
            '<add_expr> <mult_expr> <unary_expr>',
            '<add_expr> <mult_expr> <mult_expr>',
            '<add_expr> <mult_expr> <mult_symbol>',
            '<add_expr> <mult_expr> <unary_expr>',
            '<mult_expr> <mult_expr> <unary_expr>',
            '<mult_expr> <unary_expr> <id>',
        }, str_results)

    def test_graph_paths_trivial_tree(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        tree = ('<start>', None)
        self.assertEqual({(graph.get_node('<start>'),)}, graph.graph_paths_from_tree(tree, include_terminals=False))

    def test_graph_paths_open_tree_no_terminals(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        tree = (
            '<start>', [
                ('<add_expr>', [
                    ('<add_expr>', [
                        ('<mult_expr>', None)
                    ]),
                    (' ', []),
                    ('<add_symbol>', [('+', [])]),
                    (' ', []),
                    ('<mult_expr>', None)])])

        str_results = set(map(path_to_string_no_choice, graph.graph_paths_from_tree(tree, include_terminals=False)))

        self.assertEqual({
            '<start> <add_expr> <add_expr> <mult_expr>',
            '<start> <add_expr> <add_symbol>',
            '<start> <add_expr> <mult_expr>',
        }, str_results)

    def test_graph_k_paths_open_tree_no_terminals(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        tree = (
            '<start>', [
                ('<add_expr>', [
                    ('<add_expr>', [
                        ('<mult_expr>', None)
                    ]),
                    (' ', []),
                    ('<add_symbol>', [('+', [])]),
                    (' ', []),
                    ('<mult_expr>', None)])])

        str_results = set(map(path_to_string_no_choice, graph.k_paths_in_tree(
            tree, 3, include_terminals=False, include_potential_paths=False)))

        self.assertEqual({
            '<start> <add_expr> <add_expr>',
            '<add_expr> <add_expr> <mult_expr>',
            '<start> <add_expr> <add_symbol>',
            '<start> <add_expr> <mult_expr>',
        }, str_results)

    def test_grammar_nonterminal_k_paths(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        str_paths = list(map(lambda p: path_to_string(p, False), graph.k_paths(2, include_terminals=False)))
        self.assertIn('<mult_expr> <mult_expr>', str_paths)
        self.assertEqual(17, len(str_paths))

    def test_simple_grammar_2_paths(self):
        grammar = {
            "<start>": ["<AB>", "<BA>"],
            "<AB>": ["<A><B>"],
            "<BA>": ["<B><A>"],
            "<A>": ["a"],
            "<B>": ["b"]
        }
        graph = GrammarGraph.from_grammar(grammar)
        paths = graph.k_paths(2)

        str_paths = [", ".join([n.symbol for n in p if not isinstance(n, ChoiceNode)]) for p in paths]
        self.assertEqual(8, len(str_paths))

    def test_grammar_k_paths_up_to(self):
        grammar = {
            "<start>": ["<AB>", "<BA>"],
            "<AB>": ["<A><B>"],
            "<BA>": ["<B><A>"],
            "<A>": ["a"],
            "<B>": ["b"]
        }
        graph = GrammarGraph.from_grammar(grammar)
        paths = graph.k_paths(4, up_to=True)

        str_paths = [", ".join([n.symbol for n in p if not isinstance(n, ChoiceNode)]) for p in paths]

        self.assertEqual({
            '<start>, <AB>, <A>, a',
            '<start>, <BA>, <A>, a',
            '<start>, <AB>, <B>, b',
            '<start>, <BA>, <B>, b',
            '<start>, <AB>, <A>',
            '<start>, <AB>, <B>',
            '<start>, <BA>, <A>',
            '<start>, <BA>, <B>',
            '<AB>, <A>, a',
            '<AB>, <B>, b',
            '<BA>, <A>, a',
            '<BA>, <B>, b',
            '<start>, <AB>',
            '<start>, <BA>',
            '<AB>, <A>',
            '<AB>, <B>',
            '<BA>, <A>',
            '<BA>, <B>',
            '<A>, a',
            '<B>, b',
            '<start>',
            '<A>',
            '<AB>',
            '<B>',
            '<BA>',
            'a',
            'b'}, set(str_paths))

    def test_k_path_coverage(self):
        parser = EarleyParser(EXPR_GRAMMAR)
        tree = list(parser.parse("x + 42"))[0]
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        self.assertEqual(23, int(graph.k_path_coverage(tree, 3) * 100))  # 23% coverage

    def test_nonterminal_kpaths(self):
        logging.basicConfig(level=logging.INFO)
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        for nonterminal in EXPR_GRAMMAR:
            for k in range(1, 5):
                self.assertEqual(
                    set(graph.nonterminal_kpaths(nonterminal, k)),
                    set(graph.k_paths_in_tree((nonterminal, None), k)),
                    f"{k}-paths differ for nonterminal {nonterminal}"
                )

    def test_k_paths_in_tree_no_terminals(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        tree = ("<start>", [("<add_expr>", [("<mult_expr>", [("<unary_expr>", None)])])])

        concrete_paths = graph.k_paths_in_tree(tree, 3, include_potential_paths=False)
        self.assertEqual(2, len(concrete_paths))

        nonterminal_ending_paths = graph.k_paths_in_tree(tree, 3, include_terminals=False)
        nonterminal_ending_string_paths = list(map(lambda p: path_to_string(p, False), nonterminal_ending_paths))

        self.assertIn('<mult_expr> <mult_expr> <unary_expr>', nonterminal_ending_string_paths)
        self.assertIn('<mult_expr> <unary_expr> <dec_digits>', nonterminal_ending_string_paths)
        self.assertNotIn('<mult_expr> <unary_expr> ")" (1)', nonterminal_ending_string_paths)

        self.assertEqual(41, len(nonterminal_ending_paths))

        self.assertTrue(all(type(path[0]) is NonterminalNode for path in nonterminal_ending_paths))
        self.assertTrue(all(type(path[-1]) is NonterminalNode for path in nonterminal_ending_paths))

        potential_paths = graph.k_paths_in_tree(tree, 3)
        self.assertEqual(80, len(potential_paths))

        self.assertTrue(all(type(path[0]) is NonterminalNode for path in potential_paths))
        self.assertTrue(all(type(path[-1]) is not ChoiceNode for path in potential_paths))

    def test_k_paths_in_tree_with_terminals(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        tree = ("<start>", [("<add_expr>", [("<mult_expr>", [("<unary_expr>", None)])])])
        five_paths = graph.k_paths_in_tree(tree, 5)
        str_paths = set(map(path_to_string_no_choice, five_paths))

        self.assertIn('<start> <add_expr> <mult_expr> <unary_expr> ")" (1)', str_paths)
        self.assertIn('<start> <add_expr> <mult_expr> <unary_expr> <add_expr>', str_paths)
        self.assertIn('<start> <add_expr> <mult_expr> <unary_expr> "(" (1)', str_paths)

    def test_k_path_coverage_open_tree(self):
        graph = GrammarGraph.from_grammar(EXPR_GRAMMAR)
        tree = ("<start>", [("<add_expr>", [("<mult_expr>", [("<unary_expr>", None)])])])

        for i in range(1, 8):
            orig_k_paths = graph.k_paths_in_tree(tree, i)
            self.assertTrue(orig_k_paths)

            fuzzer = GrammarCoverageFuzzer(EXPR_GRAMMAR)
            for _ in range(100):
                complete_tree = fuzzer.expand_tree(copy.deepcopy(tree))
                complete_k_paths = graph.k_paths_in_tree(complete_tree, i)
                self.assertLessEqual(len(complete_k_paths), len(orig_k_paths))
                self.assertTrue(complete_k_paths.issubset(orig_k_paths))

    def test_potential_paths_without_terminals(self):
        lang_grammar = {
            "<start>":
                ["<stmt>"],
            "<stmt>":
                ["<assgn> ; <stmt>", "<assgn>"],
            "<assgn>":
                ["<var> := <rhs>"],
            "<rhs>":
                ["<var>", "<digit>"],
            "<var>": list(string.ascii_lowercase),
            "<digit>": list(string.digits)
        }
        graph = GrammarGraph.from_grammar(lang_grammar)

        tree = ('<start>', [('<stmt>', [('<assgn>', [('<var>', None), (' := ', []), ('<rhs>', None)])])])

        paths = graph.k_paths_in_tree(tree, 3, include_potential_paths=True, include_terminals=False)

        expected = {
            '<start> <stmt> <assgn>',
            '<stmt> <assgn> <rhs>',
            '<assgn> <rhs> <var>',
            '<assgn> <rhs> <digit>',
            '<stmt> <assgn> <var>'}

        self.assertEqual(expected, set(map(path_to_string_no_choice, paths)))

    def test_scriptsize_c_two_coverage(self):
        tree = ("<start>", [("<statement>", [("<declaration>", None)])])
        graph = GrammarGraph.from_grammar(SCRIPTSIZE_C_GRAMMAR)

        self.assertLess(graph.k_path_coverage(tree, 2), 1)

    def test_scriptsize_g_nested_block(self):
        graph = GrammarGraph.from_grammar(SCRIPTSIZE_C_GRAMMAR)
        inp = "{{x;}}"
        tree = list(EarleyParser(SCRIPTSIZE_C_GRAMMAR).parse(inp))[0]

        all_paths = [path_to_string((n for n in p if not isinstance(n, ChoiceNode)))
                     for p in graph.graph_paths_from_tree(tree)]

        self.assertIn('<start> <statement> <block> <statements> <statements> "" (1)', all_paths)

        three_paths = [path_to_string(p) for p in graph.k_paths_in_tree(tree, 3)]
        self.assertIn("<block> <block>-choice-1 <statements> <statements>-choice-1 <statements>", three_paths)

    def test_scriptsize_c_k_coverage(self):
        graph = GrammarGraph.from_grammar(SCRIPTSIZE_C_GRAMMAR)
        tree = ('<start>', [('<statement>', [('if', []), ('<paren_expr>', None), (' ', []), ('<statement>', None)])])
        str_paths = [path_to_string(p) for p in graph.k_paths_in_tree(tree, 2)]
        print("\n".join(str_paths))

    def test_epsilon_production_paths(self):
        grammar = {
            "<start>": ["<As>"],
            "<As>": ["", "<A><As>"],
            "<A>": ["a"]
        }
        graph = GrammarGraph.from_grammar(grammar)
        inp = "aa"
        tree = next(EarleyParser(grammar).parse(inp))

        all_paths = [path_to_string(p) for p in graph.graph_paths_from_tree(tree)]
        self.assertIn(
            '<start> <start>-choice-1 <As> <As>-choice-2 <As> <As>-choice-2 <As> <As>-choice-1 "" (1)',
            all_paths)

        tree_paths = {path_to_string(p) for p in graph.k_paths_in_tree(tree, 1)}
        self.assertIn('"" (1)', tree_paths)

        grammar_paths = {path_to_string(p) for p in graph.k_paths(1)}
        self.assertIn('"" (1)', tree_paths)

        self.assertEqual(len(grammar_paths), len(tree_paths))
        self.assertEqual(grammar_paths, tree_paths)

    def test_graph_k_paths_from_tree_no_terminals_2(self):
        grammar = {
            "<start>": ["<expr>"],
            "<expr>": [
                "<id> = <id>;",
                "nop"
            ],
            "<id>": srange(string.ascii_lowercase),
        }
        tree = ('<start>', [('<expr>', [('nop', ())])])
        graph = GrammarGraph.from_grammar(grammar)
        self.assertFalse(graph.k_paths_in_tree(tree, 3, include_terminals=False))


EXPR_GRAMMAR = {
    "<start>": ["<add_expr>"],
    "<add_expr>": ["<mult_expr>", "<add_expr> <add_symbol> <mult_expr>"],
    "<add_symbol>": ["+", "-"],
    "<mult_expr>": ["<unary_expr>", "<mult_expr> <mult_symbol> <unary_expr>"],
    "<mult_symbol>": ["*", "/", "%"],
    "<unary_expr>": ["<id>", "<unary_symbol><unary_expr>", "(<add_expr>)", "<dec_digits>"],
    "<unary_symbol>": ["+", "-", "++", "--"],
    "<dec_digits>": ["<dec_digit><maybe_dec_digits>"],
    "<maybe_dec_digits>": ["", "<dec_digits>"],
    "<dec_digit>": srange(string.digits),
    "<id>": ["x", "y", "z"]
}

SCRIPTSIZE_C_GRAMMAR = {
    "<start>": ["<statement>"],
    "<statement>": [
        "<block>",
        "<declaration>",
        "if<paren_expr> <statement> else <statement>",
        "if<paren_expr> <statement>",
        "while<paren_expr> <statement>",
        "do <statement> while<paren_expr>;",
        "<expr>;",
        ";"
    ],
    "<block>": ["{<statements>}"],
    "<statements>": ["<statement><statements>", ""],
    "<declaration>": [
        "int <id> = <expr>;",
        "int <id>;"
    ],
    "<paren_expr>": ["(<expr>)"],
    "<expr>": [
        "<id> = <expr>",
        "<test>",
    ],
    "<test>": [
        "<sum> < <sum>",
        "<sum>",
    ],
    "<sum>": [
        "<sum> + <term>",
        "<sum> - <term>",
        "<term>",
    ],
    "<term>": [
        "<paren_expr>",
        "<id>",
        "<int>",
    ],
    "<id>": srange(string.ascii_lowercase),
    "<int>": [
        "<digit_nonzero><digits>",
        "<digit>",
    ],
    "<digits>": [
        "<digit><int>",
        "<digit>",
    ],
    "<digit>": srange(string.digits),
    "<digit_nonzero>": list(set(srange(string.digits)) - {"0"}),
}

if __name__ == '__main__':
    unittest.main()
