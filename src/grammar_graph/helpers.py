import re
from typing import TypeVar, Callable, List, Tuple, Set

from grammar_graph.type_defs import Tree, Path, ParseTree, Grammar

TRAVERSE_PREORDER = 0
TRAVERSE_POSTORDER = 1

RE_NONTERMINAL = re.compile(r'(<[^<> ]*>)')


def nonterminals(expansion: str) -> List[str]:
    return RE_NONTERMINAL.findall(expansion)


def reachable_nonterminals(grammar: Grammar, _start_symbol='<start>') -> Set[str]:
    reachable = set()

    def _find_reachable_nonterminals(grammar, symbol):
        nonlocal reachable
        reachable.add(symbol)
        for expansion in grammar.get(symbol, []):
            for nonterminal in nonterminals(expansion):
                if nonterminal not in reachable:
                    _find_reachable_nonterminals(grammar, nonterminal)

    _find_reachable_nonterminals(grammar, _start_symbol)
    return reachable


def unreachable_nonterminals(grammar: Grammar, _start_symbol='<start>') -> Set[str]:
    return grammar.keys() - reachable_nonterminals(grammar, _start_symbol)


def delete_unreachable(grammar: Grammar) -> None:
    for unreachable in unreachable_nonterminals(grammar):
        del grammar[unreachable]


def traverse_tree(
        tree: ParseTree,
        action: Callable[[Path, ParseTree], None],
        abort_condition: Callable[[Path, ParseTree], bool] = lambda p, n: False,
        kind: int = TRAVERSE_PREORDER,
        reverse: bool = False) -> None:
    stack_1: List[Tuple[Path, ParseTree]] = [((), tree)]
    stack_2: List[Tuple[Path, ParseTree]] = []

    if kind == TRAVERSE_PREORDER:
        reverse = not reverse

    while stack_1:
        path, node = stack_1.pop()

        if abort_condition(path, node):
            return

        if kind == TRAVERSE_POSTORDER:
            stack_2.append((path, node))

        if kind == TRAVERSE_PREORDER:
            action(path, node)

        if node[1]:
            iterator = reversed(node[1]) if reverse else iter(node[1])

            for idx, child in enumerate(iterator):
                new_path = path + ((len(node[1]) - idx - 1) if reverse else idx,)
                stack_1.append((new_path, child))

    if kind == TRAVERSE_POSTORDER:
        while stack_2:
            action(*stack_2.pop())
