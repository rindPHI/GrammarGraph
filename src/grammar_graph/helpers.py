import re
from functools import wraps
from typing import TypeVar, Callable, List, Tuple, Set, cast

from grammar_graph.type_defs import Tree, Path, ParseTree, Grammar, ImmutableGrammar

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


def grammar_to_immutable(grammar: Grammar) -> ImmutableGrammar:
    return cast(ImmutableGrammar, tuple({k: tuple(v) for k, v in grammar.items()}.items()))


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


def parse_tree_arg_hashable(a_func: callable) -> callable:
    # This assumes that the first argument of the decorated function is a `ParseTree`

    @wraps(a_func)
    def decorated(*args, **kwargs):
        assert isinstance(args[1], ParseTree)
        args = (args[0], parse_tree_to_immutable(args[1]),) + args[2:]
        return a_func(*args, **kwargs)

    return decorated


def parse_tree_to_immutable(elem: ParseTree) -> ParseTree:
    stack: List[ParseTree] = []

    def action(_, node: ParseTree):
        if not node[1]:
            # noinspection PyTypeChecker
            stack.append((node[0], None if node[1] is None else ()))
        else:
            children = []
            for _ in range(len(node[1])):
                children.append(stack.pop())
            # noinspection PyTypeChecker
            stack.append((node[0], tuple(children)))

    traverse_tree(elem, action, kind=TRAVERSE_POSTORDER, reverse=True)

    assert len(stack) == 1
    return stack[0]