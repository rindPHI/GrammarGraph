from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Optional, List, Iterator, Dict

T = TypeVar('T')
Path = Tuple[int, ...]
Tree = Tuple[T, Optional[List['Tree[T]']]]

NonterminalType = str
Grammar = Dict[NonterminalType, List[str]]


class ParseTree(ABC):
    @classmethod
    def __subclasshook__(cls, C):
        return hasattr(C, "__iter__") and hasattr(C, "__getitem__")

    @abstractmethod
    def __iter__(self) -> Iterator[str | List['ParseTree'] | None]:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, item: int) -> str | List['ParseTree'] | None:
        raise NotImplementedError()
