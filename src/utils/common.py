from abc import ABC, abstractmethod
from collections.abc import Hashable as SupportsHash


class Hashable(ABC):
    @abstractmethod
    def _key_(self) -> SupportsHash:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self._key_())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._key_() == other._key_()
