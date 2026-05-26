import unicodedata
from collections import OrderedDict
from typing import Any


class LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str, default=None):
        if key not in self.data:
            return default

        self.data.move_to_end(key)
        return self.data[key]

    def set(self, key: str, value: Any):
        if key in self.data:
            self.data.move_to_end(key)

        self.data[key] = value

        if len(self.data) > self.maxsize:
            self.data.popitem(last=False)

    def clear(self) -> None:
        self.data.clear()


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFC", name)
    return name
