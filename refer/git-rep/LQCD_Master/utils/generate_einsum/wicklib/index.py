from typing import Dict, Type, Union


class Index:
    _letter = ""
    _counter = -1

    @classmethod
    def new(cls):
        name = f"{cls._letter}{cls._counter}"
        cls._counter += 1
        return cls(name)

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


class SpinIndex(Index):
    _letter = "σ"  # "αβγδεζηθικλμνξοπρστυφχψω"
    _counter = 0


class ColorIndex(Index):
    _letter = "c"  # "abcdefghijklmnopqrstuvwxyz"
    _counter = 0


class EinsumIndex:
    _letters = ""
    _index = -1

    @classmethod
    def new(cls):
        if cls._index >= len(cls._letters):
            raise ValueError("too many indices for einsum string")
        name = f"{cls._letters[cls._index]}"
        cls._index += 1
        return cls(name)

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    @classmethod
    def reset(cls):
        cls._index = 0


class SpinEinsumIndex(EinsumIndex):
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _index = 0


class ColorEinsumIndex(EinsumIndex):
    _letters = "abcdefghijklmnopqrstuvwxyz"
    _index = 0


class IndexMap:
    def __init__(self):
        self.spin_map: Dict[str, str] = {}
        self.color_map: Dict[str, str] = {}

    def setdefault_spin(self, key: str, index: Union[Type[Index], Type[EinsumIndex]]) -> str:
        if key not in self.spin_map:
            self.spin_map[key] = index.new().name
        return self.spin_map[key]

    def setdefault_color(self, key: str, index: Union[Type[Index], Type[EinsumIndex]]) -> str:
        if key not in self.color_map:
            self.color_map[key] = index.new().name
        return self.color_map[key]

    def set_spin(self, key: str, value: str) -> str:
        if key not in self.spin_map:
            self.spin_map[key] = value
        if self.spin_map[key] != value:
            raise ValueError(f"spin index {key} is already mapped to {self.spin_map[key]}, cannot map to {value}")
        return self.spin_map[key]

    def set_color(self, key: str, value: str) -> str:
        if key not in self.color_map:
            self.color_map[key] = value
        if self.color_map[key] != value:
            raise ValueError(f"color index {key} is already mapped to {self.color_map[key]}, cannot map to {value}")
        return self.color_map[key]

    def getdefault_spin(self, key: str) -> str:
        if key not in self.spin_map:
            return key
        return self.spin_map[key]

    def getdefault_color(self, key: str) -> str:
        if key not in self.color_map:
            return key
        return self.color_map[key]

    def get_spin(self, key: str) -> str:
        if key not in self.spin_map:
            raise ValueError(f"spin index {key} is not mapped")
        return self.spin_map[key]

    def get_color(self, key: str) -> str:
        if key not in self.color_map:
            raise ValueError(f"color index {key} is not mapped")
        return self.color_map[key]

    def delta_spin(self, left: str, right: str, index: Union[Type[Index], Type[EinsumIndex]]):
        if left in self.spin_map and right in self.spin_map:
            if self.spin_map[left] != self.spin_map[right]:
                raise ValueError(f"spin indices {left} and {right} are mapped to different values for delta tensor")
        elif left in self.spin_map:
            self.set_spin(right, self.spin_map[left])
        elif right in self.spin_map:
            self.set_spin(left, self.spin_map[right])
        else:
            A = index.new().name
            self.set_spin(left, A)
            self.set_spin(right, A)

    def delta_color(self, left: str, right: str, index: Union[Type[Index], Type[EinsumIndex]]):
        if left in self.color_map and right in self.color_map:
            if self.color_map[left] != self.color_map[right]:
                raise ValueError(f"color indices {left} and {right} are mapped to different values for delta tensor")
        elif left in self.color_map:
            self.set_color(right, self.color_map[left])
        elif right in self.color_map:
            self.set_color(left, self.color_map[right])
        else:
            a = index.new().name
            self.set_color(left, a)
            self.set_color(right, a)
