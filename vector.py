import typing as tp
import struct
import math


class Vector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_bytes(cls, file: tp.BinaryIO) -> "Vector":
        x, y, z = struct.unpack("<fff", file.read(12))
        return cls(x, -z, y)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> None:
        length = self.length()
        assert abs(length) > 0.0001
        self.x /= length
        self.y /= length
        self.z /= length

    def __add__(self, other: "Vector") -> "Vector":
        assert isinstance(other, self.__class__)
        return self.__class__(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: "Vector") -> "Vector":
        assert isinstance(other, self.__class__)
        return self.__class__(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __rmul__(self, coeff: float) -> "Vector":
        return self.__class__(
            self.x * coeff,
            self.y * coeff,
            self.z * coeff,
        )

    def __mul__(self, other: float) -> "Vector":
        return self.__rmul__(other)

    def cross_product(self, other: "Vector") -> "Vector":
        assert isinstance(other, self.__class__)

        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x

        return Vector(x, y, z)

    def dot_product(self, other: "Vector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def tuple(self) -> tp.Tuple[float, float, float]:
        return self.x, self.y, self.z

    def __repr__(self) -> str:
        return f"({self.x},{self.y},{self.z})"

    def get_any_normal(self) -> "Vector":
        SMALL = 1e-6
        if abs(self.x) < SMALL and abs(self.y) < SMALL:
            return self.__class__(0, -self.z, self.y)
        return self.__class__(-self.y, self.x, 0)
