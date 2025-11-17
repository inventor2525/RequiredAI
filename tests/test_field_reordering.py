import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from RequiredAI.json_dataclass.implementation import json_dataclass


class TestFieldReordering(unittest.TestCase):
	"""Test that field reordering handles inheritance with defaults correctly."""

	def test_all_fields_without_defaults(self):
		"""Ensure classes where no fields have defaults still work."""
		@json_dataclass
		class NoDefaults:
			a: int
			b: str

		@json_dataclass
		class ChildNoDefaults(NoDefaults):
			c: float
		
		def inner(obj: ChildNoDefaults, a=1, b="test", c=2.5):
			self.assertEqual(obj.a, a)
			self.assertEqual(obj.b, b)
			self.assertEqual(obj.c, c)

			obj_dict = obj.to_dict()
			self.assertEqual(obj_dict['a'], a)
			self.assertEqual(obj_dict['b'], b)
			self.assertEqual(obj_dict['c'], c)

			obj2 = ChildNoDefaults.from_dict(obj_dict)
			self.assertEqual(obj2.a, a)
			self.assertEqual(obj2.b, b)
			self.assertEqual(obj2.c, c)

		inner(ChildNoDefaults(1, "test", 2.5))
		inner(ChildNoDefaults(42, "custom", 3.14), a=42, b="custom", c=3.14)
		inner(ChildNoDefaults(a=100, b="mixed", c=1.0), a=100, b="mixed", c=1.0)
		inner(ChildNoDefaults(100, b="mixed", c=1.0), a=100, b="mixed", c=1.0)
		inner(ChildNoDefaults(100, "mixed", c=1.0), a=100, b="mixed", c=1.0)

	def test_all_fields_with_defaults(self):
		"""Ensure classes where all fields have defaults still work."""
		@json_dataclass
		class AllDefaults:
			a: int = 1
			b: str = "default"

		@json_dataclass
		class ChildAllDefaults(AllDefaults):
			c: float = 3.14
		
		def inner(obj: ChildAllDefaults, a=1, b="default", c=3.14):
			self.assertEqual(obj.a, a)
			self.assertEqual(obj.b, b)
			self.assertEqual(obj.c, c)

			obj_dict = obj.to_dict()
			self.assertEqual(obj_dict['a'], a)
			self.assertEqual(obj_dict['b'], b)
			self.assertEqual(obj_dict['c'], c)

		inner(ChildAllDefaults())
		inner(ChildAllDefaults(a=42), a=42)
		inner(ChildAllDefaults(b="not default"), b="not default")
		inner(ChildAllDefaults(c=1.57), c=1.57)
		inner(ChildAllDefaults(c=1.57, a=42), a=42, c=1.57)
		inner(ChildAllDefaults(42), a=42)
		inner(ChildAllDefaults(42, c=1.57), a=42, c=1.57)
		inner(ChildAllDefaults(42, "custom", 2.71), a=42, b="custom", c=2.71)
		inner(ChildAllDefaults(a=100, b="mixed", c=1.0), a=100, b="mixed", c=1.0)

	def test_parent_with_default_child_without(self):
		"""Parent has field with default, child has field without default."""
		@json_dataclass
		class Parent:
			a: int = 5

		@json_dataclass
		class Child(Parent):
			b: int
		
		def inner(obj: Child, a=5, b=10):
			self.assertEqual(obj.a, a)
			self.assertEqual(obj.b, b)

			obj_dict = obj.to_dict()
			self.assertEqual(obj_dict['a'], a)
			self.assertEqual(obj_dict['b'], b)

			obj2 = Child.from_dict(obj_dict)
			self.assertEqual(obj2.a, a)
			self.assertEqual(obj2.b, b)

		inner(Child(10))
		inner(Child(b=10))
		inner(Child(10, 7), a=7, b=10)
		inner(Child(a=7, b=20), a=7, b=20)
		inner(Child(20, a=7), a=7, b=20)
		inner(Child(b=30), b=30)
		inner(Child(30, 15), a=15, b=30)

	def test_override_parent_default(self):
		"""Test that child can override parent's default value."""
		@json_dataclass
		class Parent:
			value: int = 5

		@json_dataclass
		class Child(Parent):
			value: int = 10
			extra: str
		
		def inner(obj: Child, value=10, extra="test"):
			self.assertEqual(obj.value, value)
			self.assertEqual(obj.extra, extra)

			obj_dict = obj.to_dict()
			self.assertEqual(obj_dict['value'], value)
			self.assertEqual(obj_dict['extra'], extra)

			obj2 = Child.from_dict(obj_dict)
			self.assertEqual(obj2.value, value)
			self.assertEqual(obj2.extra, extra)

		inner(Child("test"))
		inner(Child(extra="test"))
		inner(Child("test", 20), value=20)
		inner(Child(value=20, extra="test"), value=20)
		inner(Child(extra="custom"), extra="custom")
		inner(Child("custom", value=30), value=30, extra="custom")
		inner(Child(value=30, extra="override"), value=30, extra="override")

	def test_multiple_inheritance_levels(self):
		"""Test multiple levels of inheritance with mixed defaults."""
		@json_dataclass
		class GrandParent:
			x: str = "default"

		@json_dataclass
		class Parent(GrandParent):
			y: int

		@json_dataclass
		class Child(Parent):
			z: float = 3.14
		
		def inner(obj: Child, x="default", y=42, z=3.14):
			self.assertEqual(obj.x, x)
			self.assertEqual(obj.y, y)
			self.assertEqual(obj.z, z)

			obj_dict = obj.to_dict()
			self.assertEqual(obj_dict['x'], x)
			self.assertEqual(obj_dict['y'], y)
			self.assertEqual(obj_dict['z'], z)

			obj2 = Child.from_dict(obj_dict)
			self.assertEqual(obj2.x, x)
			self.assertEqual(obj2.y, y)
			self.assertEqual(obj2.z, z)

		inner(Child(42))
		inner(Child(y=42))
		inner(Child(42, z=2.71), z=2.71)
		inner(Child(y=100, z=2.71), y=100, z=2.71)
		inner(Child(100, "custom"), x="custom", y=100)
		inner(Child(x="custom", y=100, z=2.71), x="custom", y=100, z=2.71)
		inner(Child(y=50), y=50)
		inner(Child(200, z=1.0), y=200, z=1.0)
		inner(Child(200, "alt", 1.0), x="alt", y=200, z=1.0)

	def test_complex_inheritance_pattern(self):
		"""Test complex pattern: default, no-default, default, no-default."""
		@json_dataclass
		class Base:
			a: int = 1
			b: str = "base"

		@json_dataclass
		class Middle(Base):
			c: float
			d: bool = True

		@json_dataclass
		class Derived(Middle):
			e: str
			f: int = 99
		
		def inner(obj: Derived, a=1, b="base", c=2.5, d=True, e="test", f=99):
			self.assertEqual(obj.a, a)
			self.assertEqual(obj.b, b)
			self.assertEqual(obj.c, c)
			self.assertEqual(obj.d, d)
			self.assertEqual(obj.e, e)
			self.assertEqual(obj.f, f)

			obj_dict = obj.to_dict()
			self.assertEqual(obj_dict['a'], a)
			self.assertEqual(obj_dict['b'], b)
			self.assertEqual(obj_dict['c'], c)
			self.assertEqual(obj_dict['d'], d)
			self.assertEqual(obj_dict['e'], e)
			self.assertEqual(obj_dict['f'], f)

			obj2 = Derived.from_dict(obj_dict)
			self.assertEqual(obj2.a, a)
			self.assertEqual(obj2.b, b)
			self.assertEqual(obj2.c, c)
			self.assertEqual(obj2.d, d)
			self.assertEqual(obj2.e, e)
			self.assertEqual(obj2.f, f)

		inner(Derived(2.5, "test"))
		inner(Derived(c=2.5, e="test"))
		inner(Derived(2.5, "test", f=50), f=50)
		inner(Derived(c=1.5, e="hello", f=50), c=1.5, e="hello", f=50)
		inner(Derived(1.5, "hello", a=10, b="custom", d=False, f=50), a=10, b="custom", c=1.5, d=False, e="hello", f=50)
		inner(Derived(a=10, b="custom", c=1.5, d=False, e="hello", f=50), a=10, b="custom", c=1.5, d=False, e="hello", f=50)
		inner(Derived(c=3.0, e="world"), c=3.0, e="world")
		inner(Derived(4.0, "complex", a=20, d=False, f=100), a=20, c=4.0, d=False, e="complex", f=100)
		inner(Derived(4.0, "complex", 20, "alt", False, 100), a=20, b="alt", c=4.0, d=False, e="complex", f=100)

if __name__ == '__main__':
	unittest.main()