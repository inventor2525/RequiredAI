from dataclasses import dataclass, Field, field
from dataclasses_json import dataclass_json, config
from typing import List, Dict, Any, Type, TypeVar, Generic, Callable, ClassVar, Optional
from typing import get_origin, get_args, Annotated, ForwardRef, overload, Iterator
from enum import Enum
import collections
import uuid

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
_RefByIDType = TypeVar('_RefByIDType')

ReferenceByID = Annotated[_RefByIDType, "Reference By ID"]
MISSING = object()
_default_auto_id_name = '__id__'
_default_use_small_ids = False

class IDType(Enum):
	NONE=0
	USER=1
	UUID=2
	INCREMENT=4
	
	@staticmethod
	def IsNotNone(t:'IDType') -> bool:
		return t!=None and t!=MISSING and t!=IDType.NONE

class ObjectID:
	info_field_name = '__id_info__'
	by_id_field_name = '__by_id__'
	
	id_type = str|Any
	object_type = Any
	
	class List(Generic[T]):
		def __init__(self, backing_field: List[str], item_type: Type[T]):
			self.backing_field = backing_field
			self.item_type = item_type

		def append(self, item: T):
			id_info = ObjectID.get_id_info(self.item_type)
			self.backing_field.append(id_info.id_for(item))

		def __getitem__(self, index: int | slice) -> T | List[T]:
			id_info = ObjectID.get_id_info(self.item_type)
			if isinstance(index, slice):
				return [id_info.get(item_id) for item_id in self.backing_field[index]]
			return id_info.get(self.backing_field[index])

		def __setitem__(self, index: int | slice, value: T | List[T]):
			id_info = ObjectID.get_id_info(self.item_type)
			if isinstance(index, slice):
				self.backing_field[index] = [id_info.id_for(item) for item in value]
			else:
				self.backing_field[index] = id_info.id_for(value)

		def __len__(self):
			return len(self.backing_field)

		def __iter__(self) -> Iterator[T]:
			id_info = ObjectID.get_id_info(self.item_type)
			for item_id in self.backing_field:
				yield id_info.get(item_id)
	
	class Dict(Generic[K, V]):
		def __init__(self, backing_field: Dict[Any, Any], key_type: Type[K], value_type: Type[V]):
			self.backing_field = backing_field
			self.key_type = key_type
			self.value_type = value_type
			self._key_id_info = ObjectID.get_id_info(key_type)
			self._value_id_info = ObjectID.get_id_info(value_type)

		def __getitem__(self, key: K) -> V:
			key_id = key if not self._key_id_info else self._key_id_info.id_for(key)
			value_id = self.backing_field[key_id]
			return value_id if not self._value_id_info else self._value_id_info.get(value_id)

		def __setitem__(self, key: K, value: V):
			key_id = key if not self._key_id_info else self._key_id_info.id_for(key)
			value_id = value if not self._value_id_info else self._value_id_info.id_for(value)
			self.backing_field[key_id] = value_id

		def __len__(self):
			return len(self.backing_field)

		def __iter__(self) -> Iterator[K]:
			for key_id in self.backing_field:
				yield key_id if not self._key_id_info else self._key_id_info.get(key_id)

		def items(self) -> Iterator[tuple[K, V]]:
			for key_id, value_id in self.backing_field.items():
				key = key_id if not self._key_id_info else self._key_id_info.get(key_id)
				value = value_id if not self._value_id_info else self._value_id_info.get(value_id)
				yield key, value

		def keys(self) -> Iterator[K]:
			return self.__iter__()

		def values(self) -> Iterator[V]:
			for value_id in self.backing_field.values():
				yield value_id if not self._value_id_info else self._value_id_info.get(value_id)

	def __init__(self, cls:object_type, id_type:IDType, name:str):
		self.cls = cls
		self.name = name
		self.type = id_type
	
	@staticmethod
	def generate_uuid() -> str:
		return str(uuid.uuid4())
	
	def generate_increment_id(self) -> str:
		id = getattr(self, 'current_increment_id', 0)
		self.current_increment_id = id+1
		return f"{self.cls.__name__}_{id}"
	
	@staticmethod
	def get_id_info(cls:Any) -> 'ObjectID':
		return getattr(cls, ObjectID.info_field_name, ObjectID(None, IDType.NONE, None))
	
	def setup(self):
		if self:
			# Store self on cls:
			setattr(self.cls, ObjectID.info_field_name, self)
			
			# Create any auto id fields:
			if self.type == IDType.UUID:
				setattr(self.cls,self.name, field(default_factory=ObjectID.generate_uuid, kw_only=False, init=False))
				self.cls.__annotations__[self.name] = str
			if self.type == IDType.INCREMENT:
				setattr(self.cls,self.name, field(default_factory=lambda self=self:self.generate_increment_id(), kw_only=False, init=False))
				self.cls.__annotations__[self.name] = str
			
			# Setup by id tracking of instances:
			setattr(self.cls, ObjectID.by_id_field_name, {})
			self.cls.__annotations__[ObjectID.by_id_field_name] = ClassVar[Dict[self.id_type, self.object_type]]
	
	@property
	def _by_id_(self) -> Dict[id_type, object_type]:
		'''
		The static collection for storing objects by key on self.cls.
		'''
		return getattr(self.cls, ObjectID.by_id_field_name)
	
	def get(self, id:id_type) -> object_type:
		'''
		Get object instance by id for type self.cls.
		'''
		return self._by_id_.get(id, None)
	
	def append(self, obj:object_type):
		'''
		Store obj in self.cls's static collection at obj's id.
		'''
		id_val = getattr(obj, self.name)
		self._by_id_[id_val] = obj
	
	def id_for(self, object:object_type) -> id_type:
		'''
		Get's the id 
		'''
		id_val = MISSING if not self.name else getattr(object, self.name, MISSING)
		assert id_val!=MISSING and self.type, f"No ID field assigned for type '{type(object)}'."
		return id_val
	
	def __bool__(self) -> bool:
		return self.type != IDType.NONE

class _JSON_DataclassMixin(Generic[T]):
	'''
	Used to type hind dataclass_json methods.
	
	This is not used at runtime, and you will never get an actual instance of this.
	'''
	def to_dict(self) -> Dict[str, Any]:
		pass
	def to_json(self,indent:int) -> str:
		pass
	@staticmethod
	def from_dict(dict:Dict[str,Any]) -> T:
		pass
	@staticmethod
	def from_json(j:str) -> T:
		pass

@overload
def json_dataclass(id_type:IDType=MISSING, has_id:bool=MISSING, auto_id_name:str=_default_auto_id_name, user_id_name:str=None, exclude:List[str|Type]=[collections.abc.Callable]) -> Callable[[Type[T]], Type[T] | Type[_JSON_DataclassMixin[T]]]:
	pass

@overload
def json_dataclass(_cls: Type[T]) -> Type[T] | Type[_JSON_DataclassMixin[T]]:
	pass
def json_dataclass(*args, **kwargs) -> Callable[[Type[T]], Type[T] | Type[_JSON_DataclassMixin[T]]] | Type[T] | Type[_JSON_DataclassMixin[T]]:
	def wrap(cls:Type[T]) -> Type[T]:
		return _process_class(cls, *args, **kwargs)
	
	if len(args)==1 and len(kwargs)==0 and isinstance(args[0], type):
		#if the only argument we have is a type, it's the thing we're decorating:
		return _process_class(args[0])
	# if not, we'll assume _cls is an arg and return a decorator that will treat it like one:
	return wrap

def _process_class(cls: Type[T], id_type:IDType=MISSING, has_id:bool=MISSING, auto_id_name:str=_default_auto_id_name, user_id_name:str=None, small_id:bool=_default_use_small_ids, exclude:List[str|Type]=[collections.abc.Callable]) -> Type[T] | Type[_JSON_DataclassMixin[T]]:
	# Figure out if we have an id, and what type we have if so:
	has_id = (
		has_id
		or IDType.IsNotNone(id_type)
		or auto_id_name != _default_auto_id_name
		or user_id_name
		or small_id!=_default_use_small_ids
	)
	if not has_id:
		id_type = IDType.NONE
	else:
		if IDType.IsNotNone(id_type):
			if id_type == IDType.UUID:
				assert auto_id_name, "Auto id field name must be supplied when using UUIDs."
			elif id_type == IDType.INCREMENT:
				assert auto_id_name, "Auto id field name must be supplied when using INCREMENT ids."
			elif id_type == IDType.USER:
				assert user_id_name, "User id field name must be supplied when using user supplied id fields."
			else:
				raise NotImplemented
		elif auto_id_name != _default_auto_id_name:
			id_type = IDType.INCREMENT if small_id else IDType.UUID
		elif user_id_name:
			id_type = IDType.USER
		else:
			id_type = IDType.INCREMENT if small_id else IDType.UUID
	
	obj_id = ObjectID(cls, id_type, user_id_name if id_type == IDType.USER else auto_id_name)
	obj_id.setup()
	
	# Organize what things we're to exclude:
	field_exclusion = set()
	type_exclusion = set()
	for item in exclude:
		if isinstance(item, str):
			field_exclusion.add(item)
		else:
			type_exclusion.add(item)
	
	def replace_dict_item(d: dict, old_key, new_key, new_value) -> dict:
		'''
		Changes both key and value for an item, keeping it's insertion order.
		
		This removes the key value pair for 'old_key' and inserts the new key
		value pair at the same location.
		'''
		return {
			(k if k != old_key else new_key):
			(v if k != old_key else new_value)
			for k, v in d.items()
		}
	
	U = TypeVar('U')
	def add_property(prop_name: str, prop_type: Type[U], getter: Callable[[T], U], setter: Callable[[T, U], None]) -> None:
		'''Add property of a specified type.'''
		prop = property(
			fget=getter,
			fset=setter,
			doc=f"Property {prop_name} of type {prop_type.__name__}"
		)
		setattr(cls, prop_name, prop)
	
	def exclude_field(field_name:str):
		'''
		Mark this field never to be serialized.
		'''
		default = getattr(cls, field_name, MISSING)
		if default == MISSING:
			setattr(cls, field_name, field(metadata=config(exclude=lambda _:True)))
		elif isinstance(default, Field):
			default.metadata = config(metadata=dict(default.metadata), exclude=lambda _:True)
		else:
			setattr(cls, field_name, field(default=default, exclude=lambda _:True))
	
	# Replace all fields marked ReferenceByID with a property that
	# references them by id in the static mapping of objects
	# referenced by id held inside that fields type.
	for field_name, field_type in list(cls.__annotations__.items()):
		if field_name in field_exclusion:
			exclude_field(field_name)
			continue
		
		type_origin = get_origin(field_type)
		if type_origin in type_exclusion or field_type in type_exclusion:
			exclude_field(field_name)
			continue
					
		if type_origin == Annotated:
			type_args = get_args(field_type)
			if len(type_args)==2 and type_args[1]=="Reference By ID":
				# Replace the field with a backing field, typed as the annotated type:
				field_type = type_args[0]
				if isinstance(field_type, str):
					if field_type == cls.__name__:
						field_type = cls
				elif isinstance(field_type, ForwardRef):
					fr_str = str(field_type)[12:-2]
					if fr_str == cls.__name__:
						field_type = cls
				new_field_name = f"__{field_name}__"
				
				type_origin = get_origin(field_type)
				if type_origin in type_exclusion or field_type in type_exclusion:
					exclude_field(field_name)
					continue
				
				# Set the default value if provided:
				field_default = getattr(cls, field_name, MISSING)
				if field_default is MISSING:
					setattr(cls, new_field_name, field(metadata=config(field_name=field_name)))
				elif isinstance(field_default, Field):
					field_default.metadata = config(metadata=dict(field_default.metadata), field_name=field_name)
					setattr(cls, new_field_name, field_default)
				else:
					setattr(cls, new_field_name, field(default=field_default, metadata=config(field_name=field_name)))
				
				# Replace the field with a property:
				if type_origin == None:
					cls.__annotations__ = replace_dict_item(cls.__annotations__, field_name, new_field_name, Optional[str])
					
					def prop_get(self, new_field_name=new_field_name, field_type=field_type):
						'''Gets the object by id from it's type's static map.'''
						obj_id = getattr(self, new_field_name)
						return ObjectID.get_id_info(field_type).get(obj_id)
					def prop_set(self, val, new_field_name:str=new_field_name, field_type=field_type):
						'''Store the values uuid (if it has one) to the backing field.'''
						obj_id = ObjectID.get_id_info(field_type).id_for(val)
						setattr(self, new_field_name, obj_id)
					add_property(field_name, field_type, prop_get, prop_set)
				elif type_origin == list:
					cls.__annotations__ = replace_dict_item(cls.__annotations__, field_name, new_field_name, Optional[List[str]])
					
					element_type = get_args(field_type)[0]
					def prop_get(self, new_field_name=new_field_name, element_type=element_type):
						'''Gets the object by id from it's type's static map.'''
						backing_list = getattr(self, new_field_name)
						return ObjectID.List(backing_list,element_type)
					def prop_set(self, val, new_field_name:str=new_field_name, element_type=element_type):
						'''Store the values uuid (if it has one) to the backing field.'''
						if val is None:
							setattr(self, new_field_name, None)
						else:
							ids = [ObjectID.get_id_info(element_type).id_for(element) for element in val]
							setattr(self, new_field_name, ids)
					add_property(field_name, field_type, prop_get, prop_set)
				elif type_origin == dict:
					key_type, value_type = get_args(field_type)
					key_id_info = ObjectID.get_id_info(key_type)
					value_id_info = ObjectID.get_id_info(value_type)
					dkt = key_type if not key_id_info else str
					dvt = value_type if not value_id_info else str
					cls.__annotations__ = replace_dict_item(cls.__annotations__, field_name, new_field_name, Optional[Dict[dkt,dvt]])
					
					def prop_get(self, new_field_name:str=new_field_name, key_type: Type[K]=key_type, value_type: Type[V]=value_type) -> ObjectID.Dict[K, V]:
						return ObjectID.Dict(getattr(self, new_field_name), key_type, value_type)

					def prop_set(self, value: Dict[K, V], new_field_name:str=new_field_name, key_type: Type[K]=key_type, value_type: Type[V]=value_type, key_id_info:ObjectID=key_id_info, value_id_info:ObjectID=value_id_info):
						backing_dict = {
							(k if not key_id_info else key_id_info.id_for(k)): 
							(v if not value_id_info else value_id_info.id_for(v)) 
							for k, v in value.items()
						}
						setattr(self, new_field_name, backing_dict)
					add_property(field_name, field_type, prop_get, prop_set)
	
	if obj_id:
		original_post_init = getattr(cls, '__post_init__', None)
		def new_post_init(self, obj_id=obj_id):
			obj_id.append(self)
			if original_post_init is not None:
				original_post_init(self)
		cls.__post_init__ = new_post_init
	
	cls = dataclass(cls)
	cls = dataclass_json(cls)

	return cls

if __name__ == "__main__":
	@json_dataclass(has_id=True)
	class node:
		my_val:str
		prev:ReferenceByID['node'] = field(default=None)
		next:ReferenceByID['node'] = None
		
		def add(self, new:'node'):
			self.next = new
			new.prev = self

	@json_dataclass(has_id=True)
	class holder:
		my_val:str
		all_nodes:List[node] = field(default_factory=list)
		ref_nodes:ReferenceByID[List[node]] = field(default_factory=list)
		ref_nodes_map:ReferenceByID[Dict[str, node]] = field(default_factory=dict)
		other:ReferenceByID['holder'] = None
		
		def add(self, node:node):
			if len(self.all_nodes)>0:
				self.all_nodes[-1].add(node)
			self.all_nodes.append(node)
			self.ref_nodes.append(node)
			self.ref_nodes_map[node.my_val] = node

	@json_dataclass
	class holders_holder:
		holder1:holder
		holder2:holder
	import json
	h = holder("Hi")
	h.add(node("person"))
	h.add(node("dog"))
	h.add(node("and"))
	h.add(node("cat"))
	h.all_nodes[-1].add(h.all_nodes[0])
	other = holder("bye")
	h.other = other
	hh = holders_holder(h, h.other)
	hh_dict = hh.to_dict()
	hh2 = holders_holder.from_dict(hh_dict)
	h_dict = hh_dict# h.to_dict()
	h2 = hh2.holder1

	print(json.dumps(h_dict, indent=4))
	print(h2)
	print("Bye!")

	i = 0
	current:node = h2.all_nodes[0]
	while current and i<10:
		i+=1
		print(current.my_val)
		current = current.next

	node_info = ObjectID.get_id_info(node)
	for a, b, ar, br, arm, brm in zip(h.all_nodes, h2.all_nodes, h.ref_nodes, h2.ref_nodes, h2.ref_nodes_map.keys(), h2.ref_nodes_map.values()):
		print('------')
		print(id(a), id(b), " is = ", a==b, " but their ids are: ", node_info.id_for(a), node_info.id_for(b))
		print(id(ar), id(br), " is = ", ar==br, " but their ids are: ", node_info.id_for(ar), node_info.id_for(br))
		print(arm, id(brm), node_info.id_for(brm))