from typing import cast, Literal, NoReturn, TypeVar

TEnsureNotNone = TypeVar("TEnsureNotNone")

def ensure(value: TEnsureNotNone | None) -> TEnsureNotNone:
	assert value is not None
	return cast(TEnsureNotNone, value)

import ast

def compile_error(type: type[Exception], message: str, source: ast.AST | None = None) -> NoReturn:
	raise type(f"{message} line {source.lineno if source else 'unknown'}")

class Register:
	def __eq__(self, __o: object) -> bool: return str(self) == str(__o)
	def __ne__(self, __o: object) -> bool: return str(self) != str(__o)
	def __repr__(self) -> str: raise NotImplementedError()

class AX(Register):
	def lower(self) -> Register: return AL()
	def upper(self) -> Register: return AH()
	def __repr__(self) -> str: return "ax"

class AL(Register):
	def __repr__(self) -> str: return "al"

class AH(Register):
	def __repr__(self) -> str: return "ah"

class BX(Register):
	def lower(self) -> Register: return BL()
	def upper(self) -> Register: return BH()
	def __repr__(self) -> str: return "bx"

class BL(Register):
	def __repr__(self) -> str: return "bl"

class BH(Register):
	def __repr__(self) -> str: return "bh"

class CX(Register):
	def lower(self) -> Register: return CL()
	def upper(self) -> Register: return CH()
	def __repr__(self) -> str: return "cx"

class CL(Register):
	def __repr__(self) -> str: return "cl"

class CH(Register):
	def __repr__(self) -> str: return "ch"

class DX(Register):
	def lower(self) -> Register: return DL()
	def upper(self) -> Register: return DH()
	def __repr__(self) -> str: return "dx"

class DL(Register):
	def __repr__(self) -> str: return "dl"

class DH(Register):
	def __repr__(self) -> str: return "dh"

class SI(Register):
	def __repr__(self) -> str: return "si"

class DI(Register):
	def __repr__(self) -> str: return "di"

class BP(Register):
	def __repr__(self) -> str: return "bp"

class SP(Register):
	def __repr__(self) -> str: return "sp"

class SegmentRegister:
	def __eq__(self, __o: object) -> bool: return str(self) == str(__o)
	def __ne__(self, __o: object) -> bool: return str(self) != str(__o)
	def __repr__(self) -> str: raise NotImplementedError()

class CS(SegmentRegister):
	def __repr__(self) -> str: return "cs"

class DS(SegmentRegister):
	def __repr__(self) -> str: return "ds"

class ES(SegmentRegister):
	def __repr__(self) -> str: return "es"

class SS(SegmentRegister):
	def __repr__(self) -> str: return "ss"

REGISTERS: dict[str, Register | SegmentRegister] = {
	"ax": AX(),
	"al": AL(),
	"ah": AH(),
	"bx": BX(),
	"bl": BL(),
	"bh": BH(),
	"cx": CX(),
	"cl": CL(),
	"ch": CH(),
	"dx": DX(),
	"dl": DL(),
	"dh": DH(),
	"si": SI(),
	"di": DI(),
	"bp": BP(),
	"sp": SP(),
	"cs": CS(),
	"ds": DS(),
	"es": ES(),
	"ss": SS()
}

class Label:
	next_id = 0
	
	def __init__(self, name: str | None = None) -> None:
		self.id = Label.next_id
		Label.next_id += 1
		self.name = name if name else f"__l{self.id}"
	
	def __repr__(self) -> str: return self.name

class AddressedValue:
	def __init__(self, pointer: int | Register | Label) -> None: self.pointer = pointer
	def __repr__(self) -> str: return f"[{self.pointer}]"

class Instruction:
	def __init__(self, op: str, *operands) -> None:
		self.op = op
		self.operands = operands
	
	def __repr__(self) -> str:
		if self.op == "raw": return f"\t{self.operands[0]}"
		elif self.op == "marklabel": return f"{self.operands[0]}:"
		else: return f"\t{self.op} {', '.join([str(operand) for operand in self.operands])}"

class Bits(Instruction):
	def __init__(self, value: int) -> None:
		super().__init__("bits", value)
		self.value = value

class DataByte(Instruction):
	def __init__(self, values: list[int]) -> None:
		values = [value & 0xFF for value in values]
		super().__init__("db", *values)
		self.values = values

class DataWord(Instruction):
	def __init__(self, values: list[int]) -> None:
		values = [value & 0xFFFF for value in values]
		super().__init__("dw", *values)
		self.values = values

class DataDoubleWord(Instruction):
	def __init__(self, values: list[int]) -> None:
		values = [value & 0xFFFFFFFF for value in values]
		super().__init__("dd", *values)
		self.values = values

class DataQuadWord(Instruction):
	def __init__(self, values: list[int]) -> None:
		values = [value & 0xFFFFFFFFFFFFFFFF for value in values]
		super().__init__("dq", *values)
		self.values = values

class Push(Instruction):
	def __init__(self, value: int | Register | SegmentRegister | Label) -> None:
		super().__init__("push", value)
		self.value = value

class Pop(Instruction):
	def __init__(self, target: Register | SegmentRegister) -> None:
		super().__init__("pop", target)
		self.target = target

class And(Instruction):
	def __init__(self, a: Register, b: int | Register | Label) -> None:
		super().__init__("and", a, b)

class Or(Instruction):
	def __init__(self, a: Register, b: int | Register | Label) -> None:
		super().__init__("or", a, b)

class Xor(Instruction):
	def __init__(self, a: Register, b: int | Register | Label) -> None:
		super().__init__("xor", a, b)

class Add(Instruction):
	def __init__(self, a: Register, b: int | Register | Label) -> None:
		super().__init__("add", a, b)
		self.a = a
		self.b = b

class Sub(Instruction):
	def __init__(self, a: Register, b: int | Register | Label) -> None:
		super().__init__("sub", a, b)
		self.a = a
		self.b = b

class IMul(Instruction):
	def __init__(self, b: Register) -> None:
		super().__init__("imul", b)
		self.a = AX()
		self.b = b

class Mul(Instruction):
	def __init__(self, b: Register) -> None:
		super().__init__("mul", b)
		self.a = AX()
		self.b = b

class IDiv(Instruction):
	def __init__(self, b: Register) -> None:
		super().__init__("idiv", b)
		self.a = AX()
		self.b = b

class Div(Instruction):
	def __init__(self, b: Register) -> None:
		super().__init__("div", b)
		self.a = AX()
		self.b = b

class Sar(Instruction):
	def __init__(self, a: Register, b: CL | Literal[1]) -> None:
		super().__init__("sar", a, b)
		self.a = a
		self.b = b

class Cmp(Instruction):
	def __init__(self, a: Register, b: int | Register | Label) -> None:
		super().__init__("cmp", a, b)
		self.a = a
		self.b = b

class Mov(Instruction):
	def __init__(self, target: Register, source: int | Register | Label) -> None:
		super().__init__("mov", target, source)
		self.target = target
		self.source = source

class Lod(Instruction):
	def __init__(self, target: Register, source: int | Register | Label) -> None:
		super().__init__("mov", target, AddressedValue(source))
		self.target = target
		self.source = source

class Str(Instruction):
	def __init__(self, target: int | Register | Label, source: int | Register) -> None:
		super().__init__("mov", AddressedValue(target), source)
		self.target = target
		self.source = source

class RepMovsb(Instruction):
	def __init__(self) -> None:
		super().__init__("rep movsb")

class Jmp(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jmp", target)
		self.target = target

class Jz(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jz", target)
		self.target = target

class Jnz(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jnz", target)
		self.target = target

class Je(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("je", target)
		self.target = target

class Jne(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jne", target)
		self.target = target

class Jl(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jl", target)
		self.target = target

class Jle(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jle", target)
		self.target = target

class Jg(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jg", target)
		self.target = target

class Jge(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jge", target)
		self.target = target

class Jb(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jb", target)
		self.target = target

class Jbe(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jbe", target)
		self.target = target

class Ja(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("ja", target)
		self.target = target

class Jae(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("jae", target)
		self.target = target

class Call(Instruction):
	def __init__(self, target: int | Register | Label) -> None:
		super().__init__("call", target)
		self.target = target

class Ret(Instruction):
	def __init__(self) -> None:
		super().__init__("ret")

class Intr(Instruction):
	def __init__(self, interrupt: int) -> None:
		super().__init__("int", interrupt)
		self.interrupt = interrupt

class MarkLabel(Instruction):
	def __init__(self, label: Label) -> None:
		super().__init__("marklabel", label)

class Raw(Instruction):
	def __init__(self, raw: str) -> None:
		super().__init__("raw", str(raw))

class Volatile(Instruction):
	def __init__(self) -> None:
		super().__init__("raw")

class Comment(Instruction):
	def __init__(self, comment: str) -> None:
		super().__init__("raw", f";{comment}")

class BlockComment(Comment):
	def __init__(self, comment: str) -> None: super().__init__(comment)

class StatementComment(BlockComment):
	def __init__(self, stmt: ast.stmt) -> None:
		super().__init__(type(stmt).__name__)

class ExpressionComment(BlockComment):
	def __init__(self, expr: ast.expr) -> None:
		super().__init__(ExpressionComment.get_readable_expression(expr))
	
	@staticmethod
	def get_readable_expression(expr: ast.expr) -> str:
		if isinstance(expr, ast.Name):
			return expr.id
		elif isinstance(expr, ast.Constant):
			return str(expr.value)
		elif isinstance(expr, ast.Subscript):
			return f"{ExpressionComment.get_readable_expression(expr.value)}[{ExpressionComment.get_readable_expression(expr.slice)}]"
		elif isinstance(expr, ast.BinOp):
			return f"{type(expr.op).__name__}({ExpressionComment.get_readable_expression(expr.left)}, {ExpressionComment.get_readable_expression(expr.right)})"
		else:
			return type(expr).__name__

class AddressOfExpressionComment(BlockComment):
	def __init__(self, expr: ast.expr) -> None:
		super().__init__(f"addressof {ExpressionComment.get_readable_expression(expr)}")

class CastComment(BlockComment):
	def __init__(self, from_type: "Type", to_type: "Type") -> None: super().__init__(f"{from_type} -> {to_type}")

class EndBlockComment(Comment):
	def __init__(self) -> None: super().__init__("end")

class Context:
	def prologue(self): ...
	def get_word_size(self) -> int: raise NotImplementedError()
	def get_word_mask(self) -> int: raise NotImplementedError()
	def get_pointer_size(self) -> int: return self.get_word_size() + (self.get_segment_selector_size() if self.is_using_segments() else 0)
	def is_using_segments(self) -> bool: raise NotImplementedError()
	def get_segment_selector_size(self) -> int: raise NotImplementedError()
	def get_data_segment(self) -> int: raise NotImplementedError()
	def create_data_label(self, label: Label) -> Instruction: raise NotImplementedError()

class X8616Context(Context):
	def __init__(self, org: int = 0x7C00) -> None: self.org = org
	
	def prologue(self):
		yield Instruction("bits", 16)
		yield Instruction("org", self.org)
	
	def get_word_size(self) -> int: return 2
	def get_word_mask(self) -> int: return 0xFFFF
	def is_using_segments(self) -> bool: return True
	def get_segment_selector_size(self) -> int: return 2
	def get_data_segment(self) -> int: return ((self.org >> 16) << 12) & 0xFFFF
	def create_data_label(self, label: Label) -> Instruction: return Instruction("dw", label, (self.org >> 16) << 12)

class Type:
	def __init__(self, name: str = "$unset", subtypes: "list[Type]" = []) -> None:
		self.name = name
		self.subtypes = subtypes
	
	def get_name(self) -> str: return self.name
	
	def copy(self) -> "Type":
		copy = type(self)()
		copy.subtypes = [type.copy() for type in self.subtypes]
		return copy
	
	def retype(self, subtypes: "list[Type]") -> "Type": return self.copy()
	
	def is_signed(self) -> bool: return False
	
	def is_assignable_to(self, context: Context, other: "Type") -> bool: return self == other
	
	def assign_to(self, context: Context, origin: ast.AST, target: "Type"):
		if self != target: compile_error(TypeError, f"{self} can not be assigned to {target}.", origin)
		else: return []
	
	def is_castable_from(self, context: Context, source: "Type") -> bool: return source.is_assignable_to(context, self)
	
	def cast_from(self, context: Context, origin: ast.AST, source: "Type"): yield source.assign_to(context, origin, self)
	
	def get_size(self, context: Context) -> int: raise NotImplementedError()
	
	def get_stack_size(self, context: Context) -> int:
		size = self.get_size(context)
		remainder = size % context.get_word_size()
		if remainder == 0: return size
		else: return (size + context.get_word_size()) - remainder
	
	def get_mask(self, context: Context) -> int: return (1 << (self.get_size(context) * 8)) - 1
	def get_stack_mask(self, context: Context) -> int: return (1 << (self.get_stack_size(context) * 8)) - 1
	def get_upper_mask(self, context: Context) -> int: return (1 << ((self.get_size(context) % context.get_word_size()) * 8)) - 1
	
	def __eq__(self, __o: object) -> bool:
		return isinstance(__o, Type) and __o.name == self.name and len(__o.subtypes) == len(self.subtypes) and all([__o.subtypes[i] == self.subtypes[i] for i in range(len(self.subtypes))])
	
	def __ne__(self, __o: object) -> bool: return not (self == __o)
	
	def __repr__(self) -> str:
		return self.name + (f"[{','.join([str(type) for type in self.subtypes])}]" if len(self.subtypes) > 0 else "")

class Void(Type):
	def __init__(self) -> None: super().__init__("void")
	def get_size(self, context: Context) -> int: return 0

class Bool(Type):
	def __init__(self) -> None: super().__init__("bool")
	def get_size(self, context: Context) -> int: return 1

class IntegerPrimitive(Type):
	def __init__(self, name: str) -> None: super().__init__(name)
	
	def is_assignable_to(self, context: Context, target: "Type") -> bool:
		return isinstance(target, IntegerPrimitive) and target.is_signed() == self.is_signed() and target.get_stack_size(context) >= self.get_stack_size(context)
	
	def assign_to(self, context: Context, origin: ast.AST, target: "Type"):
		yield CastComment(self, target)
		
		if target.get_stack_size(context) > self.get_stack_size(context):
			ADDITIONAL_CHARS = target.get_stack_size(context) - self.get_stack_size(context)
			ADDITIONAL_WORDS = ADDITIONAL_CHARS // context.get_word_size()
			
			yield Mov(SI(), SP())
			yield Sub(SI(), ADDITIONAL_CHARS)
			for _ in range(self.get_stack_size(context) // context.get_word_size()):
				yield Pop(AX())
				yield Str(SI(), AX())
				yield Add(SI(), context.get_word_size())
			if self.is_signed():
				yield And(AX(), 1 << ((context.get_word_size() * 8) - 1))
				yield Mov(CL(), 7)
				yield Sar(AH(), CL())
				yield Mov(AL(), AH())
			else:
				yield Xor(AX(), AX())
			for _ in range(ADDITIONAL_WORDS):
				yield Push(AX())
			yield Sub(SP(), self.get_stack_size(context))
		
		yield EndBlockComment()

class SignedInteger(IntegerPrimitive):
	def __init__(self, name: str) -> None: super().__init__(name)
	def is_signed(self) -> bool: return True

class UnsignedInteger(IntegerPrimitive):
	def __init__(self, name: str) -> None: super().__init__(name)
	def is_signed(self) -> bool: return False

class Int(SignedInteger):
	def __init__(self) -> None: super().__init__("int")
	def get_size(self, context: Context) -> int: return context.get_word_size()

class Int8(SignedInteger):
	def __init__(self) -> None: super().__init__("int8")
	def get_size(self, context: Context) -> int: return 1

class Int16(SignedInteger):
	def __init__(self) -> None: super().__init__("int16")
	def get_size(self, context: Context) -> int: return 2

class Int32(SignedInteger):
	def __init__(self) -> None: super().__init__("int32")
	def get_size(self, context: Context) -> int: return 4

class Int64(SignedInteger):
	def __init__(self) -> None: super().__init__("int64")
	def get_size(self, context: Context) -> int: return 8

class UInt(UnsignedInteger):
	def __init__(self) -> None: super().__init__("uint")
	def get_size(self, context: Context) -> int: return context.get_word_size()

class UInt8(UnsignedInteger):
	def __init__(self) -> None: super().__init__("uint8")
	def get_size(self, context: Context) -> int: return 1

class UInt16(UnsignedInteger):
	def __init__(self) -> None: super().__init__("uint16")
	def get_size(self, context: Context) -> int: return 2

class UInt32(UnsignedInteger):
	def __init__(self) -> None: super().__init__("uint32")
	def get_size(self, context: Context) -> int: return 4

class UInt64(UnsignedInteger):
	def __init__(self) -> None: super().__init__("uint64")
	def get_size(self, context: Context) -> int: return 8

class DataPtr(Type):
	def __init__(self, type: Type = Void()) -> None: super().__init__("ptr", [type])
	
	def retype(self, subtypes: "list[Type]") -> "Type":
		if len(subtypes) != 1: compile_error(ValueError, "Incorrect number of subtypes for data pointer.")
		return DataPtr(subtypes[0])
	
	def is_castable_from(self, context: Context, source: "Type") -> bool:
		return (isinstance(source, UnsignedInteger) and source.get_stack_size(context) == self.get_stack_size(context)) or super().is_castable_from(context, source)
	
	def cast_from(self, context: Context, origin: ast.AST, source: "Type"):
		if isinstance(source, UnsignedInteger) and source.get_stack_size(context) == self.get_stack_size(context): pass
		else: yield super().cast_from(context, origin, source)
	
	def get_size(self, context: Context) -> int: return context.get_pointer_size()
	def get_value_type(self) -> Type: return self.subtypes[0]

class FunctionPtr(Type):
	def __init__(self, return_type: Type = Void(), arguments: list[Type] = []) -> None: super().__init__("func", arguments + [return_type])
	
	def retype(self, subtypes: "list[Type]") -> "Type":
		if len(subtypes) < 1: compile_error(ValueError, "Incorrect number of subtypes for function pointer.")
		return FunctionPtr(subtypes[len(subtypes) - 1], subtypes[:len(subtypes) - 1])
	
	def get_size(self, context: Context) -> int: return context.get_pointer_size()
	def get_argument_types(self) -> list[Type]: return self.subtypes[:len(self.subtypes) - 1]
	def get_return_type(self) -> Type: return self.subtypes[len(self.subtypes) - 1]

class Struct(Type):
	def __init__(self, fields: list[Type] = [], names: list[str | None] | None = None) -> None:
		super().__init__("struct", fields)
		self.names = [f"_{i}" if names == None or cast(list[str | None], names)[i] == None else ensure(cast(list[str | None], names)[i]) for i in range(len(fields))]
	
	def retype(self, subtypes: "list[Type]") -> "Type": return Struct(subtypes)
	def get_size(self, context: Context) -> int: return sum([subtype.get_stack_size(context) for subtype in self.subtypes])
	def get_fields(self) -> list[Type]: return self.subtypes
	def get_names(self) -> list[str]: return self.names
	def get_field_index_from_name(self, name: str) -> int | None:
		if name.startswith("_"):
			try:
				field_index = int(name[1:])
				if field_index >= 0 and field_index < len(self.get_fields()): return field_index
			except: pass
		if name in self.names: return self.names.index(name)
		return None

DEFAULT_TYPES = {
	"void": Void(),
	"bool": Bool(),
	"uint8": UInt8(),
	"uint16": UInt16(),
	"uint32": UInt32(),
	"uint64": UInt64(),
	"uint": UInt(),
	"int8": Int8(),
	"int16": Int16(),
	"int32": Int32(),
	"int64": Int64(),
	"int": Int(),
	"ptr": DataPtr(),
	"func": FunctionPtr()
}

all_types = DEFAULT_TYPES.copy()

def get_type_from_name(name: ast.expr | str) -> Type:
	origin = name if isinstance(name, ast.AST) else None
	
	if isinstance(name, ast.Subscript):
		if isinstance(name.slice, ast.Tuple):
			return get_type_from_name(name.value).retype([get_type_from_name(subtype) for subtype in name.slice.elts])
		else:
			return get_type_from_name(name.value).retype([get_type_from_name(name.slice)])
	
	if isinstance(name, ast.Name): name = name.id
	if isinstance(name, str):
		if name in all_types: return all_types[name]
		else: compile_error(TypeError, f"Undefined type \"{name}\".", origin)
	compile_error(TypeError, f"Invalid type expression.", origin)

class Variable:
	def __init__(self, type: Type) -> None: self.type = type
	def get_type(self) -> Type: return self.type
	def is_value_only(self) -> bool: return False
	def push_address(self, context: Context, func: ast.FunctionDef, locals: "list[Local]"): ...
	def push_value(self, context: Context, func: ast.FunctionDef, locals: "list[Local]"): ...

class Argument(Variable):
	def __init__(self, index: int, type: Type) -> None:
		super().__init__(type)
		self.index = index
	
	def push_address(self, context: Context, func: ast.FunctionDef, locals: "list[Local]"):
		offset = sum([get_type_from_name(ensure(func.args.args[i].annotation)).get_stack_size(context) for i in range(len(func.args.args) - 1, self.index - 1, -1)])
		if context.is_using_segments(): yield Push(SS())
		yield Mov(SI(), BP())
		yield Add(SI(), offset + context.get_word_size())
		yield Push(SI())

class Local(Variable):
	def __init__(self, type: Type, name: str) -> None:
		super().__init__(type)
		self.name = name
	
	def push_address(self, context: Context, func: ast.FunctionDef, locals: "list[Local]"):
		offset = sum([locals[i].get_type().get_stack_size(context) for i in range(locals.index(self))]) + context.get_word_size()
		if context.is_using_segments(): yield Push(SS())
		yield Mov(SI(), BP())
		yield Sub(SI(), offset)
		yield Push(SI())

class Global(Variable):
	def __init__(self, type: Type, name: str, value: int | Label) -> None:
		super().__init__(type)
		self.name = name
		self.label = Label(name)
		self.value = value
	
	def push_address(self, context: Context, func: ast.FunctionDef, locals: "list[Local]"):
		if context.is_using_segments(): yield Push(context.get_data_segment())
		yield Push(self.label)
	
	def emit(self, context: Context):
		yield MarkLabel(self.label)
		if isinstance(self.value, int):
			yield DataByte([((self.value >> (i * 8)) & 0xFF) for i in range(self.get_type().get_stack_size(context))])
		else:
			yield context.create_data_label(self.value)

class StringConstant(Variable):
	def __init__(self, value: str) -> None:
		super().__init__(Struct([UInt8()] * (len(value) + 1)))
		self.value = value
		self.label = Label()
	
	def push_address(self, context: Context, func: ast.FunctionDef, locals: "list[Local]"):
		yield Push(self.label)
	
	def emit(self, context: Context):
		yield MarkLabel(self.label)
		yield DataByte([(ord(c) & 0xFF) for c in self.value])

class Function(Variable):
	def __init__(self, func: ast.FunctionDef) -> None:
		for i in range(len(func.args.args)):
			if func.args.args[i].annotation == None: compile_error(TypeError, "Missing type specifier.", func)
		if func.returns == None: compile_error(TypeError, "Missing type specifier.", func)
		
		super().__init__(FunctionPtr(get_type_from_name(ensure(func.returns)), [get_type_from_name(ensure(arg.annotation)) for arg in func.args.args]))
		self.func = func
	
	def is_value_only(self) -> bool: return True
	def push_address(self, context: Context, func: ast.FunctionDef, locals: "list[Local]"): raise NotImplementedError()
	def push_value(self, context: Context, func: ast.FunctionDef, locals: "list[Local]"): yield Push(Label(self.func.name))

def generate_function(context: Context, func: ast.FunctionDef, functions: list[Function], globals: list[Global], strings: list[StringConstant], user_types: list[Type]):
	def get_locals(stmt: ast.stmt):
		if isinstance(stmt, ast.FunctionDef):
			for child in stmt.body:
				for local in get_locals(child): yield local
		elif isinstance(stmt, ast.AnnAssign):
			if isinstance(stmt.target, ast.Name) and isinstance(stmt.annotation, ast.Name):
				yield Local(get_type_from_name(stmt.annotation), stmt.target.id)
		elif isinstance(stmt, ast.For):
			for child in stmt.body:
				for local in get_locals(child): yield local
		elif isinstance(stmt, ast.If):
			for child in stmt.body:
				for local in get_locals(child): yield local
			for child in stmt.orelse:
				for local in get_locals(child): yield local
		elif isinstance(stmt, ast.While):
			for child in stmt.body:
				for local in get_locals(child): yield local
	
	locals: list[Local] = [local for local in get_locals(func)]
	
	def resolve_name(name: str) -> Variable:
		arg_names = [arg.arg for arg in func.args.args]
		if name in arg_names:
			index = arg_names.index(name)
			return Argument(index, get_type_from_name(ensure(func.args.args[index].annotation)))
		else:
			for local in locals:
				if local.name == name: return local
			for global_var in globals:
				if global_var.name == name: return global_var
			for function in functions:
				if function.func.name == name: return function
			compile_error(ValueError, f"{name} is not defined.", func)
	
	def get_expression_type(expr: ast.expr) -> Type:
		if isinstance(expr, ast.Constant):
			if isinstance(expr.value, bool): return Bool()
			elif isinstance(expr.value, int):
				if expr.value < 0:
					if expr.value >= -128: return Int8()
					elif expr.value >= -32768: return Int16()
					elif expr.value >= -2147483648: return Int32()
					elif expr.value >= -9223372036854775808: return Int64()
					else: compile_error(ValueError, f"{expr.value} is not a valid integer.", expr)
				else:
					if expr.value <= 255: return UInt8()
					elif expr.value <= 65535: return UInt16()
					elif expr.value <= 4294967295: return UInt32()
					elif expr.value <= 18446744073709551615: return UInt64()
					else: compile_error(ValueError, f"{expr.value} is not a valid integer.", expr)
			else: compile_error(TypeError, "Invalid constant expression.", expr)
		elif isinstance(expr, ast.Str):
			return DataPtr(UInt8())
		elif isinstance(expr, ast.Name):
			return resolve_name(expr.id).get_type()
		elif isinstance(expr, ast.Attribute):
			struct = get_expression_type(expr.value)
			if isinstance(struct, Struct):
				field_index = struct.get_field_index_from_name(expr.attr)
				if field_index != None: return struct.get_fields()[ensure(field_index)]
			compile_error(TypeError, f"{struct} has no field named {expr.attr}", expr)
		elif isinstance(expr, ast.Subscript):
			pointer = get_expression_type(expr.value)
			if not isinstance(pointer, DataPtr): compile_error(TypeError, "Subscript requires pointer type.", expr)
			return pointer.get_value_type()
		elif isinstance(expr, ast.BinOp):
			bin_a: Type = get_expression_type(expr.left)
			bin_b: Type = get_expression_type(expr.right)
			if not bin_b.is_assignable_to(context, bin_a): compile_error(TypeError, f"{expr.op.__class__.__name__}({bin_a}, {bin_b}) is not defined.", expr)
			return bin_a
		elif isinstance(expr, ast.Compare):
			if len(expr.comparators) > 1 or len(expr.ops) > 1: compile_error(ValueError, "Combined comparisons are not supported.", expr)
			cmp_a: Type = get_expression_type(expr.left)
			cmp_b: Type = get_expression_type(expr.comparators[0])
			if not cmp_b.is_assignable_to(context, cmp_a): compile_error(ValueError, f"{expr.ops[0].__class__.__name__}({cmp_a}, {cmp_b}) is not defined.", expr)
			return cmp_a
		elif isinstance(expr, ast.BoolOp):
			if not all([get_expression_type(value).get_stack_size(context) == context.get_word_size() for value in expr.values]):
				compile_error(ValueError, f"{expr.op}({', '.join([str(get_expression_type(value)) for value in expr.values])}) is not defined.", expr)
			return Bool()
		elif isinstance(expr, ast.Call):
			if isinstance(expr.func, ast.Name):
				if expr.func.id == "bios":
					return Struct([UInt16() for _ in filter(lambda keyword: keyword.arg != "i", expr.keywords)])
				elif expr.func.id in DEFAULT_TYPES:
					return DEFAULT_TYPES[expr.func.id]
			
			ptr = get_expression_type(expr.func)
			if not isinstance(ptr, FunctionPtr): compile_error(TypeError, "Invalid function expression.", expr.func)
			return ptr.get_return_type()
		else:
			compile_error(TypeError, "Expression has no type.", expr)
	
	def perform_op(op: ast.operator | ast.cmpop, signed: bool):
		if isinstance(op, ast.Add):
			yield Pop(AX())
			yield Pop(SI())
			yield Add(AX(), SI())
			yield Push(AX())
		elif isinstance(op, ast.Sub):
			yield Pop(SI())
			yield Pop(AX())
			yield Sub(AX(), SI())
			yield Push(AX())
		elif isinstance(op, ast.Mult):
			yield Pop(SI())
			yield Pop(AX())
			yield Xor(DX(), DX())
			yield IMul(SI()) if signed else Mul(SI())
			yield Push(AX())
		elif isinstance(op, ast.Div) or isinstance(op, ast.Mod):
			yield Pop(SI())
			yield Pop(AX())
			yield Xor(DX(), DX())
			yield IDiv(SI()) if signed else Div(SI())
			yield Push(DX() if isinstance(op, ast.Mod) else AX())
		elif isinstance(op, ast.BitAnd):
			yield Pop(SI())
			yield Pop(AX())
			yield And(AX(), SI())
			yield Push(AX())
		elif isinstance(op, ast.BitOr):
			yield Pop(SI())
			yield Pop(AX())
			yield Or(AX(), SI())
			yield Push(AX())
		elif isinstance(op, ast.BitXor):
			yield Pop(SI())
			yield Pop(AX())
			yield Xor(AX(), SI())
			yield Push(AX())
		elif isinstance(op, ast.Eq):
			on_false = Label()
			end = Label()
			yield Pop(SI())
			yield Pop(AX())
			yield Cmp(SI(), AX())
			yield Jne(on_false)
			yield Push(1)
			yield Jmp(end)
			yield MarkLabel(on_false)
			yield Push(0)
			yield MarkLabel(end)
		elif isinstance(op, ast.NotEq):
			on_false = Label()
			end = Label()
			yield Pop(SI())
			yield Pop(AX())
			yield Cmp(SI(), AX())
			yield Je(on_false)
			yield Push(1)
			yield Jmp(end)
			yield MarkLabel(on_false)
			yield Push(0)
			yield MarkLabel(end)
		elif isinstance(op, ast.Lt):
			on_false = Label()
			end = Label()
			yield Pop(SI())
			yield Pop(AX())
			yield Cmp(SI(), AX())
			yield (Jge if signed else Jae)(on_false)
			yield Push(1)
			yield Jmp(end)
			yield MarkLabel(on_false)
			yield Push(0)
			yield MarkLabel(end)
		elif isinstance(op, ast.LtE):
			on_false = Label()
			end = Label()
			yield Pop(SI())
			yield Pop(AX())
			yield Cmp(SI(), AX())
			yield (Jg if signed else Ja)(on_false)
			yield Push(1)
			yield Jmp(end)
			yield MarkLabel(on_false)
			yield Push(0)
			yield MarkLabel(end)
		elif isinstance(op, ast.Gt):
			on_false = Label()
			end = Label()
			yield Pop(SI())
			yield Pop(AX())
			yield Cmp(SI(), AX())
			yield (Jle if signed else Jbe)(on_false)
			yield Push(1)
			yield Jmp(end)
			yield MarkLabel(on_false)
			yield Push(0)
			yield MarkLabel(end)
		elif isinstance(op, ast.GtE):
			on_false = Label()
			end = Label()
			yield Pop(SI())
			yield Pop(AX())
			yield Cmp(SI(), AX())
			yield (Jl if signed else Jb)(on_false)
			yield Push(1)
			yield Jmp(end)
			yield MarkLabel(on_false)
			yield Push(0)
			yield MarkLabel(end)
		else:
			compile_error(TypeError, "Invalid operator.", op)
	
	def push_value(expr: ast.expr):
		yield ExpressionComment(expr)
		if isinstance(expr, ast.Constant):
			if isinstance(expr.value, bool):
				yield Push(1 if expr.value else 0)
			elif isinstance(expr.value, int):
				constant_words = (get_expression_type(expr).get_stack_size(context) + 1) // context.get_word_size()
				for i in range(constant_words):
					yield Push(expr.value >> (((constant_words - 1) - i) * context.get_word_size() * 8))
			else:
				compile_error(TypeError, "Unsupported value type.", expr)
		elif isinstance(expr, ast.Str):
			const = StringConstant(expr.value)
			strings.append(const)
			yield Push(const.label)
		elif isinstance(expr, ast.BinOp):
			target_type = get_expression_type(expr)
			left_type = get_expression_type(expr.left)
			right_type = get_expression_type(expr.right)
			if not left_type.is_assignable_to(context, target_type): compile_error(ValueError, f"{expr.op.__class__.__name__}({left_type}, {right_type}) is not defined.", expr)
			if not right_type.is_assignable_to(context, target_type): compile_error(ValueError, f"{expr.op.__class__.__name__}({left_type}, {right_type}) is not defined.", expr)
			yield push_value(expr.left)
			yield left_type.assign_to(context, expr, target_type)
			yield push_value(expr.right)
			yield right_type.assign_to(context, expr, target_type)
			yield perform_op(expr.op, target_type.is_signed())
		elif isinstance(expr, ast.Compare):
			cmp_a: Type = get_expression_type(expr.left)
			cmp_b: Type = get_expression_type(expr.comparators[0])
			target_type = None
			if cmp_a.is_assignable_to(context, cmp_b): target_type = cmp_b
			elif cmp_b.is_assignable_to(context, cmp_a): target_type = cmp_a
			if target_type != None and ensure(target_type).get_stack_size(context) != context.get_word_size(): target_type = None
			if target_type == None: compile_error(ValueError, f"{expr.ops[0].__class__.__name__}({cmp_a}, {cmp_b}) is not defined.", expr)
			yield push_value(expr.left)
			yield cmp_a.assign_to(context, expr, ensure(target_type))
			yield push_value(expr.comparators[0])
			yield cmp_b.assign_to(context, expr, ensure(target_type))
			yield perform_op(expr.ops[0], ensure(target_type).is_signed())
		elif isinstance(expr, ast.Call):
			if isinstance(expr.func, ast.Name) and expr.func.id == "bios":
				for keyword in expr.keywords:
					if keyword.arg == None: compile_error(TypeError, "Invalid keyword argument.", keyword)
				yield bios(expr, **dict([(ensure(keyword.arg), keyword.value) for keyword in expr.keywords]))
			else:
				yield Sub(SP(), get_expression_type(expr).get_stack_size(context))
				for arg in expr.args: yield push_value(arg)
				yield push_value(expr.func)
				yield Pop(AX())
				yield Call(AX())
				yield Add(SP(), sum([get_expression_type(arg).get_stack_size(context) for arg in expr.args]))
		elif isinstance(expr, ast.Name):
			var = resolve_name(expr.id)
			if var.is_value_only(): yield var.push_value(context, func, locals)
			else:
				yield var.push_address(context, func, locals)
				yield read_address(var.get_type())
		else:
			yield push_address(expr)
			yield read_address(get_expression_type(expr))
		yield EndBlockComment()
	
	def push_address(expr: ast.expr):
		yield AddressOfExpressionComment(expr)
		if isinstance(expr, ast.Name):
			var = resolve_name(expr.id)
			if var.is_value_only(): compile_error(TypeError, "Expression is not addressable.", expr)
			yield var.push_address(context, func, locals)
		elif isinstance(expr, ast.Subscript):
			pointer = get_expression_type(expr.value)
			index = get_expression_type(expr.slice)
			if index.get_stack_size(context) != context.get_word_size(): compile_error(TypeError, f"{index} can not be used as an index offset.", expr.slice)
			if not isinstance(pointer, DataPtr): compile_error(TypeError, "Subscript requires pointer type.", expr.value)
			yield push_value(expr.value)
			yield push_value(expr.slice)
			yield Pop(AX())
			yield Mov(SI(), pointer.get_value_type().get_size(context))
			yield Mul(SI())
			yield Pop(SI())
			yield Add(SI(), AX())
			yield Push(SI())
		elif isinstance(expr, ast.Attribute):
			yield push_address(expr.value)
			struct = get_expression_type(expr.value)
			if not isinstance(struct, Struct): compile_error(TypeError, f"{struct} has no fields.", expr.value)
			field_index = struct.get_field_index_from_name(expr.attr)
			if field_index == None: compile_error(TypeError, f"{struct} has no field named {expr.attr}.", expr)
			fields = struct.get_fields()
			yield Pop(SI())
			yield Add(SI(), sum([fields[i].get_size(context) for i in range(ensure(field_index))]))
			yield Push(SI())
		else:
			compile_error(TypeError, "Invalid address expression.", expr)
		yield EndBlockComment()
	
	def read_address(type: Type):
		yield BlockComment(f"load {type}")
		yield Pop(SI())
		if context.is_using_segments(): yield Pop(DS())
		yield Add(SI(), type.get_stack_size(context) - context.get_word_size())
		yield Lod(AX(), SI())
		if type.get_size(context) % context.get_word_size() != 0:
			yield And(AX(), type.get_upper_mask(context))
		yield Push(AX())
		for _ in range((type.get_stack_size(context) // context.get_word_size()) - 1):
			yield Sub(SI(), context.get_word_size())
			yield Lod(AX(), SI())
			yield Push(AX())
		yield EndBlockComment()
	
	def write_address(type: Type):
		yield BlockComment(f"store {type}")
		if type.get_size(context) == context.get_word_size():
			yield Pop(AX())
			if type.get_size(context) < context.get_word_size(): yield And(AX(), type.get_upper_mask(context))
			yield Pop(SI())
			if context.is_using_segments(): yield Pop(DS())
			yield Str(SI(), AX())
		else:
			yield Volatile()
			yield Add(SP(), type.get_stack_size(context))
			yield Pop(DI())
			if context.is_using_segments(): yield Pop(ES())
			yield Mov(SI(), SP())
			yield Sub(SI(), context.get_word_size() + context.get_segment_selector_size())
			if context.is_using_segments():
				yield Push(SS())
				yield Pop(DS())
			yield Mov(CX(), type.get_size(context))
			yield RepMovsb()
		yield EndBlockComment()
	
	def bios(origin: ast.AST, i: ast.expr | None = None, **kwargs: ast.expr):
		if not isinstance(i, ast.expr): compile_error(TypeError, "BIOS call requires interrupt number.", origin)
		if not (isinstance(i, ast.Constant) and isinstance(i.value, int) and (i.value >= 0x00 and i.value <= 0xFF)): compile_error(TypeError, "BIOS call interrupt number must be a constant integer between 0x00 and 0xFF.", i)
		for key in kwargs.keys():
			if not key in REGISTERS: compile_error(TypeError, f"\"{key}\" is not a register.", kwargs[key])
			yield push_value(kwargs[key])
		for key in reversed(kwargs.keys()): yield Pop(REGISTERS[key])
		yield Intr(i.value)
		for key in kwargs.keys(): yield Push(REGISTERS[key])
	
	yield MarkLabel(Label(func.name))
	yield Comment("enter")
	yield Push(BP())
	yield Mov(BP(), SP())
	yield Sub(SP(), sum([local.get_type().get_stack_size(context) for local in locals]))
	
	def yield_statement(stmt: ast.stmt):
		yield StatementComment(stmt)
		if isinstance(stmt, ast.Pass): pass
		elif isinstance(stmt, ast.AnnAssign):
			if stmt.value != None:
				source_type = get_expression_type(ensure(stmt.value))
				target_type = get_type_from_name(stmt.annotation)
				if not source_type.is_assignable_to(context, target_type): compile_error(TypeError, f"{source_type} can not be assigned to {target_type}.", stmt)
				yield push_address(stmt.target)
				yield push_value(ensure(stmt.value))
				yield source_type.assign_to(context, stmt, target_type)
				yield write_address(get_expression_type(stmt.target))
		elif isinstance(stmt, ast.Assign):
			if len(stmt.targets) > 1: compile_error(ValueError, "Assigning to multiple targets is not supported.", stmt)
			target = stmt.targets[0]
			source_type = get_expression_type(stmt.value)
			target_type = get_expression_type(target)
			if not source_type.is_assignable_to(context, target_type): compile_error(TypeError, f"{source_type} can not be assigned to {target_type}.", stmt)
			yield push_address(target)
			yield push_value(stmt.value)
			yield source_type.assign_to(context, stmt, target_type)
			yield write_address(get_expression_type(target))
		elif isinstance(stmt, ast.AugAssign):
			source_type = get_expression_type(stmt.value)
			target_type = get_expression_type(stmt.target)
			if not source_type.is_assignable_to(context, target_type): compile_error(TypeError, f"{source_type} can not be assigned to {target_type}.", stmt)
			yield push_address(stmt.target)
			yield push_value(stmt.target)
			yield push_value(stmt.value)
			yield source_type.assign_to(context, stmt, target_type)
			yield perform_op(stmt.op, get_expression_type(stmt.value).is_signed())
			yield write_address(get_expression_type(stmt.target))
		elif isinstance(stmt, ast.Expr):
			if isinstance(stmt.value, ast.Str):
				yield Raw(stmt.value.value)
			else:
				yield push_value(stmt.value)
				yield Add(SP(), get_expression_type(stmt.value).get_stack_size(context))
		elif isinstance(stmt, ast.If):
			if get_expression_type(stmt.test).get_stack_size(context) > context.get_word_size(): compile_error(TypeError, f"{get_expression_type(stmt.test)} can not be used as a condition.", stmt.test)
			else_body = Label()
			end = Label()
			
			yield push_value(stmt.test)
			yield Pop(AX())
			yield Cmp(AX(), 0)
			yield Jz(end if len(stmt.orelse) == 0 else else_body)
			
			for child in stmt.body: yield yield_statement(child)
			
			if len(stmt.orelse) > 0:
				yield Jmp(end)
				yield MarkLabel(else_body)
				for child in stmt.orelse: yield yield_statement(child)
			
			yield MarkLabel(end)
		elif isinstance(stmt, ast.While):
			if get_expression_type(stmt.test).get_stack_size(context) > context.get_word_size(): compile_error(TypeError, f"{get_expression_type(stmt.test)} can not be used as a condition.", stmt.test)
			start = Label()
			end = Label()
			yield MarkLabel(start)
			
			yield push_value(stmt.test)
			yield Pop(AX())
			yield Cmp(AX(), 0)
			yield Jz(end)
			
			for child in stmt.body: yield yield_statement(child)
			
			yield Jmp(start)
			yield MarkLabel(end)
		else:
			compile_error(TypeError, "Unsupported parse tree node.", stmt)
		yield EndBlockComment()
	
	for stmt in func.body: yield yield_statement(stmt)
	
	yield Comment("leave")
	yield Mov(SP(), BP())
	yield Pop(BP())
	yield Ret()

def generate_structure(class_def: ast.ClassDef) -> None:
	fields: list[Type] = []
	names: list[str | None] = []
	for field in class_def.body:
		if not (isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)): compile_error(TypeError, "Invalid structure field.", field)
		if field.value != None: compile_error(TypeError, "Structure field initialization is not supported.", field)
		fields.append(get_type_from_name(field.annotation))
		names.append(None if field.target.id == "_" else field.target.id)
	
	all_types[class_def.name] = Struct(fields, names)

def generate_module(context: Context, module: ast.Module):
	for class_def in filter(lambda node: isinstance(node, ast.ClassDef), module.body): generate_structure(cast(ast.ClassDef, class_def))
	functions = [Function(cast(ast.FunctionDef, func)) for func in filter(lambda node: isinstance(node, ast.FunctionDef), module.body)]
	
	globals: list[Global] = []
	strings: list[StringConstant] = []
	user_types: list[Type] = []
	
	for node in module.body:
		if isinstance(node, ast.ClassDef): pass
		elif isinstance(node, ast.FunctionDef):
			yield generate_function(context, node, functions, globals, strings, user_types)
		elif isinstance(node, ast.AnnAssign):
			if not isinstance(node.target, ast.Name): compile_error(TypeError, "Invalid global name.", node)
			
			type: Type = get_type_from_name(node.annotation)
			value: Label | int
			if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
				if node.value.value >= (1 << ((type.get_size(context) * 8) - (1 if type.is_signed() else 0))) or (node.value.value < ((type.get_size(context) * -8) if type.is_signed() else 0)): compile_error(TypeError, f"{node.value.value} can not be assigned to global of type {type}.", node.value)
				value = node.value.value
			elif isinstance(node.value, ast.Str):
				const = StringConstant(node.value.value)
				value = const.label
				strings.append(const)
			else: compile_error(TypeError, "Invalid global value.", node)
			
			globals.append(Global(type, node.target.id, value))
		elif isinstance(node, ast.ImportFrom):
			if not node.module in ["asmtypes"]:
				compile_error(NameError, f"{node.module} does not exist.", node)
		else:
			compile_error(TypeError, "Unsupported parse tree node.", node)
	
	for global_var in globals: yield global_var.emit(context)
	
	for const in strings: yield const.emit(context)

class OperandMatcher:
	def __init__(self, any: bool = True, immediate: bool = False, register: bool = False, label: bool = False, dereference: bool = False, segment: bool = False, custom = lambda context, operand: True) -> None:
		self.immediate = immediate or any
		self.register = register or any
		self.label = label or any
		self.dereference = dereference or any
		self.segment = segment or any
		self.custom = custom
	
	def is_match(self, context: Context, operand) -> bool:
		return ((self.immediate and isinstance(operand, int)) or (self.register and isinstance(operand, Register)) or (self.label and isinstance(operand, Label)) or (self.dereference and isinstance(operand, AddressedValue))) and self.custom(context, operand)

class InstructionMatcher:
	def __init__(self, op_type: type[Instruction] | None = None, operands: list[OperandMatcher] | None = None, custom = lambda context, instruction: True) -> None:
		self.op_type = op_type
		self.operands = operands
		self.custom = custom
	
	def is_match(self, context: Context, instruction: Instruction) -> bool:
		return (self.op_type == None or type(instruction) == self.op_type) and ((self.operands == None) or ((len(instruction.operands) == len(ensure(self.operands))) and all([ensure(self.operands)[i].is_match(context, instruction.operands[i]) for i in range(len(ensure(self.operands)))]))) and self.custom(context, instruction)

class InstructionReplacement:
	def __init__(self, start: int, length: int, new_instructions: list[Instruction]) -> None:
		self.start = start
		self.length = length
		self.new_instructions = new_instructions
	
	def apply(self, old_instructions: list[Instruction]) -> None:
		new_instructions = old_instructions[:self.start] + self.new_instructions + old_instructions[self.start + self.length:]
		old_instructions.clear()
		old_instructions += new_instructions

class Optimization:
	def __init__(self, filter: list[list[list[InstructionMatcher]]] = []) -> None:
		self.filter = filter
	
	def is_match(self, context: Context, instructions: list[Instruction], index: int) -> bool:
		for template in self.filter:
			for line in range(len(template)):
				for matcher in template[line]:
					if matcher.is_match(context, instructions[index + line]): break
				else: break
			else: return True
		return False
	
	def get_optimization(self, context: Context, instructions: list[Instruction], index: int) -> InstructionReplacement | None: ...
	
	def apply(self, context: Context, instructions: list[Instruction], index: int) -> bool:
		optimization = self.get_optimization(context, instructions, index)
		if optimization: optimization.apply(instructions)
		return optimization != None

class PushThenPop(Optimization):
	def __init__(self) -> None:
		super().__init__([
			[
				[InstructionMatcher(Push, [OperandMatcher(any=False, register=True, immediate=True, label=True)])],
				[InstructionMatcher(Pop, [OperandMatcher(any=False, register=True)])]
			]
		])
	
	def get_optimization(self, context: Context, instructions: list[Instruction], index: int) -> InstructionReplacement | None:
		push: Push = cast(Push, instructions[index])
		pop: Pop = cast(Pop, instructions[index + 1])
		return InstructionReplacement(index, 2, [Mov(cast(Register, pop.target), cast(int | Register | Label, push.value))])

class PushThenPopAfterMove(Optimization):
	def __init__(self) -> None:
		super().__init__([
			[
				[InstructionMatcher(Push)],
				[InstructionMatcher(Mov, [OperandMatcher(any=False, register=True), OperandMatcher(any=False, register=True, immediate=True, custom=lambda ctx, operand: operand != SP())])],
				[InstructionMatcher(Pop)]
			]
		])
	
	def get_optimization(self, context: Context, instructions: list[Instruction], index: int) -> InstructionReplacement | None:
		push: Push = cast(Push, instructions[index])
		mov: Mov = cast(Mov, instructions[index + 1])
		pop: Pop = cast(Pop, instructions[index + 2])
		if mov.target == push.value: return None
		return InstructionReplacement(index, 3, [instructions[index + 1], Mov(cast(Register, pop.target), cast(int | Register, push.value))])

class AddOrSubtractZero(Optimization):
	def __init__(self) -> None:
		super().__init__([
			[
				[InstructionMatcher(Add, [OperandMatcher(), OperandMatcher(any=False, immediate=True, custom=lambda ctx, value: value == 0)]), InstructionMatcher(Sub, [OperandMatcher(), OperandMatcher(any=False, immediate=True, custom=lambda ctx, value: value == 0)])],
			]
		])
	
	def get_optimization(self, context: Context, instructions: list[Instruction], index: int) -> InstructionReplacement | None:
		return InstructionReplacement(index, 1, [])

class AddStackPointerAfterPush(Optimization):
	def __init__(self) -> None:
		super().__init__([
			[
				[InstructionMatcher(Push, [OperandMatcher()])],
				[InstructionMatcher(Add, [OperandMatcher(any=False, register=True, custom=lambda ctx, register: register == SP()), OperandMatcher(any=False, immediate=True, custom=lambda ctx, value: value >= ctx.get_word_size())])]
			]
		])
	
	def get_optimization(self, context: Context, instructions: list[Instruction], index: int) -> InstructionReplacement | None:
		add: Add = cast(Add, instructions[index + 1])
		return InstructionReplacement(index, 2, [Add(add.a, cast(int, add.b) - context.get_word_size())])

class MoveSameRegister(Optimization):
	def __init__(self) -> None:
		super().__init__([
			[
				[InstructionMatcher(Mov, [OperandMatcher(any=False, register=True), OperandMatcher(any=False, register=True)])]
			]
		])
	
	def get_optimization(self, context: Context, instructions: list[Instruction], index: int) -> InstructionReplacement | None:
		mov: Mov = cast(Mov, instructions[index])
		if mov.source != mov.target: return None
		return InstructionReplacement(index, 1, [])

class PushThenPopAndCall(Optimization):
	def __init__(self) -> None:
		super().__init__([
			[
				[InstructionMatcher(Push, [OperandMatcher(any=False, immediate=True, label=True)])],
				[InstructionMatcher(Pop, [OperandMatcher(any=False, register=True)])],
				[InstructionMatcher(Call, [OperandMatcher(any=False, register=True)])]
			]
		])
	
	def get_optimization(self, context: Context, instructions: list[Instruction], index: int) -> InstructionReplacement | None:
		push: Push = cast(Push, instructions[index])
		pop: Pop = cast(Pop, instructions[index + 1])
		call: Call = cast(Call, instructions[index + 2])
		if pop.target != call.target: return None
		return InstructionReplacement(index, 3, [Call(cast(int | Label, push.value))])

OPTIMIZATIONS = [
	PushThenPopAndCall(),
	PushThenPop(),
	PushThenPopAfterMove(),
	AddOrSubtractZero(),
	AddStackPointerAfterPush(),
	MoveSameRegister()
]

def optimize(context: Context, instructions: list[Instruction], optimizations: list[Optimization]) -> list[Instruction]:
	result = instructions.copy()
	while True:
		for i in range(len(result)):
			for optimization in optimizations:
				if optimization.is_match(context, result, i):
					if optimization.apply(context, result, i): break
			else: continue
			break
		else: break
	return result

def flatten(iterable):
	for item in iterable:
		try:
			iter(item)
		except TypeError:
			yield item
		else:
			for child in flatten(item): yield child

def main() -> int:
	def stdin() -> str:
		def lines():
			try:
				while True: yield input()
			except EOFError: pass
		return "\n".join(lines())
	
	NO_OPTIMIZATION = False
	
	context = X8616Context(0x7C00)
	result = generate_module(context, ast.parse(stdin()))
	preoptimize = flatten([context.prologue(), result])
	postoptimize = preoptimize if NO_OPTIMIZATION else optimize(context, list(filter(lambda inst: not isinstance(inst, Comment), preoptimize)), OPTIMIZATIONS)
	
	def stringify(instructions):
		indent = 0
		for inst in instructions:
			if isinstance(inst, EndBlockComment): indent = max(0, indent - 1)
			if not isinstance(inst, Volatile): yield ("\t" * indent) + str(inst)
			if isinstance(inst, BlockComment): indent += 1
	
	print(*list(stringify(postoptimize)), sep="\n")
	return 0

exit(main())