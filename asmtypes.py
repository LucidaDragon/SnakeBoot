import typing

ArithmeticError: None = None
AssertionError: None = None
AttributeError: None = None
BaseException: None = None
BlockingIOError: None = None
BrokenPipeError: None = None
BufferError: None = None
BytesWarning: None = None
ChildProcessError: None = None
ConnectionAbortedError: None = None
ConnectionError: None = None
ConnectionRefusedError: None = None
ConnectionResetError: None = None
DeprecationWarning: None = None
EOFError: None = None
Ellipsis: None = None
EncodingWarning: None = None
EnvironmentError: None = None
Exception: None = None
FileExistsError: None = None
FileNotFoundError: None = None
FloatingPointError: None = None
FutureWarning: None = None
GeneratorExit: None = None
IOError: None = None
ImportError: None = None
ImportWarning: None = None
IndentationError: None = None
IndexError: None = None
InterruptedError: None = None
IsADirectoryError: None = None
KeyError: None = None
KeyboardInterrupt: None = None
LookupError: None = None
MemoryError: None = None
ModuleNotFoundError: None = None
NameError: None = None
NotADirectoryError: None = None
NotImplemented: None = None
NotImplementedError: None = None
OSError: None = None
OverflowError: None = None
PendingDeprecationWarning: None = None
PermissionError: None = None
ProcessLookupError: None = None
RecursionError: None = None
ReferenceError: None = None
ResourceWarning: None = None
RuntimeError: None = None
RuntimeWarning: None = None
StopAsyncIteration: None = None
StopIteration: None = None
SyntaxError: None = None
SyntaxWarning: None = None
SystemError: None = None
SystemExit: None = None
TabError: None = None
TimeoutError: None = None
TypeError: None = None
UnboundLocalError: None = None
UnicodeDecodeError: None = None
UnicodeEncodeError: None = None
UnicodeError: None = None
UnicodeTranslateError: None = None
UnicodeWarning: None = None
UserWarning: None = None
ValueError: None = None
Warning: None = None
WindowsError: None = None
ZeroDivisionError: None = None
_: None = None
__build_class__: None = None
__import__: None = None
__loader__: None = None
__spec__: None = None
abs: None = None
aiter: None = None
all: None = None
anext: None = None
any: None = None
ascii: None = None
bin: None = None
breakpoint: None = None
bytearray: None = None
bytes: None = None
callable: None = None
chr: None = None
classmethod: None = None
compile: None = None
complex: None = None
copyright: None = None
credits: None = None
delattr: None = None
dict: None = None
dir: None = None
divmod: None = None
enumerate: None = None
eval: None = None
exec: None = None
exit: None = None
filter: None = None
float: None = None
format: None = None
frozenset: None = None
getattr: None = None
globals: None = None
hasattr: None = None
hash: None = None
help: None = None
hex: None = None
id: None = None
input: None = None
isinstance: None = None
issubclass: None = None
iter: None = None
len: None = None
license: None = None
list: None = None
locals: None = None
map: None = None
max: None = None
memoryview: None = None
min: None = None
next: None = None
object: None = None
oct: None = None
open: None = None
ord: None = None
pow: None = None
print: None = None
property: None = None
quit: None = None
range: None = None
repr: None = None
reversed: None = None
round: None = None
set: None = None
setattr: None = None
slice: None = None
sorted: None = None
staticmethod: None = None
sum: None = None
super: None = None
tuple: None = None
type: None = None
vars: None = None
zip: None = None

__pyint__ = int
__pystr__ = str

str: None = None

void: typing.TypeAlias = None

class _bool: ...
bool: typing.TypeAlias = __pyint__ | _bool
del _bool

class _uint: ...
uint: typing.TypeAlias = __pyint__ | _uint
del _uint

class _uint8: ...
uint8: typing.TypeAlias = __pyint__ | _uint8
del _uint8

class _uint16: ...
uint16: typing.TypeAlias = __pyint__ | _uint16
del _uint16

class _uint32: ...
uint32: typing.TypeAlias = __pyint__ | _uint32
del _uint32

class _uint64: ...
uint64: typing.TypeAlias = __pyint__ | _uint64
del _uint64

class _int: ...
int: typing.TypeAlias = __pyint__ | _int
del _int

class _int8: ...
int8: typing.TypeAlias = __pyint__ | _int8
del _int8

class _int16: ...
int16: typing.TypeAlias = __pyint__ | _int16
del _int16

class _int32: ...
int32: typing.TypeAlias = __pyint__ | _int32
del _int32

class _int64: ...
int64: typing.TypeAlias = __pyint__ | _int64
del _int64

T_PtrDataType = typing.TypeVar("T_PtrDataType")
class _ptr(typing.Generic[T_PtrDataType]): ...
del T_PtrDataType
TPtrDataType = typing.TypeVar("TPtrDataType")
ptr: typing.TypeAlias[TPtrDataType] = __pyint__ | __pystr__ | _ptr[TPtrDataType]
del TPtrDataType
del _ptr

def bios(i: uint8, ax: int16 | uint16 = ..., bx: int16 | uint16 = ..., cx: int16 | uint16 = ..., dx: int16 | uint16 = ..., si: int16 | uint16 = ..., di: int16 | uint16 = ..., bp: int16 | uint16 = ..., sp: int16 | uint16 = ..., ds: int16 | uint16 = ..., es: int16 | uint16 = ..., ss: int16 | uint16 = ...): ...

del typing