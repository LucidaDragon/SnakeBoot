# SnakeBoot
A Python subset language for low-level and bare metal programming.

# Quick Start
1. Install Python 3.10+.
2. Install `nasm`.
3. Install and run `make` or manually run the commands in the makefile.

# How To
## Code
### Linting and Syntax Highlight
```py
# Override the default Python types with SnakeBoot types.
from asmtypes import *
```
### Primitive Types
Name | Identifier | Example
--- | --- | ---
Boolean | bool | `x: bool = True`
Unsigned Native Integer | uint | `x: uint = 0`
Unsigned 8-bit Integer | uint8 | `x: uint8 = 0`
Unsigned 16-bit Integer | uint16 | `x: uint16 = 0`
Unsigned 32-bit Integer | uint32 | `x: uint32 = 0`
Unsigned 64-bit Integer | uint64 | `x: uint64 = 0`
Signed Native Integer | int | `x: int = 0`
Signed 8-bit Integer | int8 | `x: int8 = 0`
Signed 16-bit Integer | int16 | `x: int16 = 0`
Signed 32-bit Integer | int32 | `x: int32 = 0`
Signed 64-bit Integer | int64 | `x: int64 = 0`
Data Pointer | ptr[T] | `x: ptr[uint8] = "Hello World!\0"`

### Pointers & Segments
```py
# Segment pointers use the upper 16 bits of an integer value to store the segment.
vga_buffer: ptr[uint16] = 0xB0008000 # Address 0xB8000
```

### Structure Types
```py
# Structured data can be defined with the class keyword.
class Point2D:
	x: int16
	y: int16

class SecretlyPoint4D:
	x: int16
	y: int16
	# Fields defined as _ are not assigned a name.
	_: int16
	_: int16

def main() -> void:
	point_2d: Point2D
	point_4d: SecretlyPoint4D
	
	# Fields can be accessed by name with dot syntax.
	point_2d.x = 10
	point_2d.y = 20
	
	# Fields can also be accessed by index.
	point_4d._0 = 10 # SecretlyPoint4D.x 
	point_4d._1 = 20 # SecretlyPoint4D.y
	point_4d._2 = 30 # SecretlyPoint4D unnamed field 1
	point_4d._3 = 40 # SecretlyPoint4D unnamed field 2
```

### Interrupt Calls
```py
# Interrupts can be invoked with the bios macro function.
def print_character(c: uint8) -> void:
	# Raises int 0x10 with ah = 0x0E, al = c, bx = 0
	bios(i=0x10, ax=0x0E00 | c, bx=0)
```

### Inline Assembly
```py
# Inline assembly can be inserted using string expressions.
def halt() -> void:
	"hlt" # Halt the processor instead of trying to return.
```

### Entry Point
```py
# When running on bare metal, the first defined function will become the entry point.
def my_first_function() -> void: # Entry point here.
	"hlt"

def my_second_function() -> void:
	...
```

## Run
### Run on emu8086
1. Open `./output/example.asm` in emu8086.
2. Replace `bits 16` with `#make_boot#`.
3. Run the emu8086 emulator.