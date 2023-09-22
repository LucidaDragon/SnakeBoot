from asmtypes import *

class BIOSKeystroke:
	scancode_and_ascii: uint16

vga: ptr[uint16] = 0xB0008000
index: uint = 0

def boot() -> void:
	bios(i=0x10, ax=3)
	
	while True:
		keystroke: BIOSKeystroke = bios(i=0x16, ax=0)
		keystroke.scancode_and_ascii &= 0xFF
		bios(i=0x10, ax=0x0E00 | keystroke.scancode_and_ascii, bx=0)
		if keystroke.scancode_and_ascii == 0x0D:
			bios(i=0x10, ax=0x0E0A, bx=0)