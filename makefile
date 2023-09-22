build: example.bin

example.bin: example.asm output
	nasm ./output/example.asm -f bin -o ./output/example.bin

example.asm: example.py output
	cat example.py | python asm.py > ./output/example.tmp.asm
	cp ./output/example.tmp.asm ./output/example.asm
	rm ./output/example.tmp.asm

output:
	mkdir -p output

clean:
	rm -rf output