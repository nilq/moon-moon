build:
	moonc .

train:
	moonc .
	th train.lua -data_dir data/tinyshakespeare/
