# Makefile

CC = gcc
CFLAGS = -Wall -fPIC -O2 -arch arm64
SRC = src/c_class/C_functions.c
OUTDIR = src/c_class
OUT = $(OUTDIR)/C_functions.so

INCLUDES = -Isrc/c_class

all: $(OUT)

$(OUT): $(SRC)
	@mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -shared -o $(OUT) $(SRC) $(INCLUDES)

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean

