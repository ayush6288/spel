CC      ?= clang
ARCH    ?= -mcpu=apple-m5
CFLAGS  ?= -O3 $(ARCH) -ffast-math -std=gnu11 -Wall -Wextra -pthread
LDFLAGS ?= -lm -framework Accelerate -pthread

SRC = main.c model.c ternary.c notebank.c
BIN = spel

all: $(BIN)

$(BIN): $(SRC) spel.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

selftest: $(BIN)
	./$(BIN) selftest

bench: $(BIN)
	./$(BIN) bench

demo: $(BIN)
	./$(BIN) demo

clean:
	rm -f $(BIN) *.o

.PHONY: all selftest bench demo clean
