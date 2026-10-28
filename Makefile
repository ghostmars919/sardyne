CC      ?= gcc
CFLAGS  ?= -std=c11 -O3 -march=native -Wall -Wextra -fPIC
LDFLAGS ?= -shared -lm -lpthread

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  LIB := build/libengine.dylib
else
  LIB := build/libengine.so
endif

SRCS := $(shell find csrc -name '*.c')

all: $(LIB)

$(LIB): $(SRCS) $(shell find csrc -name '*.h')
	@mkdir -p build
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

debug: CFLAGS = -std=c11 -O0 -g -Wall -Wextra -fPIC -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: clean $(LIB)

test: $(LIB)
	python -m pytest -q

clean:
	rm -rf build

.PHONY: all debug test clean
