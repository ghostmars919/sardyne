# Build della libreria C (build/libengine.so|.dylib) e degli strumenti.
#
#   make              build ottimizzata in build/
#   make debug        build con ASan/UBSan in build/debug/ (non tocca quella normale)
#   make test         test Python con la build normale
#   make test-asan    test del parser GGUF con la build ASan/UBSan (Linux, gcc)
#   make ARCHFLAGS=   build portabile, senza -march=native

# GNU make definisce CC=cc di default: "CC ?= gcc" non avrebbe effetto.
ifeq ($(origin CC),default)
  CC := gcc
endif
ARCHFLAGS ?= -march=native
CFLAGS    ?= -std=c11 -O3 $(ARCHFLAGS) -Wall -Wextra -fPIC
LDFLAGS   ?= -lm -lpthread
PYTHON    ?= .venv/bin/python
BUILD     ?= build
SAN       := -fsanitize=address,undefined

ifeq ($(shell uname -s),Darwin)
  LIB := $(BUILD)/libengine.dylib
else
  LIB := $(BUILD)/libengine.so
endif

SRCS := $(wildcard csrc/*.c)
HDRS := $(wildcard csrc/*.h)
# Cambiano compilatore o flag -> si ricompila (il file cambia solo quando cambiano i flag).
FLAGS_STAMP := $(BUILD)/.flags
FLAGS_LINE  := $(CC) $(CFLAGS) $(LDFLAGS)

all: $(LIB) $(BUILD)/gguf-info

$(FLAGS_STAMP): FORCE
	@mkdir -p $(BUILD)
	@echo '$(FLAGS_LINE)' | cmp -s - $@ || echo '$(FLAGS_LINE)' > $@

$(LIB): $(SRCS) $(HDRS) $(FLAGS_STAMP)
	$(CC) $(CFLAGS) -shared $(SRCS) -o $@ $(LDFLAGS)

$(BUILD)/gguf-info: tools/gguf_info.c csrc/gguf.c csrc/gguf.h $(FLAGS_STAMP)
	$(CC) $(CFLAGS) tools/gguf_info.c csrc/gguf.c -o $@ $(LDFLAGS)

debug:
	$(MAKE) BUILD=build/debug ARCHFLAGS= \
		CFLAGS="-std=c11 -O1 -g -fno-omit-frame-pointer -Wall -Wextra -fPIC $(SAN)" \
		LDFLAGS="-lm -lpthread $(SAN)"
	@echo "Per usarla da Python: make test-asan (la libreria ASan va precaricata con LD_PRELOAD)"

test: all
	$(PYTHON) -m pytest -q

# Python non è compilato con ASan: il runtime dei sanitizer va caricato per primo con LD_PRELOAD.
test-asan: debug
	LD_PRELOAD="$$($(CC) -print-file-name=libasan.so) $$($(CC) -print-file-name=libubsan.so)" \
	ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	LLM_ENGINE_LIB=build/debug/libengine.so \
	$(PYTHON) -m pytest -q -p no:cacheprovider tests/test_gguf.py tests/test_capi.py

clean:
	rm -rf build

.PHONY: all debug test test-asan clean FORCE
