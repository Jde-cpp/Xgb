CCACHE_EXISTS := $(shell ccache -V)
ifdef CCACHE_EXISTS
    CC := ccache $(CC)
    CXX := ccache g++-8
endif
DEBUG ?= 1
CXXFLAGS_BASE = -c -g -pthread -fPIC -std=c++17 -Wall -Wno-unknown-pragmas -DJDE_GBM_EXPORTS# -Wno-unknown-pragmas
#CXXFLAGS =  -g $(INCLUDE) -pthread -c -std=c++17 -Wall -Wno-switch -fPIC
BIN_DIR_BASE:=../../bin/
OUT_DIR_BASE := .obj/
ifeq ($(DEBUG), 1)
	OUT_DIR=$(OUT_DIR_BASE)debug
	CXXFLAGS= $(CXXFLAGS_BASE) -O0 -I$(OUT_DIR) -fsanitize=address -fno-omit-frame-pointer
	BIN_DIR=$(BIN_DIR_BASE)debug
else
	OUT_DIR=$(OUT_DIR_BASE)release
	CXXFLAGS= $(CXXFLAGS_BASE) -march=native -DNDEBUG -O3 -I$(OUT_DIR)
	BIN_DIR=$(BIN_DIR_BASE)release
endif

OUTPUT_FILE = libXgb.so
OUTPUT = $(BIN_DIR)/$(OUTPUT_FILE)
LIBS =  -lxgboost
#-lstdc++fs -lMy -lDts

all: $(OUTPUT)
	$(NOECHO) $(NOOP)

OBJECTS = $(OUT_DIR)/Xgb.o $(OUT_DIR)/Booster.o $(OUT_DIR)/Dataset.o $(OUT_DIR)/XgbBoosterParams.o #$(OUT_DIR)/Parameters.o $(OUT_DIR)/FeatureInteractions.o
LDFLAGS = -pthread -shared -Wl,-z,origin -Wl,-rpath='$$ORIGIN'

$(OUTPUT): $(OUT_DIR)/stdafx.h.gch $(OBJECTS)
	$(CXX) -L$(BIN_DIR) $(LDFLAGS) $(OBJECTS) -o$(OUTPUT) $(LIBS)

$(OUT_DIR)/stdafx.h.gch: pc.h Dataset.h
	$(CXX) $(CXXFLAGS) pc.h -o$(OUT_DIR)/stdafx.h.gch -I/home/duffyj/code/libraries/spdlog/include  -I/home/duffyj/code/libraries/xgboost/include -I/home/duffyj/code/libraries/eigen -I/home/duffyj/code/libraries/json/include -I/home/duffyj/code/libraries/xgboost/rabit/include  -I$BOOST_ROOT

$(OUT_DIR)/%.o: ./%.cpp ./%.h pc.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -Wno-int-in-bool-context -Wno-ignored-attributes ./$< -o$@ -c -I/home/duffyj/code/libraries/eigen  -I/home/duffyj/code/libraries/json/include

clean:
	rm -rf -d $(OUT_DIR)/*.*
	rm -rf -d $(OUTPUT)

