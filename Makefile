.PHONY: all run build clean

CC=go
# CC=/go/bin/go

EXT = .go
PJNAME = truth_table

GOFILE = $(PJNAME)$(EXT)

BIN = bin

ifeq ($(OS),Windows_NT)
    BINEXT = .exe
    RM = if exist $(BIN) rd /s /q $(BIN)
    MKDIR = if not exist $(BIN) md $(BIN)
    ECHO = echo
    ifeq ($(PROCESSOR_ARCHITEW6432),AMD64)
        
    else
        ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
            
        endif
        ifeq ($(PROCESSOR_ARCHITECTURE),x86)
            
        endif
    endif
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        BINEXT = 
        RM = rm -rfv $(BIN)
        MKDIR = mkdir -pv $(BIN)
        ECHO = echo
    endif
    ifeq ($(UNAME_S),Darwin)
        BINEXT = 
        RM = rm -rfv $(BIN)
        MKDIR = mkdir -pv $(BIN)
        ECHO = echo
    endif
    UNAME_P := $(shell uname -p)
    ifeq ($(UNAME_P),x86_64)
        
    endif
    ifneq ($(filter %86,$(UNAME_P)),)
        
    endif
    ifneq ($(filter arm%,$(UNAME_P)),)
        
    endif
endif

TARGET=$(BIN)/$(PJNAME)$(BINEXT)


all:
	@$(ECHO) "make run   - to run the program"
	@$(ECHO) "make build - to build the program"
	@$(ECHO) "make clean - to clean the project"


run: $(PJNAME)

$(PJNAME):
	@$(CC) run $@$(EXT)


build: $(TARGET)

$(TARGET): $(GOFILE) $(BIN)
	@$(ECHO) $(@F)
	@$(CC) build -o $@ $<

$(BIN):
	@$(MKDIR)


clean:
	@$(RM)
