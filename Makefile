# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall -march=native -ffast-math -funroll-loops -fopenmp
LDFLAGS = -lX11 -lm -fopenmp

# Target executable
TARGET = galsim

# Source files and Object files
SRCS = galsim.c graphics/graphics.c
OBJS = galsim.o graphics/graphics.o

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Compile galsim.c
galsim.o: galsim.c
	$(CC) $(CFLAGS) -c $< -o $@

# Explicit rule for the graphics module
graphics/graphics.o: graphics/graphics.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET) result.gal

# Phony targets
.PHONY: all clean
