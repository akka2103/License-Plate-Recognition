#

CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm -lpthread

TARGET = capture
SRC = capture.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)