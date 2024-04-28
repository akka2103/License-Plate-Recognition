
CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c11

SRC = server.c capture.c
OBJ = $(SRC:.c=.o)
TARGET = server

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)
	
	
#----------------------
# arm-linux-gnueabihf-g++ -Wall -Wextra -pedantic -std=c++11 -pthread capture.c server.c -o server
# scp server root@192.168.50.251:/root/server
