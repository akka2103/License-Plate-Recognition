
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <syslog.h>
#include <netdb.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <sched.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <getopt.h>
#include <linux/fs.h>
#include <pthread.h>
#include "capture.h"

struct addrinfo *server_info;
struct sockaddr_in client_addr;
int socket_fd;
int client_fd;
struct addrinfo hints;



void signal_handler(int sig)
{
	if(sig==SIGINT)
	{
		syslog(LOG_INFO,"Caught SIGINT");
	}
	else if(sig==SIGTERM)
	{
		syslog(LOG_INFO,"Caught SIGTERM");
	}
	
	// Close socket and client connection 
	close(socket_fd);
	close(client_fd);
	syslog(LOG_ERR,"Connection ended with %s",inet_ntoa(client_addr.sin_addr));
	printf("Connection ended with %s\n",inet_ntoa(client_addr.sin_addr));
	exit(0); 
}




int main()
{
    int num = 1, retryflag=0;
    int get_addr, sockopt_r, bind_r, listen_r;
    
    socklen_t size = sizeof(struct sockaddr);


    openlog(NULL,LOG_PID, LOG_USER);
    //camera sequence
    printf("Starting camera......\n");
    open_device();
    init_device();
    start_capturing();

    //initialise the signal handler 
	if(SIG_ERR == signal(SIGINT,signal_handler))
	{
		syslog(LOG_ERR,"SIGINT failed");
		exit(-1);
	}
	if(SIG_ERR == signal(SIGTERM,signal_handler))
	{
		syslog(LOG_ERR,"SIGTERM failed");
		exit(-1);
	}

    // start server socket
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if(socket_fd == -1)
    {
		syslog(LOG_ERR, "Failed to create server socket");
		printf("Failed to create server socket\n");
		exit(-1);        
    }

    hints.ai_flags=AI_PASSIVE;

    get_addr = getaddrinfo(NULL,"9000",&hints,&server_info);
    if(get_addr != 0)
    {
		syslog(LOG_ERR, "Failed to get the address from getaddrinfo");
		printf("Failed to get the address from getaddrinfo\n");
		exit(-1);        
    }
    sockopt_r = setsockopt(socket_fd,SOL_SOCKET,SO_REUSEADDR,&num,sizeof(num));
    if(sockopt_r == -1)
    {
		syslog(LOG_ERR, "setsockopt call failed");
		printf("setsockopt call failed\n");
		exit(-1);        
    }
    
    //bind
    bind_r = bind(socket_fd,server_info->ai_addr,sizeof(struct sockaddr));
    if(bind_r == -1)
	{
		freeaddrinfo(server_info); 
		syslog(LOG_ERR, "bind call failed");
		printf("bind call failed\n");
		exit(-1);
	}

    freeaddrinfo(server_info); 
    
    //listen 
    listen_r = listen(socket_fd,1); 
	if(listen_r == -1)
	{
		syslog(LOG_ERR, "listen call failed");
		printf("listen call failed\n");
		exit(-1);
	}
while(1)
{
	retryflag=0;
	printf("Accepting.......\n");
	//accept
        client_fd = accept(socket_fd,(struct sockaddr *)&client_addr,&size);
	printf("connection successfully accepected\n");
	if(-1 == client_fd)
	{
		syslog(LOG_ERR, "Failed to accept the connection");
		printf("Failed to accept the connection\n");
		exit(-1);
	}
    else
	{
		syslog(LOG_INFO,"Accepted connection from %s",inet_ntoa(client_addr.sin_addr));
		printf("Accepted connection from %s\n",inet_ntoa(client_addr.sin_addr));
	}

    while(1)
    {
    		if(retryflag==1)
    			break;
    		
		int bytes_sent;
                unsigned char *img_buffer;
		img_buffer = mainloop();   //returns the image buffer (big buffer)
		bytes_sent = send(client_fd,img_buffer,((614400*6)/4),0);
		if(bytes_sent == -1)
		{
			printf("Try accepting a new connection \n");;
			retryflag=1;
			break;
		}
		printf("bytes sent = %d\n",bytes_sent);
		
		
    }
    if(retryflag==1)
    	continue;
   
  }

}
