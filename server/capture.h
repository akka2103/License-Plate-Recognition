/*
​
​ ​*/

#ifndef __CAPTURE_H__
#define __CAPTURE_H__

void start_capturing(void);
void stop_capturing(void);
void uninit_device(void);
void init_device(void);
void close_device(void);
void open_device(void);
unsigned char * mainloop(void);


#endif /* _CAPTURE_H_ */
