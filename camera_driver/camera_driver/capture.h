/*
​
​ ​*/

#ifndef __CAPTURE_H__
#define __CAPTURE_H__

static void start_capturing(void);
static void uninit_device(void);
static void init_device(void);
static void close_device(void);
static void open_device(void);
void mainloop(void);
static void stop_capturing(void);

#endif /* __CAMERA_DRIVERS_H__ */
