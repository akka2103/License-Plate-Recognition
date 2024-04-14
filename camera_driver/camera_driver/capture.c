/*
 *  Base Code by Prof. Sam Siewert
 *  Adapted by Akash Karoshi for use with USB C270 web camera and Bt878 frame
 *  
 *  
 *
 *  The original code adapted was open source from V4L2 API and had the
 *  following use and incorporation policy:
 * 
 *  This program can be used and distributed without restrictions.
 *
 *      This program is provided with the V4L2 API
 * see http://linuxtv.org/docs.php for more information
 * 
 *      Part of AESD final project Spring 2024
 * 
 * @author Akash Karoshi
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>

#include <time.h>

#include <syslog.h>
#include "capture.h"

#define CLEAR(x) memset(&(x), 0, sizeof(x))

#define HRES 320
#define VRES 240
#define HRES_STR "320"
#define VRES_STR "240"

/* v4l2_format structure is used to describe the format of a video frame,defined in the 
videodev2.h header file.

contains information about the image format, such as the width, height, pixel format, 
and other parameters. It is used when setting the format of a video device.

*/
static struct v4l2_format fmt;

struct buffer 
{
    void   *start;
    size_t  length;
};

static char            *dev_name;
static int              fd = -1;
struct buffer          *buffers;
static unsigned int     n_buffers;

static int              force_format=1;
static int              frame_count = 30;



/**
 * @brief Prints an error message and exits the program with a failure status.
 *
 * This function is used to handle unrecoverable errors. It prints an error message
 * to the standard error stream (stderr) and then exits the program with a failure
 * status (EXIT_FAILURE).
 *
 * @param s A descriptive message or context for the error.
 *
 * @note The error message includes the string passed as the argument, the current value
 *       of the `errno` variable, and the corresponding error message obtained using
 *       the `strerror` function.
 *
 * @return This function does not return as it calls `exit(EXIT_FAILURE)`.
 */
static void errno_exit(const char *s)
{
    fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
    exit(EXIT_FAILURE);
}

/**
 * @brief Wrapper function for the `ioctl` system call.
 *
 * This function encapsulates the `ioctl` system call and provides error handling.
 * It retries the `ioctl` call if it is interrupted by a signal (EINTR).
 *
 * @param fh The file handle or file descriptor on which to perform the `ioctl` operation.
 * @param request The `ioctl` request or command.
 * @param arg A pointer to a structure or variable containing additional arguments for the `ioctl` request.
 *
 * @return The return value of the `ioctl` system call.
 *         - If the `ioctl` call is successful, the return value is the result of the `ioctl` operation.
 *         - If the `ioctl` call fails and the error is not EINTR, the function returns -1.
 *
 * @note The function will retry the `ioctl` call if it is interrupted by a signal (EINTR) until it succeeds
 *       or encounters a different error.
 *
 * @warning The `arg` parameter is passed as a `void*` pointer and should be cast to the appropriate type
 *          based on the `ioctl` request being made.
 */
static int xioctl(int fh, int request, void *arg)
{
    int r;

    do 
    {
        r = ioctl(fh, request, arg);

    } while (-1 == r && EINTR == errno);

    return r;
}

// This is probably the most acceptable conversion from camera YUYV to RGB
//
// Wikipedia has a good discussion on the details of various conversions and cites good references:
// http://en.wikipedia.org/wiki/YUV
//
// Also http://www.fourcc.org/yuv.php
//
// What's not clear without knowing more about the camera in question is how often U & V are sampled compared
// to Y.
//
// E.g. YUV444, which is equivalent to RGB, where both require 3 bytes for each pixel
//      YUV422, which we assume here, where there are 2 bytes for each pixel, with two Y samples for one U & V,
//              or as the name implies, 4Y and 2 UV pairs
//      YUV420, where for every 4 Ys, there is a single UV pair, 1.5 bytes for each pixel or 36 bytes for 24 pixels

void yuv2rgb(int y, int u, int v, unsigned char *r, unsigned char *g, unsigned char *b)
{
   int r1, g1, b1;

   // replaces floating point coefficients
   int c = y-16, d = u - 128, e = v - 128;       

   // Conversion that avoids floating point
   r1 = (298 * c           + 409 * e + 128) >> 8;
   g1 = (298 * c - 100 * d - 208 * e + 128) >> 8;
   b1 = (298 * c + 516 * d           + 128) >> 8;

   // Computed values may need clipping.
   if (r1 > 255) r1 = 255;
   if (g1 > 255) g1 = 255;
   if (b1 > 255) b1 = 255;

   if (r1 < 0) r1 = 0;
   if (g1 < 0) g1 = 0;
   if (b1 < 0) b1 = 0;

   *r = r1 ;
   *g = g1 ;
   *b = b1 ;
}


unsigned int framecnt=0;
unsigned char bigbuffer[(1280*960)];



static void process_image(const void *p, int size)
{
    int i, newi, newsize=0;
    struct timespec frame_time;
    int y_temp, y2_temp, u_temp, v_temp;
    unsigned char *pptr = (unsigned char *)p;

    // record when process was called
    clock_gettime(CLOCK_REALTIME, &frame_time);    

    framecnt++;
    printf("frame %d: ", framecnt);

    // This just dumps the frame to a file now, but you could replace with whatever image
    // processing you wish.

        printf("Dump YUYV converted to RGB size %d\n", size);
       
        // Pixels are YU and YV alternating, so YUYV which is 4 bytes
        // We want RGB, so RGBRGB which is 6 bytes
        //
        for(i=0, newi=0; i<size; i=i+4, newi=newi+6)
        {
		y_temp=(int)pptr[i]; u_temp=(int)pptr[i+1]; y2_temp=(int)pptr[i+2]; v_temp=(int)pptr[i+3];
		yuv2rgb(y_temp, u_temp, v_temp, &bigbuffer[newi], &bigbuffer[newi+1], &bigbuffer[newi+2]);
		yuv2rgb(y2_temp, u_temp, v_temp, &bigbuffer[newi+3], &bigbuffer[newi+4], &bigbuffer[newi+5]);
        }

        dump_ppm(bigbuffer, ((size*6)/4), framecnt, &frame_time);

}


/**
 * @brief Reads a frame from the video device and processes it.
 *
 * This function reads a frame from the video device using memory mapping (mmap) and
 * processes the captured image data. It uses the `v4l2_buffer` structure to exchange
 * buffer information with the driver.
 *
 *
 * @return 1 on success, 0 if no buffer is available or an I/O error occurs.
 *
 * @note The function uses the `xioctl` wrapper function to execute the ioctl calls with error handling.
 *
 * @note The `buffers` array is a global variable holding the mapped buffers.
 *
 * @note The `fd` variable is a global file descriptor for the video device.
 *
 * @note The `n_buffers` variable is a global variable indicating the number of buffers.

 */
 


static int read_frame(void)
{
    struct v4l2_buffer buf;
    unsigned int i;
    int bytes_captured = -1;
    CLEAR(buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
    {
        switch (errno)
        {
            case EAGAIN:
                return 0;

            case EIO:
                /* Could ignore EIO, but drivers should only set for serious errors, although some set for
                    non-fatal errors too.
                    */
                return 0;


            default:
                printf("mmap failure\n");
                errno_exit("VIDIOC_DQBUF");
        }
    }

    assert(buf.index < n_buffers); 

    process_image(buffers[buf.index].start, buf.bytesused);

    if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
        errno_exit("VIDIOC_QBUF");          

    //printf("R");
    return 1;
}

/**
 * @brief Stops the video capture process.
 *
 * This function stops the video capture process by calling the `VIDIOC_STREAMOFF`
 * ioctl command. It sets the buffer type to `V4L2_BUF_TYPE_VIDEO_CAPTURE` and
 * passes it as an argument to the ioctl.
 *
 * @note The function uses the `xioctl` wrapper function to execute the ioctl call
 *       with error handling.
 *
 * @note If the `VIDIOC_STREAMOFF` ioctl fails, the function calls the `errno_exit`
 *       function with the error message "VIDIOC_STREAMOFF" and terminates the program.
 *
 * @return None
 *
 * @note The `fd` variable is a global file descriptor for the video device.
 */
static void stop_capturing(void)
{
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type))
        errno_exit("VIDIOC_STREAMOFF");
}

/**
 * @brief Starts the video capture process.
 *
 * This function starts the video capture process by enqueuing buffers and
 * starting the streaming. It performs the following steps:
 *
 * 1. Iterates over the allocated buffers (specified by `n_buffers`).
 * 2. For each buffer:
 *    - Sets the buffer type to `V4L2_BUF_TYPE_VIDEO_CAPTURE`.
 *    - Sets the memory type to `V4L2_MEMORY_MMAP`.
 *    - Sets the buffer index.
 *    - Enqueues the buffer using the `VIDIOC_QBUF` ioctl.
 *    - If the ioctl fails, calls the `errno_exit` function with the error message "VIDIOC_QBUF".
 * 3. Sets the buffer type to `V4L2_BUF_TYPE_VIDEO_CAPTURE`.
 * 4. Starts the streaming using the `VIDIOC_STREAMON` ioctl.
 *    - If the ioctl fails, calls the `errno_exit` function with the error message "VIDIOC_STREAMON".
 *
 * @return None
 *
 * @note The function uses the `xioctl` wrapper function to execute the ioctl calls with error handling.
 *
 * @note The `fd` variable is a global file descriptor for the video device.
 *
 * @note The `n_buffers` variable is a global variable indicating the number of allocated buffers.
 */
static void start_capturing(void)
{
    unsigned int i;
    enum v4l2_buf_type type;

    // Iterate over the allocated buffers
    for (i = 0; i < n_buffers; ++i) 
    {
        printf("allocated buffer %d\n", i);

        struct v4l2_buffer buf;
        CLEAR(buf);

        // Set the buffer type and memory type
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        // Enqueue the buffer using VIDIOC_QBUF ioctl
        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
            errno_exit("VIDIOC_QBUF");
    }

    // Set the buffer type for streaming
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    // Start the streaming using VIDIOC_STREAMON ioctl
    if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
        errno_exit("VIDIOC_STREAMON");
}

/**
 * @brief Uninitializes the video device by unmapping buffers and freeing memory.
 *
 * This function performs the necessary cleanup steps to uninitialize the video device:
 *
 * 1. Iterates over the allocated buffers (specified by `n_buffers`).
 * 2. For each buffer:
 *    - Unmaps the buffer memory using the `munmap` system call.
 *    - If `munmap` fails, calls the `errno_exit` function with the error message "munmap".
 * 3. Frees the memory allocated for the `buffers` array.
 *
 * @return None
 *
 * @note The function assumes that the `buffers` array and `n_buffers` variable are globally defined.
 *
 * @note The function uses the `errno_exit` function to handle errors and terminate the program if necessary.
 */
static void uninit_device(void)
{
    unsigned int i;
    // Iterate over the allocated buffers
    for (i = 0; i < n_buffers; ++i) {
        // Unmap the buffer memory using munmap system call
        if (-1 == munmap(buffers[i].start, buffers[i].length))
            errno_exit("munmap");
    }

    // Free the memory allocated for the buffers array
    free(buffers);
}


/**
 * @brief Initializes memory mapping for the video capture buffers.
 *
 * This function sets up the memory mapping for the video capture buffers using the
 * `VIDIOC_REQBUFS` and `VIDIOC_QUERYBUF` ioctl requests. It allocates a set of buffers
 * in the video device's memory space and maps them into the application's address space
 * for efficient access to the captured frames.
 *
 * @note The function requests 6 buffers from the video device.
 *
 * @note If the video device does not support memory mapping, an error message is printed,
 *       and the program exits with `EXIT_FAILURE`.
 *
 * @note If there is insufficient buffer memory on the video device, an error message is
 *       printed, and the program exits with `EXIT_FAILURE`.
 *
 * @note The function uses the `calloc` function to allocate memory for the `buffers` array.
 *       If the memory allocation fails, an "Out of memory" error message is printed, and
 *       the program exits with `EXIT_FAILURE`.
 *
 * @note The `buffers` array stores information about each mapped buffer, including its
 *       length and starting address.
 *
 * @note The `xioctl` function is used to send the `VIDIOC_REQBUFS` and `VIDIOC_QUERYBUF`
 *       ioctl requests to the video device driver.
 *
 * @note The `mmap` function is used to map the buffer memory into the process's address space.
 *
 * @return None
 */
static void init_mmap(void)
{
    struct v4l2_requestbuffers req;
    CLEAR(req);

    // Set the buffer count, type, and memory mode
    req.count = 6;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    // Request buffers from the video device
    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req))
    {
        if (EINVAL == errno)
        {
            fprintf(stderr, "%s does not support memory mapping\n", dev_name);
            exit(EXIT_FAILURE);
        }
        else
        {
            errno_exit("VIDIOC_REQBUFS");
        }
    }

    // Check if sufficient buffer memory is available
    if (req.count < 2)
    {
        fprintf(stderr, "Insufficient buffer memory on %s\n", dev_name);
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the buffers array
    buffers = calloc(req.count, sizeof(*buffers));
    if (!buffers)
    {
        fprintf(stderr, "Out of memory\n");
        exit(EXIT_FAILURE);
    }

    // Query and map each buffer
    for (n_buffers = 0; n_buffers < req.count; ++n_buffers)
    {
        struct v4l2_buffer buf;
        CLEAR(buf);

        // Set the buffer type, memory mode, and index
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = n_buffers;

        // Query the buffer information
        if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
            errno_exit("VIDIOC_QUERYBUF");

        // Store the buffer length
        buffers[n_buffers].length = buf.length;

        // Map the buffer memory into the process's address space
        buffers[n_buffers].start =
            mmap(NULL /* start anywhere */,
                 buf.length,
                 PROT_READ | PROT_WRITE /* required */,
                 MAP_SHARED /* recommended */,
                 fd, buf.m.offset);

        // Check if the memory mapping failed
        if (MAP_FAILED == buffers[n_buffers].start)
            errno_exit("mmap");
    }
}


/**
 * @brief Initializes the video device.
 *
 * This function initializes the video device by performing the following steps:
 * 1. Queries the device capabilities using the `VIDIOC_QUERYCAP` ioctl.
 * 2. Checks if the device supports video capture and streaming I/O.
 * 3. Sets the video cropping parameters using the `VIDIOC_CROPCAP` and `VIDIOC_S_CROP` ioctls.
 * 4. Sets the video format parameters using the `VIDIOC_S_FMT` ioctl.
 * 5. Initializes the memory mapping for the video buffers using the `init_mmap` function.
 *
 * @note The function forces the video format to a specific resolution and pixel format.
 *
 * @note If the device does not support cropping, the function ignores the error and continues.
 *
 * @note The function performs "buggy driver paranoia" checks to ensure the bytesperline and sizeimage values are valid.
 *
 * @return None
 */
static void init_device(void)
{
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    unsigned int min;

    // Query the device capabilities
    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap))
    {
        if (EINVAL == errno)
        {
            fprintf(stderr, "%s is no V4L2 device\n", dev_name);
            exit(EXIT_FAILURE);
        }
        else
        {
            errno_exit("VIDIOC_QUERYCAP");
        }
    }

    // Check if the device supports video capture
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
    {
        fprintf(stderr, "%s is no video capture device\n", dev_name);
        exit(EXIT_FAILURE);
    }

    // Check if the device supports streaming I/O
    if (!(cap.capabilities & V4L2_CAP_STREAMING))
    {
        fprintf(stderr, "%s does not support streaming i/o\n", dev_name);
        exit(EXIT_FAILURE);
    }

    /* Select video input, video standard and tune here. */

    // Set the video cropping parameters
    CLEAR(cropcap);
    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap))
    {
        crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        crop.c = cropcap.defrect; /* reset to default */
        if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop))
        {
            switch (errno)
            {
                case EINVAL:
                    /* Cropping not supported. */
                    break;
                default:
                    /* Errors ignored. */
                    break;
            }
        }
    }

    // Set the video format parameters
    CLEAR(fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    printf("FORCING FORMAT\n");
    fmt.fmt.pix.width = HRES;
    fmt.fmt.pix.height = VRES;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
        errno_exit("VIDIOC_S_FMT");

    /* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;

    // Initialize memory mapping for the video buffers
    init_mmap();
}


static void close_device(void)
{
        if (-1 == close(fd))
                errno_exit("close");

        fd = -1;
}

static void open_device(void)
{
        struct stat st;
        dev_name = "/dev/video2";
        if (-1 == stat(dev_name, &st)) {
                fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }

        if (!S_ISCHR(st.st_mode)) {
                fprintf(stderr, "%s is no device\n", dev_name);
                exit(EXIT_FAILURE);
        }

        fd = open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

        if (-1 == fd) {
                fprintf(stderr, "Cannot open '%s': %d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }
}


void mainloop(void)
{
    int i, j;
    int bytes_captured = -1;

    /*for (;;)
    {
        for (i = 0; i < 20; i++)
        {
            for (j = 0; j < 5000; j++)
            {
                // Read a frame from the video device
                if (read_frame())
                {
                    break;
                }

                // Check for errors
                if (errno != EAGAIN)
                {
                    perror("read_frame");
                    return NULL;
                }
            }

            // Check if a frame was successfully captured
            if (j != 5000)
                break;
        }

        // Check if a frame was successfully captured
        if (i != 20)
            break;

        // Timeout occurred, handle the error or perform necessary actions
        fprintf(stderr, "Timeout occurred while capturing a frame\n");
        return NULL;
    })*/

    // Return a pointer to the captured image data in the global bigbuffer
        for (;;)
    {
        fd_set fds;
        struct timeval tv;
        int r;

        FD_ZERO(&fds);
        FD_SET(fd, &fds);

        /* Timeout. */
        tv.tv_sec = 2;
        tv.tv_usec = 0;

        r = select(fd + 1, &fds, NULL, NULL, &tv);

        if (-1 == r)
        {
            if (EINTR == errno)
                continue;
            errno_exit("select");
        }

        if (0 == r)
        {
            fprintf(stderr, "select timeout\n");
            exit(EXIT_FAILURE);
        }

        if (read_frame	())
        {
            break;
        }
    }
    return bigbuffer;
}


void dump_ppm(const unsigned char *p, int size, int frame_number)
{
    int written, total, dumpfd;
    char ppm_header[100];  // Adjust the size based on your requirements
    char ppm_dumpname[30]; // Adjust the size based on your requirements

    snprintf(ppm_dumpname, sizeof(ppm_dumpname), "demo/picture%d.ppm", frame_number);
    printf("ppm_dumpname = %s\n", ppm_dumpname);
    dumpfd = open(ppm_dumpname, O_WRONLY | O_NONBLOCK | O_CREAT, 00666);
    if(dumpfd == -1)
    	errno_exit("dumpfd open failed");
    

    // PPM header construction
    snprintf(ppm_header, sizeof(ppm_header), "P6\n#Frame %d\n%s %s\n255\n", frame_number, HRES_STR, VRES_STR);

    // Write header to file
    written = write(dumpfd, ppm_header, strlen(ppm_header));

    total = 0;
   
    // Write frame data to file
    do
    {
        written = write(dumpfd, p, size);
        total += written;
    } while (total < size);
    
    close(dumpfd);
     
}

int main(int argc, char **argv)
{
    if(argc > 1)
        dev_name = argv[1];
    else
        dev_name = "/dev/video2";

    printf("Starting main\n");
    open_device();
    init_device();
    start_capturing();
    unsigned char *temp_frame;
    
   
    while(1)
    {
     	mainloop();
    }
    
    stop_capturing();
    uninit_device();
    close_device();
    printf("program end\n");
}






























