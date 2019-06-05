#ifndef VIRTCR_IOC_H
#define VIRTCR_IOC_H

#ifndef __KERNEL__
#define __user

#include <stdint.h>
#include <sys/ioctl.h>

#else

#include <linux/ioctl.h>
#endif //KERNEL



//for crypto_data_header, if these is no openssl header
#ifndef RSA_PKCS1_PADDING

#define RSA_PKCS1_PADDING	1
#define RSA_SSLV23_PADDING	2
#define RSA_NO_PADDING		3
#define RSA_PKCS1_OAEP_PADDING	4

#endif

/*
 * function arguments
*/
typedef struct VirtIOArg
{
	uint32_t cmd;
	uint32_t tid;
	void *src;
	uint32_t srcSize;
	void *dst;
	uint32_t dstSize;
	uint64_t flag;
	uint64_t param;
	uint32_t totalSize;

} VirtIOArg;
/* see ioctl-number in https://github.com/torvalds/
	linux/blob/master/Documentation/ioctl/ioctl-number.txt
*/
#define VIRTIO_IOC_ID '0xBB'

#define VIRTIO_IOC_HELLO \
	_IOWR(VIRTIO_IOC_ID,0,int)
/** module control	**/
#define VIRTIO_IOC_REGISTERFATBINARY \
	_IOWR(VIRTIO_IOC_ID,1, VirtIOArg)
#define VIRTIO_IOC_UNREGISTERFATBINARY	\
	_IOWR(VIRTIO_IOC_ID,2,unsigned long)
#define VIRTIO_IOC_REGISTERFUNCTION \
	_IOWR(VIRTIO_IOC_ID,3,unsigned long)
#define VIRTIO_IOC_LAUNCH \
	_IOWR(VIRTIO_IOC_ID,4,unsigned long)
/* memory management */
#define VIRTIO_IOC_MALLOC\
	_IOWR(VIRTIO_IOC_ID,5,unsigned long)
#define VIRTIO_IOC_MEMCPY \
	_IOWR(VIRTIO_IOC_ID,6,unsigned long)
#define VIRTIO_IOC_FREE \
	_IOWR(VIRTIO_IOC_ID,7,unsigned long)
/**	device management	**/
#define VIRTIO_IOC_GETDEVICE \
	_IOWR(VIRTIO_IOC_ID,8,unsigned long)
#define VIRTIO_IOC_GETDEVICEPROPERTIES \
	_IOWR(VIRTIO_IOC_ID,9,unsigned long)
#define VIRTIO_IOC_CONFIGURECALL \
	_IOWR(VIRTIO_IOC_ID,10,unsigned long)

#define VIRTIO_IOC_SETUPARGUMENT \
	_IOWR(VIRTIO_IOC_ID,11,unsigned long)
#define VIRTIO_IOC_GETDEVICECOUNT \
	_IOWR(VIRTIO_IOC_ID,12,unsigned long)
#define VIRTIO_IOC_SETDEVICE \
	_IOWR(VIRTIO_IOC_ID,13,unsigned long)
#define VIRTIO_IOC_DEVICERESET \
	_IOWR(VIRTIO_IOC_ID,14,unsigned long)
#define VIRTIO_IOC_STREAMCREATE \
	_IOWR(VIRTIO_IOC_ID,15,unsigned long)

#define VIRTIO_IOC_STREAMDESTROY \
	_IOWR(VIRTIO_IOC_ID,16,unsigned long)
#define VIRTIO_IOC_EVENTCREATE \
	_IOWR(VIRTIO_IOC_ID,17,unsigned long)
#define VIRTIO_IOC_EVENTDESTROY \
	_IOWR(VIRTIO_IOC_ID,18,unsigned long)
#define VIRTIO_IOC_EVENTRECORD \
	_IOWR(VIRTIO_IOC_ID,19,unsigned long)
#define VIRTIO_IOC_EVENTSYNCHRONIZE \
	_IOWR(VIRTIO_IOC_ID,20,unsigned long)

#define VIRTIO_IOC_EVENTELAPSEDTIME \
	_IOWR(VIRTIO_IOC_ID,21,unsigned long)
#define VIRTIO_IOC_THREADSYNCHRONIZE \
	_IOWR(VIRTIO_IOC_ID,22,unsigned long)
#define VIRTIO_IOC_GETLASTERROR \
	_IOWR(VIRTIO_IOC_ID,23,unsigned long)

#define VIRTIO_CUDA_HELLO 0
/** module control	**/
#define VIRTIO_CUDA_REGISTERFATBINARY 1
#define VIRTIO_CUDA_UNREGISTERFATBINARY	2
#define VIRTIO_CUDA_REGISTERFUNCTION 3
#define VIRTIO_CUDA_LAUNCH 4
/* memory management */
#define VIRTIO_CUDA_MALLOC 5
#define VIRTIO_CUDA_MEMCPY 6
#define VIRTIO_CUDA_FREE 7
/**	device management	**/
#define VIRTIO_CUDA_GETDEVICE 8
#define VIRTIO_CUDA_GETDEVICEPROPERTIES 9
#define VIRTIO_CUDA_CONFIGURECALL 10

#define VIRTIO_CUDA_SETUPARGUMENT 11
#define VIRTIO_CUDA_GETDEVICECOUNT 12
#define VIRTIO_CUDA_SETDEVICE 13
#define VIRTIO_CUDA_DEVICERESET 14
#define VIRTIO_CUDA_STREAMCREATE 15

#define VIRTIO_CUDA_STREAMDESTROY 16
#define VIRTIO_CUDA_EVENTCREATE 17
#define VIRTIO_CUDA_EVENTDESTROY 18
#define VIRTIO_CUDA_EVENTRECORD 19
#define VIRTIO_CUDA_EVENTSYNCHRONIZE 20

#define VIRTIO_CUDA_EVENTELAPSEDTIME 21
#define VIRTIO_CUDA_THREADSYNCHRONIZE 22
#define VIRTIO_CUDA_GETLASTERROR 23
#endif

