/*
 * Virtio Console and Generic Serial Port Devices
 *
 * Copyright Red Hat, Inc. 2009, 2010
 *
 * Authors:
 *  Amit Shah <amit.shah@redhat.com>
 *
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "chardev/char-fe.h"
#include "qemu/error-report.h"
#include "trace.h"
#include "hw/virtio/virtio-serial.h"
#include "qapi/error.h"
#include "qapi/qapi-events-char.h"
#include "exec/cpu-common.h"    // cpu_physical_memory_rw

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h> //PATH: /usr/local/cuda/include/builtin_types.h
// #include <driver_types.h>   // cudaDeviceProp

#include "virtio-ioc.h"
// #include "message_queue.h"
#include "memorypool.h"
#include "list.h"

#include <openssl/hmac.h> // hmac EVP_MAX_MD_SIZE
/*Encodes Base64 */
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <stdint.h> // uint32_t ...
#include <limits.h> // CHAR_BIT , usually 8
#include <signal.h>

/*
* used for pipes
*/
#define READ 0
#define WRITE 1

#ifndef min
#define min(a,b)    (((a)<(b)) ? (a) : (b))
#endif

// #define VIRTIO_CUDA_DEBUG
#ifdef VIRTIO_CUDA_DEBUG
    #define func() printf("[FUNC]%s\n",__FUNCTION__)
    #define debug(fmt, arg...) printf("[DEBUG] "fmt, ##arg)
#else
    #define func()   
    #define debug(fmt, arg...)   
#endif

#define error(fmt, arg...) printf("[ERROR]In file %s, line %d, "fmt, \
            __FILE__, __LINE__, ##arg)


#define TYPE_VIRTIO_CONSOLE_SERIAL_PORT "virtcudaport"
#define VIRTIO_CONSOLE(obj) \
    OBJECT_CHECK(VirtConsole, (obj), TYPE_VIRTIO_CONSOLE_SERIAL_PORT)

#ifndef WORKER_THREADS
#define WORKER_THREADS 32+1
#endif

/* bitmap */
typedef uint32_t word_t;
enum{BITS_PER_WORD = sizeof(word_t) * CHAR_BIT}; // BITS_PER_WORD=32
#define WORD_OFFSET(b) ((b)/BITS_PER_WORD)
#define BIT_OFFSET(b) ((b)%BITS_PER_WORD)


static const mem_size_t max_mem = 2 * GB ;
static const mem_size_t mem_pool_size = 1 * GB ;
// static MemoryPool* mpool;

typedef struct VirtConsole {
    VirtIOSerialPort parent_obj;
    char *privatekeypath;    
    char *hmac_path;
    CharBackend chr;
    guint watch;
} VirtConsole;

#define CudaFunctionMaxNum 1024
#define CudaEventMaxNum BITS_PER_WORD
#define CudaStreamMaxNum BITS_PER_WORD
#define VOL_OFFSET 0x1000
typedef struct VirtualObjectList {
    uint64_t addr;
    uint64_t v_addr;
    int size;
    struct list_head list;
} VOL;

typedef struct HostVirtualObjectList {
    uint64_t native_addr;
    uint64_t actual_addr;
    uint64_t virtual_addr;
    size_t size;
    struct list_head list;
    char file_path[64];
} HVOL;

typedef struct CudaDev {
    CUdevice device;
    struct list_head vol;
    pthread_spinlock_t vol_lock;
    struct list_head host_vol;
}CudaDev;

typedef struct KernelInfo {
    void *fatbin;
    int fatbin_size;
    CUmodule module;
    CUfunction kernel_func;
    char *func_name;
    int func_name_size;
    uint32_t func_id;
} KernelInfo;

typedef struct KernelConf {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
} KernelConf_t ;


static int total_device;   // total GPU device
static int total_port;     // port count
static QemuMutex total_port_mutex;

static CudaDev cudaDevices[WORKER_THREADS];
static int cudaFunctionNum[WORKER_THREADS];
static KernelInfo devicesKernels[WORKER_THREADS][CudaFunctionMaxNum];

static cudaEvent_t cudaEvent[WORKER_THREADS][CudaEventMaxNum];
static word_t cudaEventBitmap[WORKER_THREADS];
static cudaStream_t cudaStream[WORKER_THREADS][CudaStreamMaxNum];
static word_t cudaStreamBitmap[WORKER_THREADS];

static int pfd[WORKER_THREADS][2]; // parent pipe fd
static int cfd[WORKER_THREADS][2]; // child pipe fd

static int global_initialized = 0;
static int global_deinitialized = 0;

#define cudaCheck(call, pipes) { \
    cudaError_t err; \
    if ( (err = (call)) != cudaSuccess) { \
        fprintf(stderr, "Got error %s:%s at %s:%d\n", cudaGetErrorName(err), \
                cudaGetErrorString(err), \
                __FILE__, __LINE__); \
        write(pipes, &err, sizeof(cudaError_t)); \
        break;\
    } \
    debug("sending err %d\n", err);\
    write(pipes, &err, sizeof(cudaError_t)); \
}

#define cuCheck(call, pipes) { \
    cudaError_t err; \
    if ( (err = (call)) != cudaSuccess) { \
        char *str; \
        cuGetErrorName(err, (const char**)&str); \
        fprintf(stderr, "Got error %s at %s:%d\n", str, \
                __FILE__, __LINE__); \
        write(pipes, &err, sizeof(cudaError_t)); \
        break;\
    } \
    debug("sending err %d\n", err);\
    write(pipes, &err, sizeof(cudaError_t)); \
}

#define cudaError(err) __cudaErrorCheck(err, __LINE__)
static inline void __cudaErrorCheck(cudaError_t err, const int line)
{
    if (err != cudaSuccess) {
        char *str = (char*)cudaGetErrorString(err);
        error("CUDA Runtime API Error = %04d \"%s\" line %d\n", err, str, line);
    }
}

#define cuError(err) __cuErrorCheck(err, __LINE__)
static inline void __cuErrorCheck(cudaError_t err, const int line)
{
    char *str;
    if (err != cudaSuccess) {
        cuGetErrorName(err, (const char**)&str);
        error("CUDA Driver API Error = %04d \"%s\" line %d\n", err, str, line);
    }
}

#define HMAC_SHA256_SIZE 32 // hmac-sha256 output size is 32 bytes
#define HMAC_SHA256_COUNT (1<<10)
#define HMAC_SHA256_BASE64_SIZE 44 //HMAC_SHA256_SIZE*4/3

/*
static const char key[] = {
    0x30, 0x81, 0x9f, 0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 
    0x0d, 0x01, 0x01, 0x01, 0x05, 0xa0, 0x03, 0x81, 0x8d, 0x44, 0x30, 0x81, 
    0x89, 0x02, 0x81, 0x81, 0x0b, 0xbb, 0xbd, 0xba, 0x9a, 0x8c, 0x3c, 0x38, 
    0xa3, 0xa8, 0x09, 0xcb, 0xc5, 0x2d, 0x98, 0x86, 0xe4, 0x72, 0x99, 0xe4,
    0x3b, 0x72, 0xb0, 0x73, 0x8a, 0xac, 0x12, 0x74, 0x99, 0xa7, 0xf4, 0xd1, 
    0xf9, 0xf4, 0x22, 0xeb, 0x61, 0x7b, 0xf5, 0x11, 0xd6, 0x9b, 0x02, 0x8e, 
    0xb4, 0x59, 0xb0, 0xb5, 0xe5, 0x11, 0x80, 0xb6, 0xe3, 0xec, 0x3f, 0xd6,
    0x1a, 0xe3, 0x4b, 0x18, 0xe7, 0xda, 0xff, 0x6b, 0xec, 0x7b, 0x71, 0xb6,
    0x78, 0x79, 0xc7, 0x97, 0x90, 0x81, 0xf2, 0xbb, 0x91, 0x5f, 0xd7, 0xc1, 
    0x97, 0xf2, 0xa0, 0xc0, 0x25, 0x6b, 0xd8, 0x96, 0x84, 0xb9, 0x49, 0xba, 
    0xe9, 0xb0, 0x50, 0x78, 0xfe, 0x57, 0x78, 0x1a, 0x2d, 0x75, 0x1e, 0x1c, 
    0xbd, 0x7d, 0xfc, 0xb8, 0xf6, 0x22, 0xbc, 0x20, 0xdd, 0x3e, 0x32, 0x75, 
    0x41, 0x63, 0xdd, 0xb2, 0x94, 0xc1, 0x29, 0xcc, 0x5e, 0x15, 0xb7, 0x1c, 
    0x0f, 0x02, 0x03, 0x01, 0x80, 0x01, 0x00
};

*/

int key_len =  162;
unsigned int result_len;
unsigned char result[EVP_MAX_MD_SIZE];
char *hmac_list = NULL;
unsigned long long int hmac_count = 0;
time_t oldMTime; // 判断hmac.txt是否更新

/*
static void hmac_encode( const char * keys, unsigned int key_length,  
                const char * input, unsigned int input_length,  
                unsigned char *output, unsigned int *output_length) 
{
    const EVP_MD * engine = EVP_sha256();  
    HMAC_CTX ctx;  
    func();

    HMAC_CTX_init(&ctx);  
    HMAC_Init_ex(&ctx, keys, key_length, engine, NULL);  
    HMAC_Update(&ctx, (unsigned char*)input, input_length); 

    HMAC_Final(&ctx, output, output_length);  
    HMAC_CTX_cleanup(&ctx);  
}
*/

// from https://gist.github.com/barrysteyn/7308212
//Encodes a binary safe base 64 string
/*
static int base64_encode(const unsigned char* buffer, size_t length, char** b64text) 
{
    func();
    BIO *bio, *b64;
    BUF_MEM *bufferPtr;

    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);

    //Ignore newlines - write everything in one line
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL); 

    BIO_write(bio, buffer, length);
    BIO_flush(bio);
    BIO_get_mem_ptr(bio, &bufferPtr);
    BIO_set_close(bio, BIO_NOCLOSE);
    BIO_free_all(bio);

    *b64text=(*bufferPtr).data;
    return 0; //success
}
*/


/*
* 检查传入代码经过base64编码后的hmac值是否合法
*/
/*
static int check_hmac(const char * buffer, unsigned int buffer_len)
{
    int i =0;
    char* base64_encode_output;
    unsigned char * hmac = NULL;  
    unsigned int hmac_len = 0;  

    func();
    hmac = (unsigned char*)malloc(EVP_MAX_MD_SIZE);// EVP_MAX_MD_SIZE=64
    if (NULL == hmac) {
        error("Can't malloc %d\n", EVP_MAX_MD_SIZE);
        return 0;
    }
    hmac_encode(key, key_len, buffer, buffer_len, hmac, &hmac_len);

    if(0 == hmac_len) {
        error("HMAC encode failed!\n");
        return 0;
    } else {
        debug("HMAC encode succeeded!\n");
        debug("hmac length is %d\n", hmac_len);
        debug("hmac is :");
        
        // for(int i = 0; i < hmac_len; i++)
        // {
        //     printf("\\x%-02x", (unsigned int)hmac[i]);
        // }
        // printf("\n");
        
    }
    base64_encode(hmac, hmac_len, &base64_encode_output);
    printf("Output (base64): %s\n", base64_encode_output);
    if(hmac)
        free(hmac);
    // compare with the list
    for(i=0; i < (hmac_count/(HMAC_SHA256_BASE64_SIZE+1)); i++) {
        if ( 0 == memcmp((void *)base64_encode_output, 
            (void *)(hmac_list+i*(HMAC_SHA256_BASE64_SIZE+1)), 
            HMAC_SHA256_BASE64_SIZE) ) {
            debug("Found '%s' in hmac list.\n", base64_encode_output);
            return 1;
        }
    }
    debug("Not Found '%s' in hmac list.\n", base64_encode_output);
    return 0;
}
*/


static int get_active_ports_nr(VirtIOSerial *vser)
{
    VirtIOSerialPort *port;
    uint32_t nr_active_ports = 0;
    QTAILQ_FOREACH(port, &vser->ports, next) {
        nr_active_ports++;
    }
    return nr_active_ports;
}

void handler(int sig);
/* system call signal() calls sigaction() actually*/
__sighandler_t my_signal(int sig, __sighandler_t handler);

static VOL *find_vol_by_vaddr(uint64_t vaddr, CudaDev *dev);
static uint64_t map_addr_by_vaddr(uint64_t vaddr, CudaDev *dev);

/*
* bitmap
#include <limits.h> // for CHAR_BIT
*/
static void __set_bit(word_t *word, int n)
{
    word[WORD_OFFSET(n)] |= (1 << BIT_OFFSET(n));
}

static void __clear_bit(word_t *word, int n)
{
    word[WORD_OFFSET(n)] &= ~(1 << BIT_OFFSET(n));
}

static int __get_bit(word_t *word, int n)
{
    word_t bit = word[WORD_OFFSET(n)] & (1 << BIT_OFFSET(n));
    return bit != 0;
}

static void *gpa_to_hva(hwaddr gpa, int len)
{
    hwaddr size = (hwaddr)len;
    void *hva = cpu_physical_memory_map(gpa, &size, 0);
    if (!hva || len != size) {
        error("Failed to map MMIO memory for"
                          " gpa 0x%lx element size %u\n",
                            gpa, len);
        return NULL;
    }
    return hva;
}

static void cuda_register_fatbinary(VirtIOArg *arg, int tid)
{
    void *fat_bin;
    cudaError_t err = -1;
    uint32_t src_size = arg->srcSize;
    func();
    // check out hmac
    if( (fat_bin=gpa_to_hva((hwaddr)(arg->src), src_size))==NULL)
        return;
    debug("fat_bin is 0x%lx\n", *(uint64_t*)fat_bin);
    //    fatBinAddr = gpa_to_hva(arg->addr1);
    //  cpu_physical_memory_rws(arg->addr1, fatBinAddr, len, 0);
    //  fatBinAddr = malloc(len);

    //assert(*(uint64_t*)fat_bin ==  *(uint64_t*)fatBinAddr);
    /* check binary
    if(0==check_hmac(fat_bin, arg->srcSize)) {
        arg->cmd = cudaErrorMissingConfiguration;
        return ;
    }
    */
    int cmd =VIRTIO_CUDA_REGISTERFATBINARY;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    if (err != cudaSuccess) {
        error("init error!\n");
    }
    arg->cmd = err;
}

static void cuda_unregister_fatbinary(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    VOL *vol, *vol2;
    int f_idx;
    func();
    for(f_idx=0; f_idx<cudaFunctionNum[tid]; f_idx++) {
        free(devicesKernels[tid][f_idx].fatbin);
        free(devicesKernels[tid][f_idx].func_name);
        memset(&devicesKernels[tid][f_idx], 0, 
            sizeof(devicesKernels[tid][f_idx]));
    }

    cudaFunctionNum[tid] = 0;
    // free vol list
    list_for_each_entry_safe(vol, vol2, &cudaDevices[tid].vol, list) {
        list_del(&vol->list);
    }
    // free stream
    cudaStreamBitmap[tid] = 0xffffffff;
    memset(cudaStream[tid], 0, sizeof(cudaStream_t)*CudaStreamMaxNum);

    cudaEventBitmap[tid] = 0xffffffff;
    memset(cudaEvent[tid], 0, sizeof(cudaEvent_t)*CudaEventMaxNum);

    int cmd =VIRTIO_CUDA_UNREGISTERFATBINARY;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    if (err != cudaSuccess) {
        error("init error!\n");
    }
    arg->cmd = err;
}

__sighandler_t my_signal(int sig, __sighandler_t handler)
{
    struct sigaction act;
    struct sigaction oldact;
    act.sa_handler = handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;

    if (sigaction(sig, &act, &oldact) < 0)
        return SIG_ERR;

    return oldact.sa_handler; // return previous handler
}

void handler(int sig)
{
    printf("rev sig=%d\n", sig);
    pid_t pid;
    int status;
    pid=waitpid(-1, &status, 0);
    error("Pid %d exited.\n", pid);
    if (WIFEXITED(status)) {
        error("exited, status=%d\n", WEXITSTATUS(status));
    } else if(WIFSIGNALED(status)) {
        error("killed by signal %d\n", WTERMSIG(status));
    } else if(WIFSTOPPED(status)) {
        error("stopped by signal %d\n", WSTOPSIG(status));
    } else if(WIFCONTINUED(status)) {
        error("continued.\n");
    }

}

static void cuda_register_function(VirtIOArg *arg, int tid)
{
    hwaddr fat_bin_gpa;
    hwaddr func_name_gpa;
    uint32_t func_id;
    int fat_size, name_size;
    int nr_func = cudaFunctionNum[tid];
    KernelInfo *kernel;

    func();

    if(nr_func >= CudaFunctionMaxNum) {
        error("kernel number of thread %d is overflow.\n", tid);
        arg->cmd = cudaErrorUnknown;
        return;
    }

    fat_bin_gpa = (hwaddr)(arg->src);
    func_name_gpa = (hwaddr)(arg->dst);
    func_id = arg->flag;
    fat_size = arg->srcSize;
    name_size = arg->dstSize;
    // initialize the KernelInfo
    kernel = &devicesKernels[tid][nr_func];
    kernel->fatbin = malloc(fat_size);
    kernel->fatbin_size = fat_size;
    kernel->func_name = malloc(name_size);
    kernel->func_name_size = name_size;

    cpu_physical_memory_read(fat_bin_gpa,
        kernel->fatbin, fat_size);
    cpu_physical_memory_read(func_name_gpa,
        kernel->func_name, name_size);

    kernel->func_id = func_id;
    debug(  "Loading module... fatbin = %16p, fatbin size=0x%x, name='%s',"
            " name size=0x%x, func_id=%d, nr_func = %d\n", 
            kernel->fatbin, kernel->fatbin_size, kernel->func_name,
            kernel->func_name_size, func_id, nr_func);

    cudaFunctionNum[tid]++;
    arg->cmd = cudaSuccess;
}


static void cuda_setup_argument(VirtIOArg *arg)
{
    func();
}

static void cuda_launch(VirtIOArg *arg, int tid)
{
    cudaError_t err=-1;
    int i=0;
    uint32_t func_id, para_num, para_idx, func_idx;
    uint32_t para_size, conf_size;
    cudaStream_t stream_kernel;
    // hwaddr gpa_para = (hwaddr)(arg->src);
    // hwaddr gpa_conf = (hwaddr)(arg->dst);
    func();
    debug("thread id = %u\n", tid);

    func_id = (uint32_t)(arg->flag);
    debug(" func_id = %u\n", func_id);
    int found = 0;
    for(func_idx=0; func_idx < cudaFunctionNum[tid]; func_idx++) {
        if( devicesKernels[tid][func_idx].func_id == func_id) {
            found = 1;
            break;
        }
    }
    if(!found){
        error("Failed to find func id.\n");
        arg->cmd = cudaErrorInvalidDeviceFunction;
        return;
    }
    debug("Found func_idx = %d.\n", func_idx);
    debug("Found function  = 0x%lx.\n", 
          (uint64_t)devicesKernels[tid][func_idx].func_id);

    para_size = arg->srcSize;
    conf_size = arg->dstSize;
    
    char *para = (char *)gpa_to_hva((hwaddr)(arg->src), para_size);
    if (!para ) {
        arg->cmd = cudaErrorInvalidConfiguration;
        return ;
    }
    KernelConf_t *conf = (KernelConf_t*)gpa_to_hva((hwaddr)arg->dst, 
                                                   conf_size);
    if (!conf) {
        arg->cmd = cudaErrorInvalidConfiguration;
        return ;
    }

    /*
    char *para = (char *)malloc(para_size);
    cpu_physical_memory_read(gpa_para, para, para_size);
    KernelConf_t *conf = (KernelConf_t *)malloc(conf_size);
    cpu_physical_memory_read(gpa_conf, (void *)conf, conf_size);
    */
    para_num = *((uint32_t*)para);
    debug(" para_num = %u\n", para_num);
    
    uint8_t cudaKernelPara[512];
    memcpy(cudaKernelPara, para, para_size);

    void *p=NULL;
    para_idx = sizeof(uint32_t);
    for(i=0; i<para_num; i++) {
        if(*(uint32_t*)(&para[para_idx]) != sizeof(uint64_t))
            continue;
        p = &para[para_idx + sizeof(uint32_t)];
        uint64_t addr = map_addr_by_vaddr(  *(uint64_t*)p, 
                                            &cudaDevices[tid]);
        if(addr!=0) {
            debug("Found 0x%lx\n", addr);
            memcpy( &cudaKernelPara[para_idx + sizeof(uint32_t)], 
                    &addr,
                    sizeof(uint64_t));
        }
        debug("arg %d = 0x%llx , size=%u byte\n", i, 
              *(unsigned long long*)p, 
              *(unsigned int*)(&para[para_idx]));
        para_idx += *(uint32_t*)(&para[para_idx]) + sizeof(uint32_t);
    }

    debug("gridDim=%u %u %u\n", conf->gridDim.x, 
          conf->gridDim.y, conf->gridDim.z);
    debug("blockDim=%u %u %u\n", conf->blockDim.x,
          conf->blockDim.y, conf->blockDim.z);
    debug("sharedMem=%ld\n", conf->sharedMem);
    debug("stream=0x%lx\n", (uint64_t)(conf->stream));
    
    if(!((uint64_t)conf->stream)) {
        stream_kernel = NULL;
    }
    else {
        int pos=(uint64_t)conf->stream;
        if (__get_bit(&cudaStreamBitmap[tid], pos-1)) {
            error("No such stream, pos=%d\n", pos);
            arg->cmd=cudaErrorLaunchFailure;
            return;
        }
        stream_kernel = cudaStream[tid][pos-1];
    }
    debug("now stream=0x%lx\n", (uint64_t)(stream_kernel));
    
    KernelInfo *kernel;
    kernel = &devicesKernels[tid][func_idx];
    // communicate with pipe
    int cmd =VIRTIO_CUDA_LAUNCH;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &kernel->kernel_func, sizeof(CUfunction));
    if (kernel->kernel_func == NULL) {
        // cuError( cuModuleLoadData(&kernel->module, kernel->fatbin) );
        write(pfd[tid][WRITE], &kernel->fatbin_size, 4);
        write(pfd[tid][WRITE], kernel->fatbin, kernel->fatbin_size);
        read(cfd[tid][READ], &err, sizeof(cudaError_t));
        if (err != cudaSuccess) {
            error("register module failed\n");
            arg->cmd = err;
            return;
        }
        // cuError( cuModuleGetFunction(
        //     &kernel->kernel_func,
        //     kernel->module,
        //     kernel->func_name) );
        write(pfd[tid][WRITE], &kernel->func_name_size, 4);
        write(pfd[tid][WRITE], kernel->func_name, kernel->func_name_size);
        read(cfd[tid][READ], &err, sizeof(cudaError_t));
        if (err != cudaSuccess) {
            error("register function failed\n");
            arg->cmd = err;
            return;
        }
        read(cfd[tid][READ], &kernel->kernel_func, sizeof(CUfunction));
    }
    write(pfd[tid][WRITE], &conf->gridDim, sizeof(dim3));
    write(pfd[tid][WRITE], &conf->blockDim, sizeof(dim3));
    write(pfd[tid][WRITE], &conf->sharedMem, sizeof(size_t));
    write(pfd[tid][WRITE], &stream_kernel, sizeof(cudaStream_t));
    write(pfd[tid][WRITE], &para_size, 4);
    write(pfd[tid][WRITE], &cudaKernelPara, para_size);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    if (err != cudaSuccess) {
        error("launch error.\n");
    }
    arg->cmd = err;
}

static VOL *find_vol_by_vaddr(uint64_t vaddr, CudaDev *dev)
{
    VOL *vol;
    list_for_each_entry(vol, &dev->vol, list) {
        if(vol->v_addr <= vaddr && vaddr < (vol->v_addr+vol->size) )
            goto out;
    }
    vol = NULL;
out:
    return vol;
}

static HVOL *find_hvol_by_vaddr(uint64_t vaddr, CudaDev *dev)
{
    HVOL *hvol;
    list_for_each_entry(hvol, &dev->host_vol, list) {
        if(hvol->virtual_addr <= vaddr && vaddr < (hvol->virtual_addr+hvol->size) )
            goto out;
    }
    hvol = NULL;
out:
    return hvol;
}

static uint64_t map_addr_by_vaddr(uint64_t vaddr, CudaDev *dev)
{
    VOL *vol = find_vol_by_vaddr(vaddr, dev);
    if(vol != NULL)
        return vol->addr + (vaddr - vol->v_addr);
    return 0;
}

static void cuda_memcpy(VirtIOArg *arg, int tid)
{
    cudaError_t err;
    uint32_t size;
    void *src, *dst;
    uint64_t *gpa_array=NULL;
    int i=0;
    uint64_t addr=0;
    func();

    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, param=0x%lx\n",
          arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, arg->param);
    size = arg->srcSize;
    if (arg->flag == cudaMemcpyHostToDevice) {
        // host address
        src = malloc(size);
        if(arg->param) {
            int blocks = arg->param;
            gpa_array = (uint64_t*)gpa_to_hva((hwaddr)arg->src, blocks);
            if(!gpa_array) {
                error("No such address.\n");
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
            uint32_t offset = arg->dstSize;
            uint32_t start_offset = offset % KMALLOC_SIZE;
            int len = min(size, KMALLOC_SIZE - start_offset);
            cpu_physical_memory_read((hwaddr)gpa_array[0], src, len);
            int rsize=size;
            rsize-=len;
            offset=len;
            i=1;
            while(rsize) {
                len=min(rsize, KMALLOC_SIZE);
                cpu_physical_memory_read((hwaddr)gpa_array[i++],
                                         src+offset, len);
                offset+=len;
                rsize-=len;
            }
            assert(i == blocks);
        } else {
            cpu_physical_memory_read((hwaddr)arg->src, src, size);
        }
        // device address
        if( (addr = map_addr_by_vaddr(arg->dst, &cudaDevices[tid]))==0) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        dst = (void *)addr;
        // cuError( (err= cuMemcpyHtoD((CUdeviceptr)dst, src, size)));
        int cmd = VIRTIO_CUDA_MEMCPY;
        write(pfd[tid][WRITE], &cmd, 4);
        write(pfd[tid][WRITE], &arg->flag, 4);
        write(pfd[tid][WRITE], &size, 4);
        write(pfd[tid][WRITE], src, size);
        write(pfd[tid][WRITE], &dst, sizeof(void *));

        while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
        arg->cmd = err;
        if(err != cudaSuccess) {
            error("memcpy error!\n");
        }
    } else if (arg->flag == cudaMemcpyDeviceToHost) {
        // get host address
        dst = malloc(size);
        // get device address
        if( (addr = map_addr_by_vaddr(arg->src, &cudaDevices[tid]))==0) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        src = (void*)addr;
        // cuError( (err=cuMemcpyDtoH(dst, (CUdeviceptr)src, size)) );
        int cmd = VIRTIO_CUDA_MEMCPY;
        write(pfd[tid][WRITE], &cmd, 4);
        write(pfd[tid][WRITE], &arg->flag, 4);
        write(pfd[tid][WRITE], &size, 4);
        write(pfd[tid][WRITE], &src, sizeof(void *));

        while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
        arg->cmd = err;
        if (cudaSuccess != err) {
            error("memcpy error!\n");
            return;
        }
        read(cfd[tid][READ], dst, size);
        // copy back to VM
        if(arg->param) {
            int blocks = arg->param;
            gpa_array = (uint64_t*)gpa_to_hva((hwaddr)arg->dst, blocks);
            if(!gpa_array) {
                error("Failed to get gpa_array.\n");
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
            uint32_t offset = arg->dstSize;
            uint32_t start_offset = offset % KMALLOC_SIZE;
            int len = min(size, KMALLOC_SIZE - start_offset);
            cpu_physical_memory_write((hwaddr)gpa_array[0], dst, len);
            int rsize=size;
            rsize-=len;
            offset=len;
            i=1;
            while(rsize) {
                len=min(rsize, KMALLOC_SIZE);
                cpu_physical_memory_write((hwaddr)gpa_array[i++], 
                                          dst+offset, len);
                offset+=len;
                rsize-=len;
            }
            assert(i == blocks);
        } else {
            cpu_physical_memory_write((hwaddr)arg->dst, dst, size);
        }
    } else if (arg->flag == cudaMemcpyDeviceToDevice) {
        if( (addr = map_addr_by_vaddr(arg->src, &cudaDevices[tid]))==0) {
            error("Failed to find src virtual address %p in vol\n",
                  (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        src = (void*)addr;
        if((addr=map_addr_by_vaddr(arg->dst, &cudaDevices[tid]))==0) {
            error("Failed to find dst virtual address %p in vol\n",
                  (void *)arg->dst);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        dst = (void*)addr;
        // cuError( (err=cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, size)) );
        int cmd = VIRTIO_CUDA_MEMCPY;
        write(pfd[tid][WRITE], &cmd, 4);
        write(pfd[tid][WRITE], &arg->flag, 4);
        write(pfd[tid][WRITE], &src, sizeof(void *));
        write(pfd[tid][WRITE], &dst, sizeof(void *));
        write(pfd[tid][WRITE], &size, 4);

        while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
        arg->cmd = err;
        if (cudaSuccess != err) {
            error("memcpy error!\n");
        }
    } else {
        error("Error memcpy direction\n");
        arg->cmd = cudaErrorInvalidMemcpyDirection;
    }
}

static void cuda_memcpy_async(VirtIOArg *arg, int tid)
{
    cudaError_t err=-1;
    uint32_t size;
    void *src, *dst;
    int pos = 0;
    uint64_t addr = 0;
    uint64_t *gpa_array=NULL;
    int i=0;
    cudaStream_t stream=0;
    uint32_t init_offset=0, blocks=0;
    func();
    
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=%lu, "
        "stream=0x%lx \n",
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, arg->param);
    blocks = arg->param >> 32;
    if(blocks==0) {
        error("Failed to get blocks\n");
        arg->cmd = cudaErrorInvalidValue;
        return ;
    }
    init_offset = arg->param & 0xffffffff;
    debug("pos = 0x%x, blocks=0x%x, offset=0x%x\n",
          arg->dstSize, blocks, init_offset);
    pos = arg->dstSize;
    if (pos==0) {
        stream=0;
    } else if (!__get_bit(&cudaStreamBitmap[tid], pos-1)) {
        stream = cudaStream[tid][pos-1];
    } else {
        error("No such stream, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    debug("stream 0x%lx\n", (uint64_t)stream);
    size = arg->srcSize;
    if (arg->flag == cudaMemcpyHostToDevice) {
        src = malloc(size);
        gpa_array = (uint64_t*)gpa_to_hva((hwaddr)arg->src, blocks);
        if(!gpa_array) {
            error("No such addr!\n");
            free(src);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        uint32_t start_offset = init_offset % KMALLOC_SIZE;
        int len = min(size, KMALLOC_SIZE - start_offset);
        cpu_physical_memory_read((hwaddr)gpa_array[0], src, len);
        int rsize=size;
        int offset=len;
        rsize-=len;
        i=1;
        while(rsize) {
            len=min(rsize, KMALLOC_SIZE);
            cpu_physical_memory_read((hwaddr)gpa_array[i++],
                                     src+offset, len);
            offset+=len;
            rsize-=len;
        }
        assert(i == blocks);
        
        if((addr=map_addr_by_vaddr(arg->dst, &cudaDevices[tid]))==0) {
            error("Failed to find dst virtual addr %p in vol\n",
                  (void *)arg->dst);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        dst = (void *)addr;
        // cuError( (err= cuMemcpyHtoDAsync((CUdeviceptr)dst, (void *)src, size, 
                                         // stream)));
        int cmd = VIRTIO_CUDA_MEMCPY_ASYNC;
        write(pfd[tid][WRITE], &cmd, 4);
        write(pfd[tid][WRITE], &arg->flag, 4);
        write(pfd[tid][WRITE], &stream, sizeof(cudaStream_t));
        write(pfd[tid][WRITE], &size, 4);
        write(pfd[tid][WRITE], src, size);
        write(pfd[tid][WRITE], &dst, sizeof(void *));

        while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
        arg->cmd = err;
        if(err != cudaSuccess) {
            error("memcpy async HtoD error!\n");
        }
        free(src);
    } else if (arg->flag == cudaMemcpyDeviceToHost) {
        if((src=(void*)map_addr_by_vaddr(arg->src, &cudaDevices[tid]))==0) {
            error("Failed to find virtual addr %p in vol or hvol\n",
                  (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        /* try to fetch dst guest physical address*/
        gpa_array = (uint64_t*)gpa_to_hva((hwaddr)arg->param2, blocks);
        if(!gpa_array) {
            error("No such dst physical address 0x%lx.\n", arg->param2);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }

        dst = NULL;
        void *gdst=NULL;
        HVOL *hvol = find_hvol_by_vaddr(arg->dst, &cudaDevices[tid]);
        if (hvol) {
            int offset = arg->dst - hvol->virtual_addr;
            dst = (void*)(hvol->actual_addr+offset);
            gdst = (void*)(hvol->native_addr+ offset);
        }
        // cuError( (err=cuMemcpyDtoHAsync((void *)dst, (CUdeviceptr)src, 
        //                                 size, stream)) );
        debug("src = %p\n", src);
        debug("dst = %p\n", dst);
        debug("size = %x\n", size);
        int cmd = VIRTIO_CUDA_MEMCPY_ASYNC;
        write(pfd[tid][WRITE], &cmd, 4);
        write(pfd[tid][WRITE], &arg->flag, 4);
        write(pfd[tid][WRITE], &stream, sizeof(cudaStream_t));
        write(pfd[tid][WRITE], &size, 4);
        write(pfd[tid][WRITE], &src, sizeof(void *));
        write(pfd[tid][WRITE], &dst, sizeof(void *));

        while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
        arg->cmd = err;
        if (err != cudaSuccess) {
            error("memcpy async DtoH error!\n");
            return;
        }
        // copy back to VM
        if (dst==NULL) {
            gdst = malloc(size);
            read(cfd[tid][READ], gdst, size);
        }
        uint32_t start_offset = init_offset % KMALLOC_SIZE;
        int len = min(size, KMALLOC_SIZE - start_offset);
        cpu_physical_memory_write((hwaddr)gpa_array[0], gdst, len);
        int rsize=size;
        uint32_t offset=len;
        rsize-=len;
        i=1;
        while(rsize) {
            len=min(rsize, KMALLOC_SIZE);
            cpu_physical_memory_write((hwaddr)gpa_array[i++], 
                                      gdst+offset, len);
            offset+=len;
            rsize-=len;
        }
        assert(i == blocks);
        if (dst == NULL)
            free(gdst);
    } else if (arg->flag == cudaMemcpyDeviceToDevice) {
        if((addr=map_addr_by_vaddr(arg->src, &cudaDevices[tid]))==0) {
            error("Failed to find virtual addr %p in vol\n",
                  (void *)arg->src);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        src = (void*)addr;
        if((addr = map_addr_by_vaddr(arg->dst, &cudaDevices[tid]))==0) {
            error("Failed to find virtual addr %p in vol\n",
                  (void *)arg->dst);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        dst = (void*)addr;
        // cuError( (err=cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, 
        //                                 size, stream)) );
        int cmd = VIRTIO_CUDA_MEMCPY_ASYNC;
        write(pfd[tid][WRITE], &cmd, 4);
        write(pfd[tid][WRITE], &arg->flag, 4);
        write(pfd[tid][WRITE], &stream, sizeof(cudaStream_t));
        write(pfd[tid][WRITE], &src, sizeof(void *));
        write(pfd[tid][WRITE], &dst, sizeof(void *));
        write(pfd[tid][WRITE], &size, 4);

        while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
        if (err != cudaSuccess) {
            error("memcpy async DtoD error!\n");
            arg->cmd = err;
            return;
        }
    } else {
        error("No such memcpy direction.\n");
        arg->cmd= cudaErrorInvalidMemcpyDirection;
    }
}

static void cuda_memset(VirtIOArg *arg, int tid)
{
    cudaError_t err=-1;
    size_t count;
    int value;
    uint64_t dst;
    func();

    count = (size_t)(arg->dstSize);
    value = (int)(arg->param);
    debug("dst=0x%lx, value=0x%x, count=0x%lx\n", arg->dst, value, count);
    if((dst = map_addr_by_vaddr(arg->dst, &cudaDevices[tid]))==0) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = cudaErrorInvalidValue;
        return;
    }
    // communicate with pipe
    int cmd = VIRTIO_CUDA_MEMSET;
    write(pfd[tid][WRITE], &cmd, 4);
    // cuError( (err= cudaMemset((void*)dst, value, count)));
    write(pfd[tid][WRITE], &count, sizeof(size_t));
    write(pfd[tid][WRITE], &value, sizeof(int));
    write(pfd[tid][WRITE], &dst, sizeof(uint64_t));
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess)
        error("memset memory error!\n");
}

static void cuda_malloc(VirtIOArg *arg, int tid)
{
    cudaError_t err=-1;
    void *devPtr=NULL;
    size_t size;
    VOL *vol;
    func();

    size = arg->srcSize; 
    // communicate with pipe
    int cmd = VIRTIO_CUDA_MALLOC;
    write(pfd[tid][WRITE], &cmd, 4);
    // cudaError( (err= cudaMalloc((void **)&devPtr, size)));
    write(pfd[tid][WRITE], &size, sizeof(size_t));
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("Alloc memory error!\n");
        return;
    }
    read(cfd[tid][READ], &devPtr, sizeof(void *));

    vol = (VOL *)malloc(sizeof(VOL));
    vol->addr = (uint64_t)devPtr;
    vol->v_addr = (uint64_t)(devPtr + VOL_OFFSET);
    arg->dst =  (uint64_t)(vol->v_addr);
    vol->size = size;
    list_add_tail(&vol->list, &cudaDevices[tid].vol);
    debug("actual devPtr=0x%lx, virtual ptr=0x%lx, size=0x%lx,"
          "ret value=0x%x\n", (uint64_t)devPtr, arg->dst, size, err);
}

static void* get_shm(size_t size, char *file_path)
{
    int mmap_fd = shm_open(file_path, O_RDWR, 0);
    if (mmap_fd == -1) {
        error("Failed to open.\n");
        return NULL;
    }
    // extend
    void *addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, mmap_fd, 0);
    if (addr == MAP_FAILED) {
        error("Failed to mmap.\n");
        return NULL;
    }
    return addr;
}

static void* set_shm(size_t size, char *file_path)
{
    int res=0;
    int mmap_fd = shm_open(file_path, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
    if (mmap_fd == -1) {
        error("Failed to open.\n");
        return NULL;
    }
    // extend
    res = ftruncate(mmap_fd, size);
    if (res == -1) {
        error("Failed to ftruncate.\n");
        return NULL;
    }
    // map shared memory to address space
    void *addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, mmap_fd, 0);
    if (addr == MAP_FAILED) {
        error("Failed to mmap.\n");
        return NULL;
    }
    return addr;
}

static void unset_shm(void *addr, size_t size, char *file_path)
{
    // mmap cleanup
    int res = munmap(addr, size);
    if (res == -1) {
        error("Failed to munmap.\n");
        return;
    }
    // shm_open cleanup
    int fd = shm_unlink(file_path);
    if(fd == -1) {
        error("Failed to unlink.\n");
        return;
    }
}

static void cuda_host_register(VirtIOArg *arg, int tid)
{
    cudaError_t err=-1;
    size_t size;
    void *src;
    char file_path[64];

    func();
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, param=0x%lx\n",
          arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, arg->param);
    size = arg->srcSize;

    sprintf(file_path, "/qemu-%d-%lx", tid, arg->src);
    printf("file path = %s\n", file_path);
    src = set_shm(size, file_path);
    if (src == NULL) {
        error("Failed to allocate share memroy.\n");
        arg->cmd = cudaErrorMemoryAllocation;
        return;
    }

    int cmd = VIRTIO_CUDA_HOSTREGISTER;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &arg->flag, sizeof(unsigned int));
    write(pfd[tid][WRITE], &size, sizeof(size_t));
    write(pfd[tid][WRITE], file_path, 64);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("host register error.\n");
    }
    void *actual_addr;
    read(cfd[tid][READ], &actual_addr, sizeof(void *));
    HVOL *hvol = (HVOL *)malloc(sizeof(HVOL));
    hvol->actual_addr = (uint64_t)actual_addr;
    hvol->native_addr = (uint64_t)src;
    hvol->virtual_addr = arg->src;
    hvol->size = size;
    memcpy(hvol->file_path, file_path, 64);
    list_add_tail(&hvol->list, &cudaDevices[tid].host_vol);
}

static void cuda_host_unregister(VirtIOArg *arg, int tid)
{
    cudaError_t err=-1;
    char file_path[64];

    func();
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, param=0x%lx\n",
          arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, arg->param);

    sprintf(file_path, "/qemu-%d-%lx", tid, arg->src);

    int fd = shm_open(file_path, O_RDONLY, 0);
    if (fd == -1) {
        error("Failed to open file %s, file does not exist.\n", file_path);
        return;
    }

    HVOL *hvol, *hvol2;
    list_for_each_entry_safe(hvol, hvol2, &cudaDevices[tid].host_vol, list) {
        if (strcmp(hvol->file_path, file_path)==0 && 
            hvol->virtual_addr == arg->src ) {
            debug(  "actual actual addr=0x%lx, virtual ptr=0x%lx\n", 
                    (uint64_t)hvol->actual_addr, (uint64_t)hvol->virtual_addr);
            int cmd = VIRTIO_CUDA_HOSTUNREGISTER;
            write(pfd[tid][WRITE], &cmd, 4);
            write(pfd[tid][WRITE], &hvol->actual_addr, sizeof(uint64_t));
            while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
            arg->cmd = err;
            if (err != cudaSuccess) {
                error("free error.\n");
            }
            list_del(&hvol->list);
            unset_shm((void*)hvol->native_addr, hvol->size, file_path);
            return;
        }
    }

}
static void cuda_free(VirtIOArg *arg, int tid)
{
    cudaError_t err=-1;
    uint64_t src;
    VOL *vol, *vol2;
    func();

    src = arg->src;
    debug(" ptr = 0x%lx\n", arg->src);
    list_for_each_entry_safe(vol, vol2, &cudaDevices[tid].vol, list) {
        if (vol->v_addr == src) {
            debug(  "actual devPtr=0x%lx, virtual ptr=0x%lx\n", 
                    (uint64_t)vol->addr, src);
            int cmd = VIRTIO_CUDA_FREE;
            write(pfd[tid][WRITE], &cmd, 4);
            // cudaError( (err= cudaFree((void*)(vol->addr))) );
            write(pfd[tid][WRITE], &vol->addr, sizeof(uint64_t));
            while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
            if (err != cudaSuccess) {
                error("free error.\n");
            }
            list_del(&vol->list);
            arg->cmd = err;
            return;
        }
    }
    arg->cmd = cudaErrorInvalidValue;
    error("Failed to find ptr!\n");
}

/*
 * Let vm user see which card he actually uses
*/
static void cuda_get_device(VirtIOArg *arg, int tid)
{
    int dev = 0;
    hwaddr gpa = (hwaddr)(arg->dst);

    arg->cmd = cudaSuccess;
    dev = (int)(cudaDevices[tid].device);
    cpu_physical_memory_write(gpa, &dev, sizeof(int));
}

/*
 * done by the vgpu in vm
 * this function is useless
*/
static void cuda_get_device_properties(VirtIOArg *arg)
{
    cudaError_t err;
    int devID;
    struct cudaDeviceProp prop;
    hwaddr gpa = (hwaddr)(arg->dst);
    func();
    devID = (int)(arg->flag);
    debug("Get prop for device %d\n", devID);
    cudaError( (err=cudaGetDeviceProperties(&prop, devID)) );
    debug("Device %d : \"%s\" with compute %d.%d capability.\n", 
          devID, prop.name, prop.major, prop.minor);
    arg->cmd = err;
    cpu_physical_memory_write(gpa, &prop, arg->dstSize);
}

static void cuda_set_device(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    func();
    int dev_id = (int)(arg->flag);
    if (dev_id < 0 || dev_id > total_device) {
        error("setting error device = %d\n", dev_id);
        arg->cmd = cudaErrorInvalidDevice;
        return ;
    }
    
    cudaDevices[tid].device = (CUdevice)dev_id;

    int cmd = VIRTIO_CUDA_SETDEVICE;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &dev_id, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("set device error.\n");
        return;
    }
    debug("set devices=%d\n", (int)(arg->flag));
    /* clear kernel function addr in parent process, 
    because cudaSetDevice will clear all resources related with host thread.
    */
    int func_idx = 0;
    for(func_idx=0; func_idx < cudaFunctionNum[tid]; func_idx++)
        devicesKernels[tid][func_idx].kernel_func=NULL;
}

static void cuda_set_device_flags(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    func();
    unsigned int flags = (unsigned int)arg->flag;
    debug("set devices flags=%d\n", flags);
    int cmd = VIRTIO_CUDA_SETDEVICEFLAGS;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &flags, sizeof(unsigned int));
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("set device flags error.\n");
    }
}

static void cuda_get_device_count(VirtIOArg *arg)
{
    func();
    debug("Device count=%d.\n", total_device);
    arg->cmd = (int32_t)cudaSuccess;
    arg->flag = (uint64_t)total_device;
}

static void cuda_device_reset(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    func();
    /* TO DO
    * should not use
    * cuCtxDestroy(cudaDevices[worker_cur_device[tid]].context ) ;
    * reinit all global variables
    */
    // free memory
    VOL *vol, *vol2;
    list_for_each_entry_safe(vol, vol2, &cudaDevices[tid].vol, list) {
        // cudaError( cudaFree((void*)(vol->addr))) ;
        list_del(&vol->list);
    }
    // free stream
/*    for(int pos = 1; pos <= BITS_PER_WORD; pos++) {
        if(!__get_bit(&cudaStreamBitmap[tid], pos-1)) {
            cudaError( cudaStreamDestroy(cudaStream[tid][pos-1]) );
        }
    }*/
    // cudaStreamBitmap[tid] = 0xfffffffe;
    cudaStreamBitmap[tid] = 0xffffffff;
    memset(cudaStream[tid], 0, sizeof(cudaStream_t)*CudaStreamMaxNum);
    // free event
    /*for(int pos = 1; pos <= BITS_PER_WORD; pos++) {
        if(!__get_bit(&cudaEventBitmap[tid], pos-1)) {
            cudaError( cudaEventDestroy(cudaEvent[tid][pos-1]) );
        }
    }*/
    cudaEventBitmap[tid] = 0xffffffff;
    memset(cudaEvent[tid], 0, sizeof(cudaEvent_t)*CudaEventMaxNum);
    int cmd = VIRTIO_CUDA_DEVICERESET;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("create stream error.\n");
    }
    debug("reset devices\n");
}

static void cuda_stream_create(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    func();
    pos = ffs(cudaStreamBitmap[tid]);
    if (!pos) {
        error("stream number is up to %d\n", CudaStreamMaxNum);
        return;
    }
    //cudaError( (err = cudaStreamCreate(&cudaStream[tid][pos-1]) ));
    int cmd = VIRTIO_CUDA_STREAMCREATE;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("create stream error.\n");
        return;
    }
    read(cfd[tid][READ], &cudaStream[tid][pos-1], sizeof(cudaStream_t));

    arg->flag = (uint64_t)pos;
    debug("create stream 0x%lx, idx is %u\n",
          (uint64_t)cudaStream[tid][pos-1], pos-1);
    __clear_bit(&cudaStreamBitmap[tid], pos-1);
}

static void cuda_stream_destroy(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    uint32_t pos;
    func();
    pos = arg->flag;
    if (__get_bit(&cudaStreamBitmap[tid], pos-1)) {
        error("No such stream, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    debug("destroy stream 0x%lx\n", (uint64_t)cudaStream[tid][pos-1]);
    // cudaError( (err=cudaStreamDestroy(cudaStream[tid][pos-1]) ));
    int cmd = VIRTIO_CUDA_STREAMDESTROY;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &cudaStream[tid][pos-1], sizeof(cudaStream_t));
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("destroy stream error.\n");
        return;
    }
    __set_bit(&cudaStreamBitmap[tid], pos-1);
}

static void cuda_event_create(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    func();
    pos = ffs(cudaEventBitmap[tid]);
    if(!pos) {
        error("event number is up to %d\n", CudaEventMaxNum);
        return;
    }
    int cmd = VIRTIO_CUDA_EVENTCREATE;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("create event error.\n");
        return;
    }
    read(cfd[tid][READ], &cudaEvent[tid][pos-1], sizeof(cudaEvent_t));
    arg->flag = (uint64_t)pos;
    __clear_bit(&cudaEventBitmap[tid], pos-1);
    debug("create event 0x%lx, idx is %u\n",
          (uint64_t)cudaEvent[tid][pos-1], pos-1);
}

static void cuda_event_create_with_flags(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    unsigned int flag=0;
    func();
    pos = ffs(cudaEventBitmap[tid]);
    if(!pos) {
        error("event number is up to %d\n", CudaEventMaxNum);
        return;
    }
    flag = arg->flag;
    // cudaError( (err=cudaEventCreateWithFlags(&cudaEvent[tid][pos-1], flag) ));
    int cmd = VIRTIO_CUDA_EVENTCREATEWITHFLAGS;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &flag, sizeof(unsigned int));
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("create event with flags error.\n");
        return;
    }
    read(cfd[tid][READ], &cudaEvent[tid][pos-1], sizeof(cudaEvent_t));
    arg->dst = (uint64_t)pos;
    __clear_bit(&cudaEventBitmap[tid], pos-1);
    debug("create event 0x%lx with flag %u, idx is %u\n",
          (uint64_t)cudaEvent[tid][pos-1], flag, pos-1);
}

static void cuda_event_destroy(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    func();
    pos = arg->flag;
    if (__get_bit(&cudaEventBitmap[tid], pos-1)) {
        error("No such event, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    debug("destroy event 0x%lx\n", (uint64_t)cudaEvent[tid][pos-1]);
    // cudaError( (err=cudaEventDestroy(cudaEvent[tid][pos-1])) );
    int cmd = VIRTIO_CUDA_EVENTDESTROY;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &cudaEvent[tid][pos-1], sizeof(cudaEvent_t));
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("destroy event error.\n");
        return;
    }
    __set_bit(&cudaEventBitmap[tid], pos-1);
}

static void cuda_event_record(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    uint64_t epos = 0, spos = 0;
    cudaStream_t stream;
    func();
    epos = arg->src;
    spos = arg->dst;
    debug("event pos = 0x%lx\n", epos);
    if (epos<=0 || __get_bit(&cudaEventBitmap[tid], epos-1)) {
        error("No such event, pos=0x%lx\n", epos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    debug("stream pos = 0x%lx\n", spos);
    if (spos==0) {
        stream=0;
    } else if (!__get_bit(&cudaStreamBitmap[tid], spos-1)) {
        stream = cudaStream[tid][spos-1];
    } else {
        error("No such stream, pos=0x%lx\n", spos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    debug("record event 0x%lx, stream=0x%lx\n",
          (uint64_t)cudaEvent[tid][epos-1], (uint64_t)stream);
    // cudaError((err=cudaEventRecord(cudaEvent[tid][epos-1], 
                                   // cudaStream[tid][spos-1])));
    int cmd = VIRTIO_CUDA_EVENTRECORD;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &cudaEvent[tid][epos-1], sizeof(cudaEvent_t));
    write(pfd[tid][WRITE], &stream, sizeof(cudaStream_t));
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("record event error.\n");
        return;
    }
}

static void cuda_event_synchronize(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    func();
    pos = arg->flag;
    if (__get_bit(&cudaEventBitmap[tid], pos-1)) {
        error("No such event, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    debug("sync event 0x%lx\n", (uint64_t)cudaEvent[tid][pos-1]);
    // cudaError( (err=cudaEventSynchronize(cudaEvent[tid][pos-1])) );
    int cmd = VIRTIO_CUDA_EVENTSYNCHRONIZE;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &cudaEvent[tid][pos-1], sizeof(cudaEvent_t));
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("synchronize event error.\n");
        return;
    }
}

static void cuda_event_elapsedtime(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    int start_pos, stop_pos;
    float time = 0;
    func();
    start_pos = arg->src;
    stop_pos = arg->dst;
    if (__get_bit(&cudaEventBitmap[tid], start_pos-1)) {
        error("No such event, pos=%d\n", start_pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    if (__get_bit(&cudaEventBitmap[tid], stop_pos-1)) {
        error("No such event, pos=%d\n", stop_pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    debug("start event 0x%lx\n", (uint64_t)cudaEvent[tid][start_pos-1]);
    debug("stop event 0x%lx\n", (uint64_t)cudaEvent[tid][stop_pos-1]);
    // cudaError( (err=cudaEventElapsedTime(&time, 
    //                                      cudaEvent[tid][start_pos-1], 
    //                                      cudaEvent[tid][stop_pos-1])) );
    int cmd = VIRTIO_CUDA_EVENTELAPSEDTIME;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &cudaEvent[tid][start_pos-1], sizeof(cudaEvent_t));
    write(pfd[tid][WRITE], &cudaEvent[tid][stop_pos-1], sizeof(cudaEvent_t));
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("event calc elapsed time error.\n");
        return;
    }
    read(cfd[tid][READ], &time, sizeof(float));
    arg->flag = (uint64_t)time;
}

static void cuda_device_synchronize(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    func();
    int cmd =VIRTIO_CUDA_DEVICESYNCHRONIZE;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
}

static void cuda_thread_synchronize(VirtIOArg *arg, int tid)
{
    func();
    /*
    * cudaThreadSynchronize is deprecated
    * cudaError( (err=cudaThreadSynchronize()) );
    */
    cuda_device_synchronize(arg, tid);
}

static void cuda_get_last_error(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    func();
    // communicate with pipe
    int cmd =VIRTIO_CUDA_GETLASTERROR;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
}

static void cuda_mem_get_info(VirtIOArg *arg, int tid)
{
    cudaError_t err = -1;
    size_t freeMem, totalMem;
    func();
    int cmd =VIRTIO_CUDA_MEMGETINFO;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("get mem info error!\n");
        return;
    }
    read(cfd[tid][READ], &freeMem, sizeof(size_t));
    read(cfd[tid][READ], &totalMem, sizeof(size_t));
    arg->srcSize = freeMem;
    arg->dstSize = totalMem;
    debug("free memory = %lu, total memory = %lu.\n", freeMem, totalMem);
}

/*
static inline void cpu_physical_memory_read(hwaddr addr,
                                            void *buf, int len)

static inline void cpu_physical_memory_write(hwaddr addr,
                                             const void *buf, int len)
*/
static void cuda_gpa_to_hva(VirtIOArg *arg)
{
    int a;
    hwaddr gpa = (hwaddr)(arg->src);
    //a = gpa_to_hva(arg->src);
    cpu_physical_memory_read(gpa, &a, sizeof(int));
    debug("a=%d\n", a);
    a++;
    cpu_physical_memory_write(gpa, &a, sizeof(int));
    arg->cmd = 1;//(int32_t)cudaSuccess;
}

/* 
*   Callback function that's called when the guest sends us data.  
 * Guest wrote some data to the port. This data is handed over to
 * the app via this callback.  The app can return a size less than
 * 'len'.  In this case, throttling will be enabled for this port.
 */
static ssize_t flush_buf(VirtIOSerialPort *port,
                         const uint8_t *buf, ssize_t len)
{
    int tid;
    int ret;

    debug("port->id=%d, len=%ld\n", port->id, len);
    tid = port->id;// % total_port;
    if (len != sizeof(VirtIOArg)) {
        error("buf len should be %lu, not %ld\n", sizeof(VirtIOArg), len);
        return 0;
    }

    VirtIOArg *msg = (VirtIOArg *)malloc(len);
    memcpy((void *)msg, (void *)buf, len);

    switch(msg->cmd) {
        case VIRTIO_CUDA_HELLO:
        cuda_gpa_to_hva(msg);
        break;
    case VIRTIO_CUDA_REGISTERFATBINARY:
        cuda_register_fatbinary(msg, tid);
        break;
    case VIRTIO_CUDA_UNREGISTERFATBINARY:
        cuda_unregister_fatbinary(msg, tid);
        break;
    case VIRTIO_CUDA_REGISTERFUNCTION:
        cuda_register_function(msg, tid);
        break;
    case VIRTIO_CUDA_LAUNCH:
        cuda_launch(msg, tid);
        break;
    case VIRTIO_CUDA_MALLOC:
        cuda_malloc(msg, tid);
        break;
    case VIRTIO_CUDA_HOSTREGISTER:
        cuda_host_register(msg, tid);
        break;
    case VIRTIO_CUDA_HOSTUNREGISTER:
        cuda_host_unregister(msg, tid);
        break;
    case VIRTIO_CUDA_MEMCPY:
        cuda_memcpy(msg, tid);
        break;
    case VIRTIO_CUDA_FREE:
        cuda_free(msg, tid);
        break;
    case VIRTIO_CUDA_GETDEVICE:
        cuda_get_device(msg, tid);
        break;
    case VIRTIO_CUDA_GETDEVICEPROPERTIES:
        cuda_get_device_properties(msg);
        break;
    case VIRTIO_CUDA_CONFIGURECALL:
        break;
    case VIRTIO_CUDA_SETUPARGUMENT:
        cuda_setup_argument(msg);
        break;
    case VIRTIO_CUDA_GETDEVICECOUNT:
        cuda_get_device_count(msg);
        break;
    case VIRTIO_CUDA_SETDEVICE:
        cuda_set_device(msg, tid);
        break;
    case VIRTIO_CUDA_SETDEVICEFLAGS:
        cuda_set_device_flags(msg, tid);
        break;
    case VIRTIO_CUDA_DEVICERESET:
        cuda_device_reset(msg, tid);
        break;
    case VIRTIO_CUDA_STREAMCREATE:
        cuda_stream_create(msg, tid);
        break;
    case VIRTIO_CUDA_STREAMDESTROY:
        cuda_stream_destroy(msg, tid);
        break;
    case VIRTIO_CUDA_EVENTCREATE:
        cuda_event_create(msg, tid);
        break;
    case VIRTIO_CUDA_EVENTCREATEWITHFLAGS:
        cuda_event_create_with_flags(msg, tid);
        break;
    case VIRTIO_CUDA_EVENTDESTROY:
        cuda_event_destroy(msg, tid);
        break;
    case VIRTIO_CUDA_EVENTRECORD:
        cuda_event_record(msg, tid);
        break;
    case VIRTIO_CUDA_EVENTSYNCHRONIZE:
        cuda_event_synchronize(msg, tid);
        break;
    case VIRTIO_CUDA_EVENTELAPSEDTIME:
        cuda_event_elapsedtime(msg, tid);
        break;
    case VIRTIO_CUDA_THREADSYNCHRONIZE:
        cuda_thread_synchronize(msg, tid);
        break;
    case VIRTIO_CUDA_GETLASTERROR:
        cuda_get_last_error(msg, tid);
        break;
    case VIRTIO_CUDA_MEMCPY_ASYNC:
        cuda_memcpy_async(msg, tid);
        break;
    case VIRTIO_CUDA_MEMSET:
        cuda_memset(msg, tid);
        break;
    case VIRTIO_CUDA_DEVICESYNCHRONIZE:
        cuda_device_synchronize(msg, tid);
        break;
    case VIRTIO_CUDA_MEMGETINFO:
        cuda_mem_get_info(msg, tid);
        break;
    default:
        error("[+] header.cmd=%u, nr= %u \n",
              msg->cmd, _IOC_NR(msg->cmd));
        return 0;
    }
    ret = virtio_serial_write(port, (const uint8_t *)msg, 
                                  sizeof(VirtIOArg));
    if (ret < sizeof(VirtIOArg)) {
            error("write error.\n");
            virtio_serial_throttle_port(port, true);
        }
    free(msg);
    debug("[+] WRITE BACK\n");
    return 0;
}

/* Callback function that's called when the guest opens/closes the port */
static void set_guest_connected(VirtIOSerialPort *port, int guest_connected)
{
    func();
    DeviceState *dev = DEVICE(port);
    // VirtIOSerial *vser = VIRTIO_CUDA(dev);
    debug("guest_connected=%d\n", guest_connected);

    if (dev->id) {
        qapi_event_send_vserport_change(dev->id, guest_connected,
                                        &error_abort);
    }
}

/* 
 * Enable/disable backend for virtio serial port
 * default enable is whether vm is running.
 * When vm is running, enable backend, otherwise disable backend.
 */
static void virtconsole_enable_backend(VirtIOSerialPort *port, bool enable)
{
    int port_id = port->id;
    int thread_id;
    func();
    debug("port id=%d, enable=%d\n", port->id, enable);

    if(!enable && global_deinitialized) {
        if (!total_port)
            return;
        thread_id = port_id; //%total_port;
        debug("Ending subprocess %d !\n",thread_id);
        /*
        * kill tid child process
        */
        int cmd = 0;
        write(pfd[thread_id][WRITE], &cmd, 4);
        return;
    }
    if(enable && !global_deinitialized)
        global_deinitialized = 1;

    return ;
}

static void guest_writable(VirtIOSerialPort *port)
{
    func();
    return;
}

static void spawn_subprocess_by_port(VirtIOSerialPort *port)
{
    pid_t cpid =0;
    int port_id = port->id;
    int tid = port_id; //%total_port;
    debug("Starting subprocess %d !\n", tid);

    // initialize message pipe
    pipe(pfd[tid]);
    pipe(cfd[tid]);
    cudaDevices[tid].device = (port_id-1)%total_device;
        /*reserved index 0 for thread*/
    cudaStreamBitmap[tid] = 0xffffffff;
    cudaEventBitmap[tid] = 0xffffffff;
    memset(cudaEvent[tid], 0, sizeof(cudaEvent_t)*CudaEventMaxNum);
    memset(cudaStream[tid], 0, sizeof(cudaStream_t)*CudaStreamMaxNum);
    cudaFunctionNum[tid]=0;


    my_signal(SIGCHLD, handler);

    cpid = fork();
    if(cpid == 0) {
        printf("child pid=%d\n", getpid());
        debug("child's parent ppid=%d\n", getppid());
        close(pfd[tid][WRITE]);
        close(cfd[tid][READ]);
        int cmd;
        int data_len=0;
        int i=0;
        // cuda variable
        int devID = 0;
        CUdevice cuDevice;
        CUcontext cuContext;
        CUmodule module;
        CUfunction kernel_func;
        cudaError_t err;
        // kernel
        dim3 gridDim;
        dim3 blockDim;
        size_t sharedMem;
        cudaStream_t stream;
        //
        uint64_t addr;

        while (1) {
            while(read(pfd[tid][READ], &cmd, 4) == 0);
            if(cmd==0)
                break;
            switch(cmd) {
                case VIRTIO_CUDA_REGISTERFATBINARY: {
                    err = cuInit(0);
                    if (err != cudaSuccess)
                    {
                        exit(EXIT_FAILURE);
                    }
                    // pick up device with zero ordinal (default, or devID)
                    err=cuDeviceGet(&cuDevice, devID);
                    if (err != cudaSuccess)
                    {
                        exit(EXIT_FAILURE);
                    }
                    // Create context
                    err =cuCtxCreate(&cuContext, 0, cuDevice);
                    if (err != cudaSuccess)
                    {
                        exit(EXIT_FAILURE);
                    }
                    write(cfd[tid][WRITE], &err, sizeof(cudaError_t));
                    break;
                }
                case VIRTIO_CUDA_UNREGISTERFATBINARY: {
                    cuCheck(cuCtxDestroy(cuContext), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_SETDEVICE: {
                    int dev = 0;
                    read(pfd[tid][READ], &dev, 4);
                    cudaCheck(cudaSetDevice(dev), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_SETDEVICEFLAGS: {
                    unsigned int flags;
                    read(pfd[tid][READ], &flags, sizeof(unsigned int));
                    cudaCheck(cudaSetDeviceFlags(flags), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_DEVICERESET: {
                    cudaCheck(cudaDeviceReset(), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_REGISTERFUNCTION: {
                    break;
                }
                case VIRTIO_CUDA_MALLOC: {
                    size_t size;
                    // void *devPtr;
                    CUdeviceptr ptr;
                    read(pfd[tid][READ], &size, sizeof(size_t));
                    debug("allocate 0x%lx\n", size);
                    // cudaCheck(cudaMalloc((void **)&devPtr, size), cfd[tid][1]);
                    cuCheck(cuMemAlloc(&ptr, size), cfd[tid][WRITE]);
                    write(cfd[tid][WRITE], &ptr, sizeof(CUdeviceptr));
                    break;
                }
                case VIRTIO_CUDA_FREE: {
                    read(pfd[tid][READ], &addr, sizeof(uint64_t));
                    debug("free addr %lx\n", addr);
                    cudaCheck( cudaFree((void*)addr) , cfd[tid][WRITE]);
                    // cuCheck(cuMemFree((CUdeviceptr)addr), cfd[tid][1]);
                    break;
                }
                case VIRTIO_CUDA_MEMSET: {
                    size_t count;
                    int value;
                    read(pfd[tid][READ], &count, sizeof(size_t));
                    read(pfd[tid][READ], &value, sizeof(int));
                    read(pfd[tid][READ], &addr, sizeof(uint64_t));
                    cudaCheck(cudaMemset((void *)addr, value, count), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_MEMCPY : {
                    int direction;
                    read(pfd[tid][READ], &direction, 4);
                    void *dst;
                    void *src;
                    if (direction == cudaMemcpyHostToDevice) {
                        // cuError( (err= cuMemcpyHtoD((CUdeviceptr)dst, src, size)));
                        read(pfd[tid][READ], &data_len, 4);
                        src = malloc(data_len);
                        read(pfd[tid][READ], src, data_len);
                        read(pfd[tid][READ], &dst, sizeof(void *));
                        cuCheck(cuMemcpyHtoD((CUdeviceptr)dst, src, data_len), cfd[tid][WRITE]);
                        free(src);
                    } else if (direction == cudaMemcpyDeviceToHost) {
                        // cuError( (err=cuMemcpyDtoH(dst, (CUdeviceptr)src, size)) );
                        read(pfd[tid][READ], &data_len, 4);
                        dst = malloc(data_len);
                        read(pfd[tid][READ], &src, sizeof(void *));
                        cuCheck(cuMemcpyDtoH(dst, (CUdeviceptr)src, data_len), cfd[tid][WRITE]);
                        // debug("float point [0] = %f\n", *(float*)dst);
                        write(cfd[tid][WRITE], dst, data_len);
                        free(dst);
                    } else if (direction == cudaMemcpyDeviceToDevice) {
                        read(pfd[tid][READ], &src, sizeof(void *));
                        read(pfd[tid][READ], &dst, sizeof(void *));
                        read(pfd[tid][READ], &data_len, 4);
                        cuCheck(cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, data_len), cfd[tid][WRITE]);
                    }
                    break;
                }
                case VIRTIO_CUDA_MEMCPY_ASYNC : {
                    int direction;
                    read(pfd[tid][READ], &direction, 4);
                    CUstream stream;
                    size_t bytecount;
                    if (direction == cudaMemcpyHostToDevice) {
                        // cuError( (err= cuMemcpyHtoDAsync((CUdeviceptr)dst, (void *)src, size, 
                                         // stream)));
                        CUdeviceptr dst;
                        void *src;
                        read(pfd[tid][READ], &stream, sizeof(CUstream));
                        read(pfd[tid][READ], &bytecount, 4);
                        src = malloc(bytecount);
                        read(pfd[tid][READ], src, bytecount);
                        read(pfd[tid][READ], &dst, sizeof(CUdeviceptr));
                        cuCheck(cuMemcpyHtoDAsync(dst, src, bytecount, stream), cfd[tid][WRITE]);
                        free(src);
                    } else if (direction == cudaMemcpyDeviceToHost) {
                        // cuError( (err=cuMemcpyDtoHAsync((void *)dst, (CUdeviceptr)src, 
                        //                                 size, stream)) );
                        CUdeviceptr src;
                        void *dst=NULL;
                        read(pfd[tid][READ], &stream, sizeof(CUstream));
                        read(pfd[tid][READ], &bytecount, 4);
                        read(pfd[tid][READ], &src, sizeof(CUdeviceptr));
                        read(pfd[tid][READ], &dst, sizeof(void *));
                        debug("bytecount is %lx\n", bytecount);
                        debug("src is %llx\n", src);
                        debug("dst is %p\n", dst);
                        if (dst == NULL)
                            dst = malloc(bytecount);
                        cuCheck(cuMemcpyDtoHAsync(dst, src, bytecount, stream), cfd[tid][WRITE]);
                        if (dst == NULL) {
                            write(cfd[tid][WRITE], dst, bytecount);
                            free(dst);
                        }
                    } else if (direction == cudaMemcpyDeviceToDevice) {
                        // cuError( (err=cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, 
                        //                                 size, stream)) );
                        CUdeviceptr dst;
                        CUdeviceptr src;
                        read(pfd[tid][READ], &stream, sizeof(CUstream));
                        read(pfd[tid][READ], &src, sizeof(CUdeviceptr));
                        read(pfd[tid][READ], &dst, sizeof(CUdeviceptr));
                        read(pfd[tid][READ], &bytecount, 4);
                        cuCheck(cuMemcpyDtoDAsync(dst, src, bytecount, stream), cfd[tid][WRITE]);
                    }
                    break;
                }
                case VIRTIO_CUDA_LAUNCH: {
                    read(pfd[tid][READ], &kernel_func, sizeof(CUfunction));
                    if (kernel_func==NULL) {
                        // register function
                        read(pfd[tid][READ], &data_len, 4);
                        void *fat_bin = malloc(data_len);
                        read(pfd[tid][READ], fat_bin, data_len);
                        cuCheck( cuModuleLoadData(&module, fat_bin), cfd[tid][WRITE]);

                        read(pfd[tid][READ], &data_len, 4);
                        void * func_name = malloc(data_len);
                        read(pfd[tid][READ], func_name, data_len);
                        cuCheck( cuModuleGetFunction(&kernel_func,
                                                    module,
                                                    func_name) , cfd[tid][WRITE]);
                        write(cfd[tid][WRITE], &kernel_func, sizeof(CUfunction));
                    }
                    // read(pfd[tid][READ], &kernel_func, sizeof(CUfunction));
                    read(pfd[tid][READ], &gridDim, sizeof(dim3));
                    read(pfd[tid][READ], &blockDim, sizeof(dim3));
                    read(pfd[tid][READ], &sharedMem, sizeof(size_t));
                    read(pfd[tid][READ], &stream, sizeof(cudaStream_t));
                    read(pfd[tid][READ], &data_len, 4);
                    uint8_t *para=malloc(data_len);
                    read(pfd[tid][READ], para, data_len);

                    uint32_t para_num = *((uint32_t*)para);
                    debug(" para_num = %u\n", para_num);
                    void **para_buf = malloc(para_num * sizeof(void*));
                    int para_idx = sizeof(uint32_t);
                    for(i=0; i<para_num; i++) {
                        para_buf[i] = &para[para_idx + sizeof(uint32_t)];
                        debug("arg %d = 0x%llx , size=%u byte\n", i, 
                              *(unsigned long long*)para_buf[i], 
                              *(unsigned int*)(&para[para_idx]));
                        para_idx += *(uint32_t*)(&para[para_idx]) + sizeof(uint32_t);
                    }

                    cuCheck(cuLaunchKernel( kernel_func,
                                            gridDim.x, gridDim.y, gridDim.z,
                                            blockDim.x, blockDim.y, blockDim.z,
                                            sharedMem,
                                            stream,
                                            para_buf,
                                            NULL) , cfd[tid][WRITE]);
                    free(para);
                    free(para_buf);
                    break;
                }
                case VIRTIO_CUDA_MEMGETINFO: {
                    size_t freeMem, totalMem;
                    cudaCheck(cudaMemGetInfo(&freeMem, &totalMem), cfd[tid][WRITE]);
                    write(cfd[tid][WRITE], &freeMem, sizeof(size_t));
                    write(cfd[tid][WRITE], &totalMem, sizeof(size_t));
                    break;
                }
                case VIRTIO_CUDA_DEVICESYNCHRONIZE : {
                    cudaCheck(cudaDeviceSynchronize(), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_GETLASTERROR: {
                    cudaCheck(cudaGetLastError(), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_STREAMCREATE: {
                    cudaStream_t stream;
                    cudaCheck(cudaStreamCreate(&stream), cfd[tid][WRITE] );
                    write(cfd[tid][WRITE], &stream, sizeof(cudaStream_t));
                    break;
                }
                case VIRTIO_CUDA_STREAMDESTROY: {
                    cudaStream_t stream;
                    read(pfd[tid][READ], &stream, sizeof(cudaStream_t));
                    cudaCheck(cudaStreamDestroy(stream), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_EVENTCREATE: {
                    cudaEvent_t event;
                    cudaCheck(cudaEventCreate(&event), cfd[tid][WRITE]);
                    write(cfd[tid][WRITE], &event, sizeof(cudaEvent_t));
                    break;
                }
                case VIRTIO_CUDA_EVENTCREATEWITHFLAGS: {
                    cudaEvent_t event;
                    unsigned int flags;
                    read(pfd[tid][READ], &flags, sizeof(unsigned int));
                    cudaCheck(cudaEventCreateWithFlags(&event, flags), cfd[tid][WRITE]);
                    write(cfd[tid][WRITE], &event, sizeof(cudaEvent_t));
                    break;
                }
                case VIRTIO_CUDA_EVENTDESTROY: {
                    cudaEvent_t event;
                    read(pfd[tid][READ], &event, sizeof(cudaEvent_t));
                    cudaCheck(cudaEventDestroy(event), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_EVENTRECORD: {
                    cudaEvent_t event;
                    cudaStream_t stream;
                    read(pfd[tid][READ], &event, sizeof(cudaEvent_t));
                    read(pfd[tid][READ], &stream, sizeof(cudaStream_t));
                    cudaCheck(cudaEventRecord(event, stream), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_EVENTSYNCHRONIZE: {
                    cudaEvent_t event;
                    read(pfd[tid][READ], &event, sizeof(cudaEvent_t));
                    cudaCheck(cudaEventSynchronize(event), cfd[tid][WRITE]);
                    break;
                }
                case VIRTIO_CUDA_EVENTELAPSEDTIME: {
                    cudaEvent_t start, end;
                    float time;
                    read(pfd[tid][READ], &start, sizeof(cudaEvent_t));
                    read(pfd[tid][READ], &end, sizeof(cudaEvent_t));
                    cudaCheck(cudaEventElapsedTime(&time, start, end), cfd[tid][WRITE]);
                    write(cfd[tid][WRITE], &time, sizeof(float));
                    break;
                }
                case VIRTIO_CUDA_HOSTREGISTER: {
                    unsigned int flags;
                    size_t size;
                    char file_path[64];
                    read(pfd[tid][READ], &flags, sizeof(unsigned int));
                    read(pfd[tid][READ], &size, sizeof(size_t));
                    read(pfd[tid][READ], file_path, 64);
                    debug("size=0x%lx\n", size);
                    debug("file_path=%s\n", file_path);
                    void *addr = get_shm(size, file_path);
                    if (addr == NULL) {
                        error("Failed to get share memory in subprocess.\n");
                    }
                    cudaCheck(cudaHostRegister(addr, size, flags), cfd[tid][WRITE]);
                    write(cfd[tid][WRITE], &addr, sizeof(void *));
                    debug("host register addr=%p\n", addr);
                    break;
                }
                case VIRTIO_CUDA_HOSTUNREGISTER: {
                    void *addr;
                    read(pfd[tid][READ], &addr, sizeof(void *));
                    cudaCheck(cudaHostUnregister(addr), cfd[tid][WRITE]);
                    // munmap
                    break;
                }
                default:
                    error("No such cmd %d\n", cmd);
            }
        }
        debug("child process finish\n");
        exit(EXIT_SUCCESS);
    }
    close(pfd[tid][READ]);
    close(cfd[tid][WRITE]);
}

/* 
* Guest is now ready to accept data (virtqueues set up). 
* When the guest has asked us for this information it means
* the guest is all setup and has its virtqueues
* initialised. If some app is interested in knowing about
* this event, let it know.
* ESPECIALLY, when front driver insmod the driver.
*/
static void virtconsole_guest_ready(VirtIOSerialPort *port)
{
    VirtIOSerial *vser = port->vser;
    func();
    debug("port %d is ready.\n", port->id);
    qemu_mutex_lock(&total_port_mutex);
    total_port = get_active_ports_nr(vser);
    qemu_mutex_unlock(&total_port_mutex);
    if( total_port > WORKER_THREADS-1) {
        error("Too much ports, over %d\n", WORKER_THREADS);
        return;
    }
}

static void init_device_once(VirtIOSerial *vser)
{
    qemu_mutex_lock(&vser->init_mutex);
    if (global_initialized==1) {
        debug("global_initialized already!\n");
        qemu_mutex_unlock(&vser->init_mutex);
        return;
    }
    global_initialized = 1;
    global_deinitialized = 0;
    qemu_mutex_unlock(&vser->init_mutex);
    // debug("vser->gpus[i].name=%s\n", vser->gpus[vser->gcount-1]->prop.name);
    total_port = 0;
    qemu_mutex_init(&total_port_mutex);
    // cuError( cuInit(0));
    if(!vser->gcount) {
        error("init error, can not get gpu count \n");
        return;
    }
    else {
        debug("vser->gcount=%d\n", vser->gcount);
        total_device = vser->gcount;
    }
}

static void deinit_device_once(VirtIOSerial *vser)
{
    qemu_mutex_lock(&vser->deinit_mutex);
    if(global_deinitialized==0) {
        qemu_mutex_unlock(&vser->deinit_mutex);
        return;
    }
    global_deinitialized=0;
    qemu_mutex_unlock(&vser->deinit_mutex);
    func();

    qemu_mutex_destroy(&total_port_mutex);
}

static void init_port(VirtIOSerialPort *port)
{
    int tid = port->id;
    INIT_LIST_HEAD(&cudaDevices[tid].vol);
    INIT_LIST_HEAD(&cudaDevices[tid].host_vol);
}

static void deinit_port(VirtIOSerialPort *port)
{
    return;
}
/*
 * The per-port (or per-app) realize function that's called when a
 * new device is found on the bus.
*/
static void virtconsole_realize(DeviceState *dev, Error **errp)
{
    func();
    VirtIOSerialPort *port = VIRTIO_SERIAL_PORT(dev);
    VirtIOSerial *vser = port->vser;
    VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_GET_CLASS(dev);
    debug("port->id = %d\n", port->id );
    if (port->id == 0 && !k->is_console) {
        error_setg(errp, "Port number 0 on virtio-serial devices reserved "
                   "for virtconsole devices for backward compatibility.");
        return;
    }

    virtio_serial_open(port);

    /* init GPU device
    */
    init_device_once(vser);
    init_port(port);
    spawn_subprocess_by_port(port);

}

/*
* Per-port unrealize function that's called when a port gets
* hot-unplugged or removed.
*/
static void virtconsole_unrealize(DeviceState *dev, Error **errp)
{
    func();
    VirtConsole *vcon = VIRTIO_CONSOLE(dev);
    VirtIOSerialPort *port = VIRTIO_SERIAL_PORT(dev);
    VirtIOSerial *vser = port->vser;

    deinit_device_once(vser);
    deinit_port(port);
    if (vcon->watch) {
        g_source_remove(vcon->watch);
    }

}

static Property virtserialport_properties[] = {
    DEFINE_PROP_CHR("chardev", VirtConsole, chr),
    DEFINE_PROP_STRING("privatekeypath", VirtConsole, privatekeypath),
    DEFINE_PROP_STRING("hmac_path", VirtConsole, hmac_path), 
    DEFINE_PROP_END_OF_LIST(),
};

static void virtserialport_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_CLASS(klass);

    k->realize = virtconsole_realize;
    k->unrealize = virtconsole_unrealize;
    k->have_data = flush_buf;
    k->set_guest_connected = set_guest_connected;
    k->enable_backend = virtconsole_enable_backend;
    k->guest_ready = virtconsole_guest_ready;
    k->guest_writable = guest_writable;
    dc->props = virtserialport_properties;
}

static const TypeInfo virtserialport_info = {
    .name          = TYPE_VIRTIO_CONSOLE_SERIAL_PORT,
    .parent        = TYPE_VIRTIO_SERIAL_PORT,
    .instance_size = sizeof(VirtConsole),
    .class_init    = virtserialport_class_init,
};

static void virtconsole_register_types(void)
{
    type_register_static(&virtserialport_info);
}

type_init(virtconsole_register_types)
