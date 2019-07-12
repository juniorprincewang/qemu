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
#include "message_queue.h"
#include "list.h"

#include <openssl/hmac.h> // hmac EVP_MAX_MD_SIZE
/*Encodes Base64 */
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <stdint.h>

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

#define cudaError(err) __cudaErrorCheck(err, __LINE__)

#define TYPE_VIRTIO_CONSOLE_SERIAL_PORT "virtcudaport"
#define VIRTIO_CONSOLE(obj) \
    OBJECT_CHECK(VirtConsole, (obj), TYPE_VIRTIO_CONSOLE_SERIAL_PORT)

#ifndef WORKER_THREADS
#define WORKER_THREADS 32
#endif

typedef struct VirtConsole {
    VirtIOSerialPort parent_obj;
    char *privatekeypath;    
    char *hmac_path;
    CharBackend chr;
    guint watch;
} VirtConsole;

#define CudaFunctionMaxNum 1024
#define CudaEventMaxNum 32
#define CudaStreamMaxNum 32
#define CudaFunctionName 128

typedef struct VirtualObjectList {
    uint64_t addr;
    uint64_t v_addr;
    int size;
    struct list_head list;
} VOL;

typedef struct CudaDev {
    CUdevice device;
    CUcontext context;
    unsigned int kernel_func_id[CudaFunctionMaxNum];
    CUfunction kernel_func[CudaFunctionMaxNum];
    CUmodule module;
    struct list_head vol;
    pthread_spinlock_t vol_lock;
}CudaDev;

typedef struct KernelInfo {
    void *fatBin;
    char func_name[CudaFunctionName];
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
CudaDev *cudaDevices;
CudaDev zeroedDevice;
int cudaFunctionNum[WORKER_THREADS];
CUdevice worker_cur_device[WORKER_THREADS];
KernelInfo devicesKernels[WORKER_THREADS][CudaFunctionMaxNum];

cudaEvent_t cudaEvent[CudaEventMaxNum];
uint32_t cudaEventNum;
cudaStream_t cudaStream[CudaStreamMaxNum];
uint32_t cudaStreamNum;
int version;


static int global_initialized = 0;
static int global_deinitialized = 0;

// cudaError_t global_err; //
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

typedef struct Workload
{
    int device_id;
    VirtIOSerialPort *port;
    cudaStream_t stream;
} Workload;

Workload works[WORKER_THREADS];
static struct message_queue worker_queue[WORKER_THREADS];
static QemuThread worker_threads[WORKER_THREADS];

static void *worker_processor(void *arg);
static VOL *find_vol_by_vaddr(uint64_t vaddr, CudaDev *dev);

// static void load_module_kernel(int devID, void *fatBin, 
//                 char *funcName, unsigned int funcID, int funcIndex)
// {
//     func();
//     debug("Loading module... fatBin=%16p, name=%s, funcID=%d\n", 
//             fatBin, funcName, funcID);
//     cuError( cuModuleLoadData(&cudaDevices[devID].module, fatBin) );
//     cuError( cuModuleGetFunction(&cudaDevices[devID].kernel_func[funcIndex], 
//         cudaDevices[devID].module, funcName) );
//     cudaDevices[devID].kernel_func_id[funcIndex] = funcID;
// }

static cudaError_t initialize_device(unsigned int tid)
{
    int dev = worker_cur_device[tid];
    func();
    // device reset
    if ( !memcmp(&zeroedDevice, &cudaDevices[dev], sizeof(CudaDev)) ) {
        cuError( cuDeviceGet(&cudaDevices[dev].device, dev) );
        cuError( cuCtxCreate(&cudaDevices[dev].context, 0, cudaDevices[dev].device) );
        debug("Device was reset therefore no context\n");
    } else {
        cuError( cuCtxSetCurrent(cudaDevices[dev].context) );
        debug("Cuda device %d\n", cudaDevices[dev].device);
    }
    return cudaSuccess;
}

static unsigned int get_current_id(unsigned int tid)
{
    return tid%total_device;
}

/*
static void *memdup(const void *src, size_t n)
{
    void *dst;
    dst = malloc(n);
    if(dst == NULL)
        return NULL;
    return memcpy(dst, src, n);
}
*/

static void init_device(VirtIOSerial *vser)
{
    unsigned int i;
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
    cuError( cuInit(0));
    if(!vser->gcount)
        cuError( cuDeviceGetCount(&total_device) );
    else {
        debug("vser->gcount=%d\n", vser->gcount);
        total_device = vser->gcount;
    }
    cuError( cuDriverGetVersion(&version) );
    cudaDevices = (CudaDev *)malloc(total_device * sizeof(CudaDev));
    memset(&zeroedDevice, 0, sizeof(CudaDev));
    i = total_device;
    while(i-- != 0) {
        debug("[+] Create context for device %d\n", i);
        memset(&cudaDevices[i], 0, sizeof(CudaDev));
        cuError( cuDeviceGet(&cudaDevices[i].device, i) );
        cuError( cuCtxCreate(&cudaDevices[i].context, 0, cudaDevices[i].device) );
        memset(&cudaDevices[i].kernel_func, 0, sizeof(CUfunction) * CudaFunctionMaxNum);
        INIT_LIST_HEAD(&cudaDevices[i].vol);
        pthread_spin_init(&cudaDevices[i].vol_lock, PTHREAD_PROCESS_PRIVATE);
    }
    cudaEventNum = 0;
    cudaStreamNum = 0;
    for(i =0; i<CudaEventMaxNum; i++)
        memset(&cudaEvent[i], 0, sizeof(cudaEvent_t));
    memset(cudaStream, 0, sizeof(cudaStream_t)*CudaStreamMaxNum);
}

static void deinit_device(VirtIOSerial *vser)
{
    int i=0;
    qemu_mutex_lock(&vser->deinit_mutex);
    if(global_deinitialized==0) {
        qemu_mutex_unlock(&vser->deinit_mutex);
        return;
    }
    global_deinitialized=0;
    qemu_mutex_unlock(&vser->deinit_mutex);
    func();
    qemu_mutex_destroy(&total_port_mutex);

    debug("free cudaDevices stuff\n");
    i = total_device;
    while(i-- != 0) {
        pthread_spin_destroy(&cudaDevices[i].vol_lock);
        // list_del(&cudaDevices[i].vol);
        if ( memcmp(&zeroedDevice, &cudaDevices[i], sizeof(CudaDev)) != 0) {
           cuError( cuCtxDestroy(cudaDevices[i].context) );
        }
    }
    free(cudaDevices);
}

static void cuda_register_fatbinary(VirtIOArg *arg)
{
    void *fat_bin;
    hwaddr gpa;
    uint32_t src_size = arg->srcSize;
    func();
    // check out hmac
    gpa = (hwaddr)(arg->src);
    hwaddr src_len = (hwaddr)src_size;
    fat_bin = cpu_physical_memory_map(gpa, &src_len, 0);
    if (!fat_bin || src_len != src_size) {
        error("Failed to map MMIO memory for"
                          " gpa 0x%lx element size %u\n",
                            gpa, src_size);
        return ;
    }
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
    
    arg->cmd = cudaSuccess;
}

static void cuda_unregister_fatinary(VirtIOArg *arg, int tid)
{
    int f_idx;
    func();
    for(f_idx=0; f_idx<cudaFunctionNum[tid]; f_idx++) {
        free(devicesKernels[tid][f_idx].fatBin);
        devicesKernels[tid][f_idx].fatBin = NULL;
        memset(devicesKernels[tid][f_idx].func_name, 0, 
            sizeof(devicesKernels[tid][f_idx].func_name));
    }
}

static void cuda_register_function(VirtIOArg *arg, int tid)
{
    hwaddr fat_bin_gpa;
    hwaddr func_name_gpa;
    uint32_t func_id;
    uint32_t fat_size, name_size;
    int nr_func = cudaFunctionNum[tid];
    int dev_id;
    func();
    fat_bin_gpa = (hwaddr)(arg->src);
    func_name_gpa = (hwaddr)(arg->dst);
    func_id = arg->flag;
    fat_size = arg->srcSize;
    name_size = arg->dstSize;
    // initialize the KernelInfo
    devicesKernels[tid][nr_func].fatBin = malloc(fat_size);

    cpu_physical_memory_read(fat_bin_gpa, \
        devicesKernels[tid][nr_func].fatBin, fat_size);
    cpu_physical_memory_read(func_name_gpa, \
        devicesKernels[tid][nr_func].func_name, name_size);

    devicesKernels[tid][nr_func].func_id = func_id;
    debug("Loading module... fatBin = %16p, name='%s',"
            " func_id=%d, nr_func = %d\n", 
        devicesKernels[tid][nr_func].fatBin, \
        (char*)devicesKernels[tid][nr_func].func_name, \
        func_id, \
        nr_func);
    dev_id = worker_cur_device[tid];
    cuError( cuModuleLoadData(&cudaDevices[dev_id].module, 
                            devicesKernels[tid][nr_func].fatBin) );
    cuError( cuModuleGetFunction(&cudaDevices[dev_id].kernel_func[nr_func], \
                                    cudaDevices[dev_id].module, \
                            (char*)devicesKernels[tid][nr_func].func_name) );
    cudaDevices[dev_id].kernel_func_id[nr_func] = func_id;
    cudaFunctionNum[tid]++;
    arg->cmd = cudaSuccess;
}


static void cuda_setup_argument(VirtIOArg *arg)
{
    func();
}

static void cuda_launch(VirtIOArg *arg, int tid)
{
    cudaError_t err;
    int i=0;
    uint32_t func_id, para_num, para_idx, func_idx;
    int dev_id;
    void **para_buf;
    uint32_t para_size, conf_size;
    hwaddr hw_para_size, hw_conf_size;
    VOL *vol;
    // hwaddr gpa_para = (hwaddr)(arg->src);
    // hwaddr gpa_conf = (hwaddr)(arg->dst);
    func();
    debug("thread id = %u\n", tid);
    para_size = arg->srcSize;
    conf_size = arg->dstSize;
    
    hw_para_size = (hwaddr)para_size;
    hw_conf_size = (hwaddr)conf_size;

    char *para = (char *)cpu_physical_memory_map((hwaddr)(arg->src), &hw_para_size, 0);
    if (!para || hw_para_size != para_size) {
        error("Failed to map MMIO memory for"
                          " gpa 0x%lx element size %u\n",
                            arg->src, para_size);
        return ;
    }
    KernelConf_t *conf = (KernelConf_t*)cpu_physical_memory_map(\
        (hwaddr)(arg->dst), &hw_conf_size, 0);
    if (!conf || hw_conf_size != conf_size) {
        error("Failed to map MMIO memory for"
                          " gpa 0x%lx element size %u\n",
                            arg->dst, conf_size);
        return ;
    }
    
    dev_id = worker_cur_device[tid];
    /*
    char *para = (char *)malloc(para_size);
    cpu_physical_memory_read(gpa_para, para, para_size);
    KernelConf_t *conf = (KernelConf_t *)malloc(conf_size);
    cpu_physical_memory_read(gpa_conf, (void *)conf, conf_size);
    */
    func_id = (uint32_t)(arg->flag);
    debug(" func_id = %u\n", func_id);
    para_num = *((uint32_t*)para);
    debug(" para_num = %u\n", para_num);
    
    para_buf = malloc(para_num * sizeof(void*));
    para_idx = sizeof(uint32_t);
    for(i=0; i<para_num; i++) {
        para_buf[i] = &para[para_idx + sizeof(uint32_t)];
        vol = find_vol_by_vaddr(*(uint64_t*)para_buf[i], &cudaDevices[dev_id]);
        if(vol!= NULL) {
            debug("Found %lx\n", vol->addr);
            para_buf[i] = &vol->addr;
        }
        debug("arg %d = 0x%llx , size=%u byte\n", i, 
            *(unsigned long long*)para_buf[i], *(unsigned int*)(&para[para_idx]));
        para_idx += *(uint32_t*)(&para[para_idx]) + sizeof(uint32_t);
    }
    int found = 0;
    for(func_idx=0; func_idx < cudaFunctionNum[tid]; func_idx++) {
        if( cudaDevices[dev_id].kernel_func_id[func_idx] 
                    == func_id) {
            found = 1;
            break;
        }
    }
    if(!found){
        error("Failed to find func id.\n");
        free(para_buf);
        return;
    }
    debug("Found func_idx = %d.\n", func_idx);
    debug("Found function  = %lu.\n", 
            (uint64_t)(cudaDevices[dev_id].kernel_func[func_idx]));

    debug("gridDim=%u %u %u\n", conf->gridDim.x, 
        conf->gridDim.y, conf->gridDim.z);
    debug("blockDim=%u %u %u\n", conf->blockDim.x, 
        conf->blockDim.y, conf->blockDim.z);
    debug("sharedMem=%ld\n", conf->sharedMem);
    debug("stream=%lu\n", (uint64_t)(conf->stream));
    /*
    cudaError( (err= cudaConfigureCall(conf->gridDim, 
        conf->blockDim, conf->sharedMem, conf->stream)));
    */
    cuError( (err = cuLaunchKernel(
        cudaDevices[dev_id].kernel_func[func_idx], \
        conf->gridDim.x, conf->gridDim.y, conf->gridDim.z,\
        conf->blockDim.x, conf->blockDim.y, conf->blockDim.z,\
        conf->sharedMem, conf->stream, para_buf, NULL) ) );
    arg->cmd = err;
    free(para_buf);
    para_buf = NULL;
}

static VOL *find_vol_by_vaddr(uint64_t vaddr, CudaDev *dev)
{
    VOL *vol;
    pthread_spin_lock(&dev->vol_lock);
    list_for_each_entry(vol, &dev->vol, list) {
        if(vol->v_addr == vaddr)
            goto out;
    }
    vol = NULL;
out:
    pthread_spin_unlock(&dev->vol_lock);
    return vol;
}

static void cuda_memcpy(VirtIOArg *arg, int tid)
{
    cudaError_t err;
    uint32_t size;
    void *src, *dst;
    VOL *vol;
    int dev_id = worker_cur_device[tid];
    func();
    // in case cudaDeviceReset was the previous call
    initialize_device(tid);
    debug("src=0x%lx, srcSize=%d, dst=0x%lx, dstSize=%d, kind=%lu\n", \
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag);
    size = arg->srcSize;
    if (arg->flag == cudaMemcpyHostToDevice) {
        hwaddr src_len = (hwaddr)size;
        src = cpu_physical_memory_map(arg->src, &src_len, 0);
        if (!src || src_len != size) {
            error("Failed to map MMIO memory for"
                              " gpa 0x%lx element size %u\n",
                                arg->src, size);
            return ;
        }
        vol = find_vol_by_vaddr(arg->dst, &cudaDevices[dev_id]);
        if(vol == NULL) {
            error("Failed to find virtual address %p in vol\n", (void *)arg->dst);
            return;
        }
        if(vol->size != size) {
            error("Failed to match size in vol\n");
            return;
        }
        dst = (void *)vol->addr;
        cuError( (err= cuMemcpyHtoD((CUdeviceptr)dst, (void *)src, size)));
    } else if (arg->flag == cudaMemcpyDeviceToHost) {
        vol = find_vol_by_vaddr(arg->src, &cudaDevices[dev_id]);
        if(vol == NULL) {
            error("Failed to find virtual address %p in vol\n", (void *)arg->src);
            return;
        }
        if(vol->size != size) {
            error("Failed to match size in vol\n");
            return;
        }
        src = (void*)vol->addr;

        hwaddr dst_len = (hwaddr)size;
        // gpa => hva
        dst = cpu_physical_memory_map(arg->dst, &dst_len, 0);
        if (!dst || dst_len != size) {
            error("Failed to map MMIO memory for"
                              " gpa 0x%lx element size %u\n",
                                arg->dst, size);
            return ;
        }
        //testOver4K(buf, size);
        cuError( (err=cuMemcpyDtoH((void *)dst, (CUdeviceptr)src, size)) );
        // err = 0;
    } else if (arg->flag == cudaMemcpyDeviceToDevice) {
        vol = find_vol_by_vaddr(arg->src, &cudaDevices[dev_id]);
        if(vol == NULL) {
            error("Failed to find virtual address %p in vol\n", (void *)arg->src);
            return;
        }
        if(vol->size != size) {
            error("Failed to match size in vol\n");
            return;
        }
        src = (void*)vol->addr;
        vol = find_vol_by_vaddr(arg->dst, &cudaDevices[dev_id]);
        if(vol == NULL) {
            error("Failed to find virtual address %p in vol\n", (void *)arg->dst);
            return;
        }
        if(vol->size != size) {
            error("Failed to match size in vol\n");
            return;
        }
        dst = (void*)vol->addr;
        cuError( (err=cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, size)) );
    }
    arg->cmd = err;
    debug(" return value=%d\n", err);
}

static void cuda_memcpy_async(VirtIOArg *arg, int tid)
{
    cudaError_t err;
    uint32_t size;
    void *src, *dst;
    uint64_t idx = arg->param;
    // int dev_id = worker_cur_device[tid];
    func();
    // in case cudaDeviceReset was the previous call
    initialize_device(tid);
    debug("src=0x%lx, srcSize=%d, dst=9x%lx, dstSize=%d, kind=%lu, "
        "stream idx = %lu \n", \
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, arg->param);
    size = arg->srcSize;
    if (arg->flag == cudaMemcpyHostToDevice) {
        hwaddr src_len = (hwaddr)size;
        src = cpu_physical_memory_map(arg->src, &src_len, 0);
        if (!src || src_len != size) {
            error("Failed to map MMIO memory for"
                              " gpa 0x%lx element size %u\n",
                                arg->src, size);
            return ;
        }
        dst = (void *)arg->dst;
        cuError( (err= cuMemcpyHtoDAsync((CUdeviceptr)dst, \
                                        (void *)src, size, cudaStream[idx])));
    } else if (arg->flag == cudaMemcpyDeviceToHost) {
        src = (void *)(arg->src);
        hwaddr dst_len = (hwaddr)size;
        // gpa => hva
        dst = cpu_physical_memory_map(arg->dst, &dst_len, 0);
        if (!dst || dst_len != size) {
            error("Failed to map MMIO memory for"
                              " gpa 0x%lx element size %u\n",
                                arg->dst, size);
            return ;
        }
        //testOver4K(buf, size);
        cuError( (err=cuMemcpyDtoHAsync((void *)dst, \
                                (CUdeviceptr)src, size, cudaStream[idx])) );
        // err = 0;
    } else if (arg->flag == cudaMemcpyDeviceToDevice) {
        src = (void *)(arg->src);
        dst = (void *)(arg->dst);
        cuError( (err=cuMemcpyDtoDAsync((CUdeviceptr)dst, \
                                (CUdeviceptr)src, size, cudaStream[idx])) );
    }
    arg->cmd = err;
    debug(" return value=%d\n", err);
}

static void cuda_memset(VirtIOArg *arg, int tid)
{
    cudaError_t err;
    size_t count;
    int value;
    void *dst;
    func();
    // in case cudaDeviceReset was the previous call
    initialize_device(tid);
    count = (size_t)(arg->dstSize);
    value = (int)(arg->param);
    dst = (void *)(arg->dst);
    debug("dst=0x%lx, value=%d, count=%lu\n", \
        arg->dst, value, count);
    cuError( (err= cudaMemset(dst, value, count)));
    arg->cmd = err;
    debug(" return value=%d\n", err);
}

static void cuda_malloc(VirtIOArg *arg, int tid)
{
    cudaError_t err;
    void *devPtr;
    uint32_t size;
    VOL *vol;
    int dev_id = worker_cur_device[tid];
    func();

    // in case cudaReset was the previous call
    initialize_device(tid);
    size = arg->srcSize;
    cudaError( (err= cudaMalloc(&devPtr, size)));
    vol = (VOL *)malloc(sizeof(VOL));
    vol->addr = (uint64_t)devPtr;
    vol->v_addr = (uint64_t)(devPtr + 0x1000);
    arg->dst =  (uint64_t)(vol->v_addr);
    vol->size = size;
    pthread_spin_lock(&cudaDevices[dev_id].vol_lock);
    list_add_tail(&vol->list, &cudaDevices[dev_id].vol);
    pthread_spin_unlock(&cudaDevices[dev_id].vol_lock);
    arg->cmd = err;
    debug(" actual devPtr=0x%lx, virtual ptr=0x%lx, size=%d, return value=%d\n", 
        (uint64_t)devPtr, arg->dst, size, err);
}

static void cuda_free(VirtIOArg *arg, int tid)
{
    cudaError_t err;
    uint64_t src;
    VOL *vol, *vol2;
    int dev_id = worker_cur_device[tid];
    func();
    // in case of cudaReset
    initialize_device(tid);
    src = (arg->src);
    debug(" ptr = 0x%lx\n", arg->src);
    list_for_each_entry_safe(vol, vol2, &cudaDevices[dev_id].vol, list) {
        if (vol->v_addr == src) {
            cudaError( (err= cudaFree((void*)(vol->addr))) );
            pthread_spin_lock(&cudaDevices[dev_id].vol_lock);
            list_del(&vol->list);
            pthread_spin_unlock(&cudaDevices[dev_id].vol_lock);
            arg->cmd = err;
            return;
        }
    }
    arg->cmd = cudaErrorInvalidValue;
    error("Failed to find ptr!\n");
}

static void cuda_get_device(VirtIOArg *arg, int tid)
{
    int dev = 0;
    hwaddr gpa = (hwaddr)(arg->dst);
    func();
    initialize_device(tid);
    arg->cmd = cudaSuccess;
    dev = (int)(cudaDevices[worker_cur_device[tid]].device);
    cpu_physical_memory_write(gpa, &dev, sizeof(int));
}

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
    debug("Device %d : \"%s\" with compute %d.%d capability.\n", devID, prop.name, prop.major, prop.minor);
    arg->cmd = err;
    cpu_physical_memory_write(gpa, &prop, arg->dstSize);
}

static void cuda_set_device(VirtIOArg *arg, int tid)
{
    func();
    int dev_id = (int)(arg->flag);
    if (dev_id < 0 || dev_id > total_device-1) {
        error("setting error device = %d\n", dev_id);
        arg->cmd = cudaErrorInvalidDevice;
        return ;
    }
    worker_cur_device[tid] = dev_id;
    initialize_device(tid);
    arg->cmd = cudaSuccess;
    debug("set devices=%d\n", (int)(arg->flag));
}

static void cuda_set_device_flags(VirtIOArg *arg)
{

}

static void cuda_get_device_count(VirtIOArg *arg)
{
    unsigned int id = get_current_id( (unsigned int)arg->tid );
    func();
    initialize_device(id);
    debug("Device count=%d.\n", total_device);
    arg->cmd = (int32_t)cudaSuccess;
    arg->flag = (uint64_t)total_device;
}

static void cuda_device_reset(VirtIOArg *arg, int tid)
{
    func();
    // should get rid of events for current devices
    cuCtxDestroy(cudaDevices[worker_cur_device[tid]].context ) ;
    memset( &cudaDevices[worker_cur_device[tid]], 0, sizeof(CudaDev) );
    debug("reset devices\n");
}

static void cuda_stream_create(VirtIOArg *arg)
{
    cudaError_t err = 0;
    func();

    cudaError( (err = cudaStreamCreate(&cudaStream[cudaStreamNum]) ));
    arg->flag = (uint64_t)cudaStreamNum;
    debug("create stream %lu, idx is %u\n", (uint64_t)cudaStream[cudaStreamNum], cudaStreamNum);
    cudaStreamNum = (cudaStreamNum+1)%CudaStreamMaxNum;
    arg->cmd = err;
}

static void cuda_stream_destroy(VirtIOArg *arg)
{
    cudaError_t err = 0;
    uint32_t idx;
    func();
    idx = arg->flag;
    cudaError( (err=cudaStreamDestroy(cudaStream[idx]) ));
    // cudaError( (err=cudaStreamDestroy(stream) ));
    arg->cmd = err;
    memset(&cudaStream[idx], 0, sizeof(cudaStream_t));
}

static void cuda_event_create(VirtIOArg *arg)
{
    cudaError_t err = 0;
    uint32_t idx = 0;
    func();
    idx = cudaEventNum;
    cudaError( (err=cudaEventCreate(&cudaEvent[idx]) ));
    arg->flag = (uint64_t)idx;
    arg->cmd = err;
    cudaEventNum = (cudaEventNum+1)%CudaEventMaxNum;
    debug("create event %lu, idx is %u\n", (uint64_t)cudaEvent[idx], idx);
}

static void cuda_event_create_with_flags(VirtIOArg *arg)
{
    cudaError_t err = 0;
    uint32_t idx = 0;
    unsigned int flag=0;
    func();
    idx = cudaEventNum;
    flag = arg->flag;
    cudaError( (err=cudaEventCreateWithFlags(&cudaEvent[idx], flag) ));
    arg->dst = (uint64_t)idx;
    arg->cmd = err;
    cudaEventNum = (cudaEventNum+1)%CudaEventMaxNum;
    debug("create event %lu with flag %u, idx is %u\n", \
        (uint64_t)cudaEvent[idx], flag, idx);
}

static void cuda_event_destroy(VirtIOArg *arg)
{
    cudaError_t err = 0;
    uint32_t idx = 0;
    func();
    idx = arg->flag;
    cudaError( (err=cudaEventDestroy(cudaEvent[idx])) );
    arg->cmd = err;
    debug("destroy event %lu\n", (uint64_t)cudaEvent[idx]);
    memset(&cudaEvent[idx], 0, sizeof(cudaEvent_t));
}

static void cuda_event_record(VirtIOArg *arg)
{
    cudaError_t err = 0;
    uint32_t idx = 0, sidx = 0;
    func();
    idx = arg->src;
    sidx = arg->dst;
    debug("event idx = %u\n", idx);
    if(sidx == (uint32_t)-1)
    {
        // debug("record event= %u , streams = 0\n", cudaEvent[idx]);
        cudaError( (err=cudaEventRecord(cudaEvent[idx], 0)) );
    }
    else
    {
//      debug("record event %u, stream=%u\n", event, stream);
        cudaError( (err=cudaEventRecord(cudaEvent[idx], cudaStream[sidx])) );
    }
    arg->cmd = err;
}

static void cuda_event_synchronize(VirtIOArg *arg)
{
    cudaError_t err = 0;
    uint32_t idx = 0;
    func();
    idx = arg->flag;
    // debug("record event %u , idx = %lu\n", cudaEvent[idx], idx);
    cudaError( (err=cudaEventSynchronize(cudaEvent[idx])) );
    arg->cmd = err;
}

static void cuda_event_elapsedtime(VirtIOArg *arg)
{
    cudaError_t err = 0;
    uint64_t start_idx, stop_idx;
    float        time = 0;
    func();
    // unsigned int id = get_current_id( (unsigned int)arg->tid );
    start_idx = arg->src;
    stop_idx = arg->dst;
    debug("start_idx = %lu , stop_idx = %lu\n", start_idx, stop_idx);
    cudaError( (err=cudaEventElapsedTime(&time, cudaEvent[start_idx], cudaEvent[stop_idx])) );
    arg->cmd = err;
    arg->flag = (uint64_t)time;
    // debug("event start %d to end %d, elapsedtime %f\n", (int)cudaEvent[start_idx], (int)cudaEvent[stop_idx], time);
}

static void cuda_thread_synchronize(VirtIOArg *arg)
{
    cudaError_t err = 0;
    func();
    // cudaError( (err=cudaThreadSynchronize()) );
    cudaError( (err=cudaDeviceSynchronize()) );
    arg->cmd = err;
}

static void cuda_device_synchronize(VirtIOArg *arg)
{
    cudaError_t err = 0;
    func();
    cudaError( (err=cudaDeviceSynchronize()) );
    arg->cmd = err;
}

static void cuda_get_last_error(VirtIOArg *arg)
{
    cudaError_t err = 0;
    func();
    cudaError( (err=cudaGetLastError()) );
    arg->cmd = err;
}

static void cuda_mem_get_info(VirtIOArg *arg)
{
    cudaError_t err = 0;
    size_t freeMem, totalMem;
    func();
    cudaError( (err=cudaMemGetInfo(&freeMem, &totalMem)) );
    arg->cmd = err;
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

static void init_worker_context(int port_id, int dev_id)
{
    func();
    int tid = port_id % total_port;
    worker_cur_device[tid] = dev_id;
    cuError( cuCtxSetCurrent(cudaDevices[dev_id].context) );
    debug("thread %d => device %d\n", tid, dev_id);
}

static void deinit_worker_context(int dev_id)
{
    func();
}

/*
* worker process of thread
*/
static void *worker_processor(void *arg)
{
    VirtIOArg *msg;
    int ret=0;
    Workload *work = (Workload*)arg;
    VirtIOSerialPort *port = work->port;
    int port_id = port->id;
    int device_id = work->device_id;
    int tid = port_id%total_port;
    debug("worker thread id=%d\n", tid);
    init_worker_context(port_id, device_id);

    while(1) {
        msg = message_queue_read(&worker_queue[tid]);
        switch(msg->cmd) {
            case VIRTIO_CUDA_HELLO:
            cuda_gpa_to_hva(msg);
            break;
        case VIRTIO_CUDA_REGISTERFATBINARY:
            cuda_register_fatbinary(msg);
            break;
        case VIRTIO_CUDA_UNREGISTERFATBINARY:
            cuda_unregister_fatinary(msg, tid);
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
            message_queue_message_free(&worker_queue[tid], msg);
            deinit_worker_context(device_id);
            return NULL;
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
            cuda_set_device_flags(msg);
            break;
        case VIRTIO_CUDA_DEVICERESET:
            cuda_device_reset(msg, tid);
            break;
        case VIRTIO_CUDA_STREAMCREATE:
            cuda_stream_create(msg);
            break;
        case VIRTIO_CUDA_STREAMDESTROY:
            cuda_stream_destroy(msg);
            break;
        case VIRTIO_CUDA_EVENTCREATE:
            cuda_event_create(msg);
            break;
        case VIRTIO_CUDA_EVENTCREATEWITHFLAGS:
            cuda_event_create_with_flags(msg);
            break;
        case VIRTIO_CUDA_EVENTDESTROY:
            cuda_event_destroy(msg);
            break;
        case VIRTIO_CUDA_EVENTRECORD:
            cuda_event_record(msg);
            break;
        case VIRTIO_CUDA_EVENTSYNCHRONIZE:
            cuda_event_synchronize(msg);
            break;
        case VIRTIO_CUDA_EVENTELAPSEDTIME:
            cuda_event_elapsedtime(msg);
            break;
        case VIRTIO_CUDA_THREADSYNCHRONIZE:
            cuda_thread_synchronize(msg);
            break;
        case VIRTIO_CUDA_GETLASTERROR:
            cuda_get_last_error(msg);
            break;
        case VIRTIO_CUDA_MEMCPY_ASYNC:
            cuda_memcpy_async(msg, tid);
            break;
        case VIRTIO_CUDA_MEMSET:
            cuda_memset(msg, tid);
            break;
        case VIRTIO_CUDA_DEVICESYNCHRONIZE:
            cuda_device_synchronize(msg);
            break;
        case VIRTIO_CUDA_MEMGETINFO:
            cuda_mem_get_info(msg);
            break;
        default:
            error("[+] header.cmd=%u, nr= %u \n", \
                msg->cmd, _IOC_NR(msg->cmd));
            return NULL;
        }
        message_queue_message_free(&worker_queue[tid], msg);
        ret = virtio_serial_write(port, (const uint8_t *)msg, sizeof(VirtIOArg));
        if (ret < sizeof(VirtIOArg)) {
            error("write error.\n");
            virtio_serial_throttle_port(port, true);
        }
        debug("[+] WRITE BACK\n");
    }
    return NULL;
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
    int i;
    func();

    debug("port->id=%d, len=%ld\n", port->id, len);
    tid = port->id % total_port;
    for(i =0; i< len/sizeof(VirtIOArg); i++) {
        VirtIOArg *mq_block = message_queue_message_alloc_blocking(&worker_queue[tid]);
        memcpy((void *)mq_block, buf, sizeof(VirtIOArg));
        message_queue_write(&worker_queue[tid], mq_block);
    }

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
    VirtIOArg *poison;
    func();
    debug("port id=%d, enable=%d\n", port->id, enable);

    if(!enable && global_deinitialized) {
        if (!total_port)
            return;
        thread_id = port_id%total_port;
        debug("Ending thread %d computing workloads and queue!\n",thread_id);
        poison = message_queue_message_alloc_blocking(
            &worker_queue[thread_id]);
        poison->cmd = VIRTIO_CUDA_CONFIGURECALL;
        message_queue_write(&worker_queue[thread_id], poison);
        qemu_thread_join(&worker_threads[thread_id]);
        message_queue_destroy(&worker_queue[thread_id]);
        
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

static void spawn_thread_by_port(VirtIOSerialPort *port)
{
    char thread_name[16];
    int port_id = port->id;
    int thread_id = port_id%total_port;
    debug("Starting thread %d computing workloads and queue!\n",thread_id);
    // initialize message queue for worker_threads[thread_id]
    message_queue_init(&worker_queue[thread_id], sizeof(VirtIOArg), 512);
    works[thread_id].device_id = (port_id-1)%total_device;
    works[thread_id].port = port;
    sprintf(thread_name, "thread%d", thread_id);
    qemu_thread_create(&worker_threads[thread_id], thread_name, 
            worker_processor, &works[thread_id], QEMU_THREAD_JOINABLE);
    cudaFunctionNum[thread_id]=0;
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
    // debug("nr active ports =%d\n", total_port);

    spawn_thread_by_port(port);
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
    init_device(vser);
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
    deinit_device(vser);
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
