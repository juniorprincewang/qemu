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
#include "hw/virtio/virtio-access.h"
#include "qapi/error.h"
#include "qapi/qapi-events-char.h"
#include "exec/cpu-common.h"    // cpu_physical_memory_rw

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h> //PATH: /usr/local/cuda/include/builtin_types.h
// #include <driver_types.h>   // cudaDeviceProp
#include <cublas_v2.h>
#include <curand.h>
#include "virtio-ioc.h"
#include <openssl/hmac.h> // hmac EVP_MAX_MD_SIZE
/*Encodes Base64 */
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <stdint.h> // uint32_t ...
#include <limits.h> // CHAR_BIT , usually 8

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

typedef struct VirtConsole {
    VirtIOSerialPort parent_obj;
    char *privatekeypath;    
    char *hmac_path;
    CharBackend chr;
    guint watch;
} VirtConsole;

typedef struct KernelConf {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
} KernelConf_t ;

static int total_device;   // total GPU device
static int total_port;     // port count
static QemuMutex total_port_mutex;

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

#define cuErrorExit(call) { \
    cudaError_t err; \
    if ( (err = (call)) != cudaSuccess) { \
        char *str; \
        cuGetErrorName(err, (const char**)&str); \
        fprintf(stderr, "Got error %s at %s:%d\n", str, \
                __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
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

#define cublasCheck(fn) { \
        cublasStatus_t __err = fn; \
        if (__err != CUBLAS_STATUS_SUCCESS) { \
            error("Fatal cublas error: %d (at %s:%d)\n", \
                (int)(__err), __FILE__, __LINE__); \
        } \
}

#define curandCheck(fn) { \
        curandStatus_t __err = fn; \
        if (__err != CURAND_STATUS_SUCCESS) { \
            error("Fatal curand error: %d (at %s:%d)\n", \
                (int)(__err), __FILE__, __LINE__); \
        } \
}

#define execute_with_context(call, context) {\
    cuErrorExit(cuCtxPushCurrent(context));\
    cudaError( (call) );\
    cuErrorExit(cuCtxPopCurrent(&context));\
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

static VOL *find_vol_by_vaddr(uint64_t vaddr, struct list_head *header);
static uint64_t map_device_addr_by_vaddr(uint64_t vaddr, struct list_head *header);

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

static void deinit_primary_context(CudaContext *ctx)
{
    VOL *vol, *vol2;
    HVOL *hvol, *hvol2;
    ctx->dev         = 0;
    ctx->moduleCount = 0;
    ctx->initialized = 0;
    memset(&ctx->cudaStreamBitmap, ~0, sizeof(ctx->cudaStreamBitmap));
    memset(&ctx->cudaEventBitmap, ~0, sizeof(ctx->cudaEventBitmap));
    memset(ctx->cudaStream, 0, sizeof(cudaStream_t)*CudaStreamMaxNum);
    memset(ctx->cudaEvent, 0, sizeof(cudaEvent_t)*CudaEventMaxNum);
    // free struct list
    list_for_each_entry_safe(vol, vol2, &ctx->vol, list) {
        list_del(&vol->list);
        free(vol);
    }
    list_for_each_entry_safe(hvol, hvol2, &ctx->host_vol, list) {
        list_del(&hvol->list);
        free(hvol);
    }
    memset(ctx->modules, 0, sizeof(ctx->modules));
}

static void cuda_register_fatbinary(VirtIOArg *arg, ThreadContext *tctx)
{
    void *fat_bin;
    int m_idx;
    uint32_t fatbin_size    = arg->srcSize;
    CudaContext *ctx        = &tctx->contexts[DEFAULT_DEVICE];
    func();
/*    // check out hmac
    if( (fat_bin=gpa_to_hva((hwaddr)arg->dst, fatbin_size))==NULL) {
        error("Failed to translate address in fatbinary registeration!\n");
        arg->cmd = cudaErrorUnknown;
        return;
    }*/
    /*very first initialize in this port*/
/*    if (! (tctx->deviceBitmap & 1<<DEFAULT_DEVICE)) {
        tctx->cur_dev = DEFAULT_DEVICE;
        ctx->moduleCount = 0;
        memset(ctx->modules, 0, sizeof(ctx->modules));
        cuErrorExit(cuDeviceGet(&ctx->dev, DEFAULT_DEVICE));
    }*/
    fat_bin = malloc(fatbin_size);
    cpu_physical_memory_read((hwaddr)arg->dst, fat_bin, fatbin_size);

    m_idx = ctx->moduleCount++;
    debug("fat_bin gva is 0x%lx\n", (uint64_t)arg->src);
    debug("fat_bin gpa is 0x%lx\n", (uint64_t)arg->dst);
    debug("fat_bin size is 0x%x\n", fatbin_size);
    debug("fat_bin hva is 0x%lx\n", (uint64_t)fat_bin);
    debug("fat_bin hva is at %p\n", fat_bin);
    debug("module = %d\n", m_idx);
    if (ctx->moduleCount > CudaModuleMaxNum-1) {
        error("Fatbinary number is overflow.\n");
        exit(-1);
    }
    ctx->modules[m_idx].handle              = (size_t)arg->src;
    ctx->modules[m_idx].fatbin_size         = fatbin_size;
    ctx->modules[m_idx].cudaKernelsCount    = 0;
    ctx->modules[m_idx].cudaVarsCount       = 0;
    ctx->modules[m_idx].fatbin              = fat_bin;
    
    //assert(*(uint64_t*)fat_bin ==  *(uint64_t*)fatBinAddr);
    /* check binary
    if(0==check_hmac(fat_bin, arg->srcSize)) {
        arg->cmd = cudaErrorMissingConfiguration;
        return ;
    }
    */
    arg->cmd = cudaSuccess;
}

static void cuda_unregister_fatbinary(VirtIOArg *arg, ThreadContext *tctx)
{
    int i=0, idx=0;
    CudaContext *ctx = NULL;
    CudaModule *mod = NULL;
    size_t handle   = (size_t)arg->src;
    
    func();
    for (idx = 0; idx < tctx->deviceCount; idx++) {
        if (idx==0 || tctx->deviceBitmap & 1<<idx) {
            ctx = &tctx->contexts[idx];
            for (i=0; i < ctx->moduleCount; i++) {
                mod = &ctx->modules[i];
                if (mod->handle == handle) {
                    debug("Unload module 0x%lx\n", mod->handle);
                    for(int j=0; j<mod->cudaKernelsCount; j++) {
                        free(mod->cudaKernels[j].func_name);
                    }
                    for(int j=0; j<mod->cudaVarsCount; j++) {
                        free(mod->cudaVars[j].addr_name);
                    }
                    if(ctx->initialized)
                        cuErrorExit(cuModuleUnload(mod->module));
                    free(mod->fatbin);
                    memset(mod, 0, sizeof(CudaModule));
                    break;
                }
            }
            ctx->moduleCount--;
            if (!ctx->moduleCount && ctx->initialized) {
                cuErrorExit(cuDevicePrimaryCtxRelease(ctx->dev));
                deinit_primary_context(ctx);
            }
        }
    }
}

static void cuda_register_function(VirtIOArg *arg, ThreadContext *tctx)
{
    hwaddr func_name_gpa;
    size_t func_id;
    int name_size;
    unsigned int fatbin_size;
    int i       = 0;
    CudaKernel *kernel;
    size_t fatbin_handle;
    CudaModule *cuda_module = NULL;
    CudaContext *ctx        = &tctx->contexts[DEFAULT_DEVICE];
    int m_num   = ctx->moduleCount;
    int kernel_count = -1;
    func();

    fatbin_handle   = (size_t)arg->src;
    fatbin_size     = arg->srcSize;
    for (i=0; i < m_num; i++) {
        if (ctx->modules[i].handle == fatbin_handle 
                && ctx->modules[i].fatbin_size == fatbin_size) {
            cuda_module = &ctx->modules[i];
            break;
        }   
    }
    if (!cuda_module) {
        error("Failed to find such fatbinary 0x%lx\n", fatbin_handle);
        arg->cmd = cudaErrorInvalidValue;
        return;
    }
    kernel_count = cuda_module->cudaKernelsCount++;
    if(kernel_count >= CudaFunctionMaxNum) {
        error("kernel number is overflow.\n");
        arg->cmd = cudaErrorUnknown;
        return;
    }
    func_name_gpa   = (hwaddr)arg->dst;
    func_id         = (size_t)arg->flag;
    name_size       = arg->dstSize;
    // initialize the CudaKernel
    kernel                  = &cuda_module->cudaKernels[kernel_count];
    kernel->func_name       = malloc(name_size);
    kernel->func_name_size  = name_size;
    cpu_physical_memory_read(func_name_gpa, kernel->func_name, name_size);
    kernel->func_id         = func_id;
    debug(  "Loading module... fatbin = 0x%lx, fatbin size=0x%x, name='%s',"
            " name size=0x%x, func_id=0x%lx, kernel_count = %d\n", 
            fatbin_handle, fatbin_size, kernel->func_name,
            name_size, func_id, kernel_count);
    // cuErrorExit(cuCtxPushCurrent(ctx->context));
    // cuErrorExit(cuModuleGetFunction(&kernel->kernel_func, cuda_module->module, kernel->func_name));
    // cuErrorExit(cuCtxPopCurrent(&ctx->context));
    arg->cmd = cudaSuccess;
}

static void cuda_register_var(VirtIOArg *arg, ThreadContext *tctx)
{
    hwaddr var_name_gpa;
    size_t host_var;
    int fatbin_size;
    int name_size;
    CudaContext *ctx = &tctx->contexts[DEFAULT_DEVICE];
    int m_num       = ctx->moduleCount;
    int var_count   = 0;
    int i           = 0;
    CudaMemVar      *var;
    size_t          fatbin_handle;
    CudaModule      *cuda_module = NULL;

    func();

    fatbin_handle   = (size_t)arg->src;
    fatbin_size     = arg->srcSize;
    for (i=0; i < m_num; i++) {
        if (ctx->modules[i].handle == fatbin_handle 
                && ctx->modules[i].fatbin_size == fatbin_size) {
            cuda_module = &ctx->modules[i];
            break;
        }   
    }
    if (!cuda_module) {
        error("Failed to find such fatbinary 0x%lx\n", fatbin_handle);
        arg->cmd = cudaErrorInvalidValue;
        return;
    }
    var_count = cuda_module->cudaVarsCount++;
    if(var_count >= CudaVariableMaxNum) {
        error("var number is overflow.\n");
        arg->cmd = cudaErrorUnknown;
        return;
    }

    var_name_gpa    = (hwaddr)arg->dst;
    host_var        = (size_t)arg->flag;
    name_size       = arg->dstSize;
    // initialize the CudaKernel
    var                 = &cuda_module->cudaVars[var_count];
    var->addr_name      = malloc(name_size);
    var->addr_name_size = name_size;
    var->global         = arg->param2 ? 1 : 0;

    cpu_physical_memory_read(var_name_gpa, var->addr_name, name_size);
    var->host_var   = host_var;
    debug(  "Loading module... fatbin = 0x%lx, fatbin size=0x%x, var name='%s',"
            " name size=0x%x, host_var=0x%lx, var_count = %d, global =%d\n", 
            fatbin_handle, fatbin_size, var->addr_name,
            name_size, host_var, var_count, var->global);
    // cuErrorExit(cuCtxPushCurrent(ctx->context));
    // cuErrorExit(cuModuleGetGlobal(&var->device_ptr, &var->mem_size, cuda_module->module, var->addr_name));
    // cuErrorExit(cuCtxPopCurrent(&ctx->context));
    arg->cmd = cudaSuccess;
}

static void cuda_set_device(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = NULL;
    cudaError_t err = -1;
    func();
    int dev_id = (int)(arg->flag);
    debug("set devices=%d\n", dev_id);
    if (dev_id < 0 || dev_id > tctx->deviceCount-1) {
        error("setting error device = %d\n", dev_id);
        arg->cmd = cudaErrorInvalidDevice;
        return ;
    }
    
    if (! (tctx->deviceBitmap & 1<<dev_id)) {
        tctx->cur_dev = dev_id;
        ctx = &tctx->contexts[dev_id];
        memcpy(ctx->modules, &tctx->contexts[DEFAULT_DEVICE].modules, 
                sizeof(ctx->modules));
        cuErrorExit(cuDeviceGet(&ctx->dev, dev_id));
        // init_device_module(ctx);
    }
    cudaError(err = cudaSetDevice(dev_id));
    arg->cmd = err;
    /* clear kernel function addr in parent process, 
    because cudaSetDevice will clear all resources related with host thread.
    */
}

static void init_device_module(CudaContext *ctx)
{
    int i=0, j=0;
    CudaModule *module = NULL;
    CudaKernel *kernel = NULL;
    CudaMemVar *var = NULL;
    debug("sub module number = %d\n", ctx->moduleCount);
    for (i=0; i < ctx->moduleCount; i++) {
        module = &ctx->modules[i];
        if (!module)
            return;
        cuError(cuModuleLoadData(&module->module, module->fatbin));
        debug("kernel count = %d\n", module->cudaKernelsCount);
        for(j=0; j<module->cudaKernelsCount; j++) {
            kernel = &module->cudaKernels[j];
            cuError(cuModuleGetFunction(&kernel->kernel_func, module->module, kernel->func_name));
        }
        debug("var count = %d\n", module->cudaVarsCount);
        for(j=0; j<module->cudaVarsCount; j++) {
            var = &module->cudaVars[j];
            cuError(cuModuleGetGlobal(&var->device_ptr, &var->mem_size, module->module, var->addr_name));
        }
    }
}

static void init_primary_context(CudaContext *ctx)
{
    if(!ctx->initialized) {
        cuErrorExit(cuDeviceGet(&ctx->dev, ctx->tctx->cur_dev));
        cuErrorExit(cuDevicePrimaryCtxRetain(&ctx->context, ctx->dev));
        cuErrorExit(cuCtxSetCurrent(ctx->context));
        ctx->initialized = 1;
        ctx->tctx->deviceBitmap |= 1<< ctx->tctx->cur_dev;
        init_device_module(ctx);
    }
}

static void cuda_set_device_flags(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    func();
    unsigned int flags = (unsigned int)arg->flag;
    debug("set devices flags=%d\n", flags);
    cudaError(err = cudaSetDeviceFlags(flags));
    if (err == cudaErrorSetOnActiveProcess) 
        err = cudaSuccess;
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("set device flags error.\n");
    }
}

static void cuda_launch(VirtIOArg *arg, VirtIOSerialPort *port)
{
    uint32_t para_num=0, para_idx=0;
    uint32_t para_size=0, conf_size=0;
    cudaStream_t stream_kernel = 0;
    int i = 0;
    int j = 0;
    ThreadContext *tctx = port->thread_context;
    CudaContext *ctx    = &tctx->contexts[tctx->cur_dev];
    int m_num           = ctx->moduleCount;
    VirtIODevice *vdev  = VIRTIO_DEVICE(port->vser);
    CudaModule *cuda_module = NULL;
    CudaKernel *kernel  = NULL;
    size_t func_handle  = 0;
    void **para_buf     = NULL;
    uint64_t addr       = 0;
    
    func();
    init_primary_context(ctx);
    func_handle = (size_t)arg->flag;
    debug(" func_id = 0x%lx\n", func_handle);

    for (i=0; i < m_num; i++) {
        cuda_module = &ctx->modules[i];
        for (j=0; j < cuda_module->cudaKernelsCount; j++) {
            if (cuda_module->cudaKernels[j].func_id == func_handle) {
                kernel = &cuda_module->cudaKernels[j];
                debug("Found func_id\n");
                break;
            }
        }
    }
    if (!kernel) {
        error("Failed to find func id.\n");
        exit(-1);
        arg->cmd = cudaErrorInvalidDeviceFunction;
        return;
    }
    para_size = arg->srcSize;
    conf_size = arg->dstSize;
    
    char *para = (char *)gpa_to_hva((hwaddr)(arg->src), para_size);
    if (!para ) {
        arg->cmd = cudaErrorInvalidConfiguration;
        error("Invalid para configure.\n");
        return ;
    }
    KernelConf_t *conf=(KernelConf_t*)gpa_to_hva((hwaddr)arg->dst, conf_size);
    if (!conf) {
        arg->cmd = cudaErrorInvalidConfiguration;
        error("Invalid kernel configure.\n");
        return ;
    }
    para_num = virtio_ldl_p(vdev, para);
    debug(" para_num = %u\n", para_num);
    para_buf = malloc(para_num * sizeof(void*));
    para_idx = sizeof(uint32_t);
    for(i=0; i<para_num; i++) {
        para_buf[i] = &para[para_idx + sizeof(uint32_t)];
        addr = map_device_addr_by_vaddr(virtio_ldq_p(vdev, para_buf[i]), &ctx->vol);
        if(addr!=0) {
            debug("Found 0x%lx\n", addr);
            memcpy(para_buf[i], &addr, sizeof(uint64_t));
        }
        debug("arg %d = 0x%lx , size=%u byte\n", i, 
            virtio_ldq_p(vdev, para_buf[i]),
            virtio_ldl_p(vdev, &para[para_idx]));
        para_idx += virtio_ldl_p(vdev, &para[para_idx]) + sizeof(uint32_t);
    }

    debug("gridDim=%u %u %u\n", conf->gridDim.x, 
          conf->gridDim.y, conf->gridDim.z);
    debug("blockDim=%u %u %u\n", conf->blockDim.x,
          conf->blockDim.y, conf->blockDim.z);
    debug("sharedMem=%ld\n", conf->sharedMem);
    debug("stream=0x%lx\n", (uint64_t)(conf->stream));
    
    if(!((uint64_t)conf->stream)) {
        stream_kernel = NULL;
    } else {
        int pos=(uint64_t)conf->stream;
        if (__get_bit(&ctx->cudaStreamBitmap, pos-1)) {
            error("No such stream, pos=%d\n", pos);
            arg->cmd=cudaErrorInvalidConfiguration;
            return;
        }
        stream_kernel = ctx->cudaStream[pos-1];
    }
    debug("now stream=0x%lx\n", (uint64_t)(stream_kernel));

    cuErrorExit(cuCtxPushCurrent(ctx->context));
    cuErrorExit(cuLaunchKernel( kernel->kernel_func,
                            conf->gridDim.x, conf->gridDim.y, conf->gridDim.z,
                            conf->blockDim.x, conf->blockDim.y, conf->blockDim.z,
                            conf->sharedMem,
                            stream_kernel,
                            para_buf, NULL));
    cuErrorExit(cuCtxPopCurrent(&ctx->context));
    arg->cmd = cudaSuccess;
}

static VOL *find_vol_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    VOL *vol;
    list_for_each_entry(vol, header, list) {
        if(vol->v_addr <= vaddr && vaddr < (vol->v_addr+vol->size) )
            goto out;
    }
    vol = NULL;
out:
    return vol;
}

static HVOL *find_hvol_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    HVOL *hvol;
    list_for_each_entry(hvol, header, list) {
        if(hvol->virtual_addr <= vaddr && vaddr < (hvol->virtual_addr+hvol->size) )
            goto out;
    }
    hvol = NULL;
out:
    return hvol;
}

static uint64_t map_device_addr_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    VOL *vol = find_vol_by_vaddr(vaddr, header);
    if(vol != NULL)
        return vol->addr + (vaddr - vol->v_addr);
    return 0;
}

static uint64_t map_host_addr_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    HVOL *hvol = find_hvol_by_vaddr(vaddr, header);
    if(hvol != NULL)
        return hvol->addr + (vaddr - hvol->virtual_addr);
    return 0;
}

static void remove_hvol_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    HVOL *hvol, *hvol2;
    list_for_each_entry_safe(hvol, hvol2, header, list) {
        if (hvol->virtual_addr == vaddr) {
            debug("Found memory maped ptr=0x%lx\n", (uint64_t)hvol->addr);
            munmap((void*)hvol->addr, hvol->size);
            close(hvol->fd);
            list_del(&hvol->list);
            free(hvol);
            return;
        }
    }
    error("Found no memory maped ptr=0x%lx\n", vaddr);
}

static void cuda_memcpy(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err;
    uint32_t size;
    void *src, *dst;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    init_primary_context(ctx);
    debug("tid = %d, src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, param=0x%lx, param2=0x%lx\n",
          arg->tid,  arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, arg->param, arg->param2);
    size = arg->srcSize;
    if (arg->flag == cudaMemcpyHostToDevice) {
        // device address
        if( (dst = (void*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        if(arg->param) {
            if( (src = (void*)map_host_addr_by_vaddr(arg->src, &ctx->host_vol))==NULL) {
                error("Failed to find virtual addr %p in host vol\n", (void *)arg->src);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
        } else {
            if((src= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
                error("No such physical address 0x%lx.\n", arg->param2);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
        }
        debug("src=%p, size=0x%x, dst=%p\n", src, size, dst);
        execute_with_context( (err= cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)), ctx->context);
        arg->cmd = err;
        if(err != cudaSuccess) {
            error("memcpy cudaMemcpyHostToDevice error!\n");
        }
        return;
    } else if (arg->flag == cudaMemcpyDeviceToHost) {
        // get device address
        if( (src = (void*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==0) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        if(arg->param) {
            if( (dst = (void*)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==0) {
                error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
        } else {
            if((dst= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
                error("No such physical address 0x%lx.\n", arg->param2);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
        }
        debug("src=%p, size=0x%x, dst=%p\n", src, size, dst);
        execute_with_context( (err=cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)), ctx->context);
        arg->cmd = err;
        if (cudaSuccess != err) {
            error("memcpy cudaMemcpyDeviceToHost error!\n");
            return;
        }
        return;
    } else if (arg->flag == cudaMemcpyDeviceToDevice) {
        if( (src = (void *)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
            error("Failed to find src virtual address %p in vol\n",
                  (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        if((dst=(void *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find dst virtual address %p in vol\n",
                  (void *)arg->dst);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        debug("src=%p, size=0x%x, dst=%p\n", src, size, dst);
        execute_with_context( (err=cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)), ctx->context );
        arg->cmd = err;
        if (cudaSuccess != err) {
            error("memcpy cudaMemcpyDeviceToDevice error!\n");
        }
        return;
    } else {
        error("Error memcpy direction\n");
        arg->cmd = cudaErrorInvalidMemcpyDirection;
    }
}

static void cuda_memcpy_async(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    uint32_t size;
    void *src, *dst;
    int pos = 0;
    cudaStream_t stream=0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    init_primary_context(ctx);
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=%lu, "
        "stream=0x%lx , src2=0x%lx\n",
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
        arg->param, arg->src2);
    pos = arg->src2;
    if (pos==0) {
        stream=0;
    } else if (!__get_bit(&ctx->cudaStreamBitmap, pos-1)) {
        stream = ctx->cudaStream[pos-1];
    } else {
        error("No such stream, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    debug("stream 0x%lx\n", (uint64_t)stream);
    size = arg->srcSize;
    if (arg->flag == cudaMemcpyHostToDevice) {
        if((dst=(void*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==0) {
            error("Failed to find dst virtual addr %p in vol\n",
                  (void *)arg->dst);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        if(arg->param) {
            if( (src = (void*)map_host_addr_by_vaddr(arg->src, &ctx->host_vol))==NULL) {
                error("Failed to find virtual addr %p in host vol\n", (void *)arg->src);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
        } else {
            if((src= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
                error("No such physical address 0x%lx.\n", arg->param2);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
        }
        execute_with_context((err= cudaMemcpyAsync(dst, src, size, 
                            cudaMemcpyHostToDevice, stream)), ctx->context);
        debug("src = %p\n", src);
        debug("dst = %p\n", dst);
        arg->cmd = err;
        if(err != cudaSuccess) {
            error("memcpy async HtoD error!\n");
        }
        return;
    } else if (arg->flag == cudaMemcpyDeviceToHost) {
        if((src=(void*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==0) {
            error("Failed to find virtual addr %p in vol or hvol\n",
                  (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        if(arg->param) {
            if( (dst = (void*)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==0) {
                error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
        } else {
            if((dst= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
                error("No such physical address 0x%lx.\n", arg->param2);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
        }
        execute_with_context( (err=cudaMemcpyAsync(dst, src, size, 
                        cudaMemcpyDeviceToHost, stream)), ctx->context );
        debug("src = %p\n", src);
        debug("dst = %p\n", dst);
        debug("size = 0x%x\n", size);
        debug("(int)dst[0] = 0x%x\n", *(int*)dst);
        debug("(int)dst[1] = 0x%x\n", *((int*)dst+1));
        arg->cmd = err;
        if (err != cudaSuccess) {
            error("memcpy async DtoH error!\n");
        }
        return;
    } else if (arg->flag == cudaMemcpyDeviceToDevice) {
        if((src = (void*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==0) {
            error("Failed to find virtual addr %p in vol\n",
                  (void *)arg->src);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        if((dst = (void*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==0) {
            error("Failed to find virtual addr %p in vol\n",
                  (void *)arg->dst);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        execute_with_context( (err=cudaMemcpyAsync(dst, src, size, 
                            cudaMemcpyDeviceToDevice, stream)), ctx->context );
        arg->cmd = err;
        if (err != cudaSuccess) {
            error("memcpy async DtoD error!\n");
        }
        return;
    } else {
        error("No such memcpy direction.\n");
        arg->cmd= cudaErrorInvalidMemcpyDirection;
        return;
    }
}

static void cuda_memcpy_to_symbol(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    size_t size;
    void *src, *dst;
    size_t var_handle=0;
    size_t var_offset=0;
    int found       = 0;
    int i           = 0;
    int j           = 0;
    int var_count   = 0;
    int m_num       = 0;
    CudaMemVar      *var;
    CudaModule      *cuda_module;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=%lu, "
        "param=0x%lx, param2=0x%lx \n",
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
        arg->param, arg->param2);
    var_handle = (size_t)arg->dst;
    m_num = ctx->moduleCount;
    for (i=0; i < m_num; i++) {
        cuda_module = &ctx->modules[i];
        var_count = cuda_module->cudaVarsCount;
        for (j=0; j < var_count; j++) {
            var = &cuda_module->cudaVars[j];
            if (var->host_var == var_handle) {
                found = 1;
                break;
            }
        }
    }
    if (found == 0) {
        error("Failed to find such var handle 0x%lx\n", var_handle);
        arg->cmd = cudaErrorNoKernelImageForDevice;
        return;
    }

    size = arg->srcSize;
    if (arg->flag != 0) {
        if((src = (void*)map_host_addr_by_vaddr(arg->src, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    } else {
        if((src= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
            error("No such src physical address 0x%lx.\n", arg->param2);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    }
    var_offset = (size_t)arg->param;
    dst = (void *)(var->device_ptr+var_offset);
    execute_with_context(err=cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), ctx->context);
    arg->cmd = err;
    if(err != cudaSuccess) {
        error("memcpy to symbol HtoD error!\n");
    }
}

static void cuda_memcpy_from_symbol(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    size_t  size;
    void    *dst, *src;
    size_t  var_handle=0;
    size_t  var_offset=0;
    int found       = 0;
    int i           = 0;
    int j           = 0;
    int var_count   = 0;
    CudaMemVar      *var;
    CudaModule      *cuda_module;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int m_num       = 0;

    func();
    init_primary_context(ctx);
    debug(  "src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=0x%lx, "
            "param=0x%lx, param2=0x%lx,\n",
            arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
            arg->param, arg->param2);
 
    var_handle = (size_t)arg->src;
    m_num = ctx->moduleCount;
    for (i=0; i < m_num; i++) {
        cuda_module = &ctx->modules[i];
        var_count = cuda_module->cudaVarsCount;
        for (j=0; j < var_count; j++) {
            var = &cuda_module->cudaVars[j];
            if (var->host_var == var_handle) {
                found = 1;
                break;
            }
        }
    }
    if (found == 0) {
        error("Failed to find such var handle 0x%lx\n", var_handle);
        arg->cmd = cudaErrorNoKernelImageForDevice;
        return;
    }
    size = arg->dstSize;
    var_offset = (size_t)arg->param;
    src = (void*)(var->device_ptr+var_offset);
    if (arg->flag != 0) {
        if((dst = (void*)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    } else {
        if((dst= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
            error("No such dst physical address 0x%lx.\n", arg->param2);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    }
    execute_with_context(err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("memcpy from symbol DtoH error!\n");
    }
}


static void cuda_memset(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    size_t count;
    int value;
    uint64_t dst;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    count = (size_t)(arg->param);
    value = (int)(arg->param2);
    debug("dst=0x%lx, value=0x%x, count=0x%lx\n", arg->dst, value, count);
    if((dst = map_device_addr_by_vaddr(arg->dst, &ctx->vol))==0) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = cudaErrorInvalidValue;
        return;
    }
    debug("dst=0x%lx\n", dst);
    execute_with_context( (err= cudaMemset((void*)dst, value, count)), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess)
        error("memset memory error!\n");
}

static void cuda_malloc(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    void *devPtr=NULL;
    size_t size = 0;
    VOL *vol = NULL;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    init_primary_context(ctx);
    size = arg->srcSize;    
    execute_with_context(err= cudaMalloc((void **)&devPtr, size), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("Alloc memory error!\n");
        return;
    }

    vol = (VOL *)malloc(sizeof(VOL));
    vol->addr   = (uint64_t)devPtr;
    vol->v_addr = (uint64_t)(devPtr + VOL_OFFSET);
    arg->dst    = (uint64_t)(vol->v_addr);
    vol->size   = size;
    list_add_tail(&vol->list, &ctx->vol);
    debug("actual devPtr=0x%lx, virtual ptr=0x%lx, size=0x%lx,"
          "ret value=0x%x\n", (uint64_t)devPtr, arg->dst, size, err);
}

static int set_shm(size_t size, char *file_path)
{
    int res=0;
    int mmap_fd = shm_open(file_path, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
    if (mmap_fd == -1) {
        error("Failed to open with errno %s\n", strerror(errno));
        return 0;
    }
    shm_unlink(file_path);
    // extend
    res = ftruncate(mmap_fd, size);
    if (res == -1) {
        error("Failed to ftruncate.\n");
        return 0;
    }
    return mmap_fd;
}

static void mmapctl(VirtIOArg *arg, VirtIOSerialPort *port)
{
    size_t size;
    void *dst;
    void *addr = NULL;
    char path[128];
    int fd = 0;
    ThreadContext *tctx = port->thread_context;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, param=0x%lx\n",
          arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->param);

    size = arg->srcSize;
    snprintf(path, sizeof(path), "/qemu_%lu_%u_%lx", 
            (long)getpid(), port->id, arg->src);
    fd = set_shm(size, path);
    if(!fd)
        return;
    if(arg->param) {
        int blocks = arg->param;
        uint64_t *gpa_array = (uint64_t*)gpa_to_hva((hwaddr)arg->dst, blocks);
        if(!gpa_array) {
            error("Failed to get gpa_array.\n");
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        
        uint32_t offset = arg->dstSize;
        uint32_t start_offset = offset % KMALLOC_SIZE;
        int len = min(size, KMALLOC_SIZE - start_offset);
        void *tmp = gpa_to_hva((hwaddr)gpa_array[0], len);
        munmap(tmp, len);
        addr = mmap(tmp, len, PROT_READ | PROT_WRITE, MAP_SHARED|MAP_FIXED, fd, 0);
        assert(addr != MAP_FAILED);
        debug("addr1 %p is mapped to %p\n", tmp, addr);
        int rsize=size;
        rsize-=len;
        offset=len;
        int i=1;
        while(rsize) {
            len=min(rsize, KMALLOC_SIZE);
            tmp = gpa_to_hva((hwaddr)gpa_array[i++], len);
            munmap(tmp, len);
            addr = mmap(tmp, len, PROT_READ | PROT_WRITE, MAP_SHARED|MAP_FIXED, fd, offset);
            assert(addr != MAP_FAILED);
            debug("addr%d %p is mapped to %p\n", i, tmp, addr);
            offset+=len;
            rsize-=len;
        }
        debug("i=%d, KMALLOC_SIZE=0x%lx, blocks=%d\n", i, KMALLOC_SIZE, blocks);
        assert(i == blocks);
        dst = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    } else {
        addr = gpa_to_hva((hwaddr)arg->dst, size);
        munmap(addr, size);
        dst = mmap(addr, size, PROT_READ | PROT_WRITE, MAP_SHARED|MAP_FIXED, fd, 0);
        assert(dst != MAP_FAILED);
        debug("addr %p is mapped to %p\n", addr, dst);
        dst = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }
    debug("0x%lx is mapped to %p\n", arg->src, dst);
    HVOL *hvol  = (HVOL *)malloc(sizeof(HVOL));
    hvol->addr  = (uint64_t)dst;
    hvol->virtual_addr  = arg->src;
    hvol->size  = size;
    hvol->fd    = fd;
    list_add_tail(&hvol->list, &ctx->host_vol);
}

static void munmapctl(VirtIOArg *arg, VirtIOSerialPort *port)
{
    size_t size;
    void *dst;
    ThreadContext *tctx = port->thread_context;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, param=0x%lx\n",
          arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->param);

    size = arg->srcSize;    
    if(arg->param) {
        int blocks = arg->param;
        uint64_t *gpa_array = (uint64_t*)gpa_to_hva((hwaddr)arg->dst, blocks);
        if(!gpa_array) {
            error("Failed to get gpa_array.\n");
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        
        uint32_t offset = arg->dstSize;
        uint32_t start_offset = offset % KMALLOC_SIZE;
        int len = min(size, KMALLOC_SIZE - start_offset);
        dst = gpa_to_hva((hwaddr)gpa_array[0], len);
        munmap(dst, len);
        mmap(dst, len, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0);
        int rsize=size;
        rsize-=len;
        offset=len;
        int i=1;
        while(rsize) {
            len=min(rsize, KMALLOC_SIZE);
            dst = gpa_to_hva((hwaddr)gpa_array[i++], len);
            munmap(dst, len);
            mmap(dst, len, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0);
            offset+=len;
            rsize-=len;
        }
        assert(i == blocks);
    } else {
        dst = gpa_to_hva((hwaddr)arg->dst, size);
        munmap(dst, size);
        mmap(dst, size, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0);
    }
    remove_hvol_by_vaddr(arg->src, &ctx->host_vol);
    debug("Finish munmap!\n");
}

static void cuda_host_register(VirtIOArg *arg, VirtIOSerialPort *port)
{
    cudaError_t err=-1;
    size_t size;
    void *ptr;
    unsigned int flags = 0;
    ThreadContext *tctx = port->thread_context;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    init_primary_context(ctx);
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, param=0x%lx\n",
          arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, arg->param);

    size = arg->srcSize;
    flags = (unsigned int)arg->flag;
    // get host address
    if(arg->param) {
        if((ptr = (void*)map_host_addr_by_vaddr(arg->src, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    } else {
        if((ptr= gpa_to_hva((hwaddr)arg->dst, size)) == NULL) {
            error("No such physical address 0x%lx.\n", arg->dst);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    }
    debug("ptr = %p\n", ptr);
    execute_with_context((err = cudaHostRegister(ptr, size, flags)), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("host register error.\n");
    }
}

static void cuda_host_unregister(VirtIOArg *arg, VirtIOSerialPort *port)
{
    cudaError_t err=-1;
    ThreadContext *tctx = port->thread_context;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    void *ptr;
    size_t size;

    func();
    init_primary_context(ctx);
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, param=0x%lx\n",
          arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, arg->param);
    size = arg->srcSize;
    // get host address
    if(arg->param) {
        if((ptr = (void*)map_host_addr_by_vaddr(arg->src, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    } else {
        if((ptr= gpa_to_hva((hwaddr)arg->dst, size)) == NULL) {
            error("No such physical address 0x%lx.\n", arg->dst);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    }
    execute_with_context((err = cudaHostUnregister(ptr)), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("Failed to unregister memory.\n");
    }
}
static void cuda_free(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    VOL *vol, *vol2;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    debug(" ptr = 0x%lx\n", arg->src);
    list_for_each_entry_safe(vol, vol2, &ctx->vol, list) {
        if (vol->v_addr == arg->src) {
            debug(  "actual devPtr=0x%lx, virtual ptr=0x%lx\n", 
                    (uint64_t)vol->addr, arg->src);
            execute_with_context( (err= cudaFree((void*)(vol->addr))), ctx->context);
            arg->cmd = err;
            if (err != cudaSuccess) {
                error("free error.\n");
                return;
            }
            list_del(&vol->list);
            free(vol);
            return;
        }
    }
    arg->cmd = cudaErrorInvalidValue;
    error("Failed to free ptr. Not found it!\n");
}

/*
 * Let vm user see which card he actually uses
*/
static void cuda_get_device(VirtIOArg *arg, int tid)
{
    return;
}

/*
 * done by the vgpu in vm
 * this function is useless
*/
static void cuda_get_device_properties(VirtIOArg *arg)
{
    return;
}

static void cuda_get_device_count(VirtIOArg *arg)
{
    func();
    debug("Device count=%d.\n", total_device);
    arg->cmd = (int32_t)cudaSuccess;
    arg->flag = (uint64_t)total_device;
}

static void cuda_device_reset(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    func();
    cuErrorExit(cuDeviceGet(&ctx->dev, tctx->cur_dev));
    cuErrorExit(cuDevicePrimaryCtxReset(ctx->dev));
    deinit_primary_context(ctx);
    tctx->deviceBitmap &= ~(1 << tctx->cur_dev);
    arg->cmd = cudaSuccess;
    debug("reset devices\n");
}

static void cuda_stream_create(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    pos = ffs(ctx->cudaStreamBitmap);
    if (!pos) {
        error("stream number is up to %d\n", CudaStreamMaxNum);
        return;
    }
    // cudaError(err = cudaStreamCreate(&ctx->cudaStream[pos-1]));
    execute_with_context(err = cudaStreamCreate(&ctx->cudaStream[pos-1]), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("create stream error.\n");
        return;
    }
    arg->flag = (uint64_t)pos;
    debug("create stream 0x%lx, idx is %u\n",
          (uint64_t)ctx->cudaStream[pos-1], pos-1);
    __clear_bit(&ctx->cudaStreamBitmap, pos-1);
}

static void cuda_stream_create_with_flags(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    unsigned int flag=0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    pos = ffs(ctx->cudaStreamBitmap);
    if (!pos) {
        error("stream number is up to %d\n", CudaStreamMaxNum);
        return;
    }
    flag = (unsigned int)arg->flag;
    execute_with_context( (err=cudaStreamCreateWithFlags(&ctx->cudaStream[pos-1], flag)), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("create stream with flags error.\n");
        return;
    }
    arg->dst = (uint64_t)pos;
    __clear_bit(&ctx->cudaStreamBitmap, pos-1);
    debug("create stream 0x%lx with flag %u, idx is %u\n",
          (uint64_t)ctx->cudaStream[pos-1], flag, pos-1);
}

static void cuda_stream_destroy(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint32_t pos;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    pos = arg->flag;
    if (__get_bit(&ctx->cudaStreamBitmap, pos-1)) {
        error("No such stream, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    debug("destroy stream 0x%lx\n", (uint64_t)ctx->cudaStream[pos-1]);
    // cudaError((err=cudaStreamDestroy(ctx->cudaStream[pos-1]) ));
    execute_with_context((err=cudaStreamDestroy(ctx->cudaStream[pos-1])), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("destroy stream error.\n");
        return;
    }
    __set_bit(&ctx->cudaStreamBitmap, pos-1);
}

static void cuda_stream_synchronize(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint32_t pos;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    init_primary_context(ctx);
    pos = arg->flag;
    if (__get_bit(&ctx->cudaStreamBitmap, pos-1)) {
        error("No such stream, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    debug("destroy stream 0x%lx\n", (uint64_t)ctx->cudaStream[pos-1]);
    // cudaError( (err=cudaStreamSynchronize(ctx->cudaStream[pos-1]) ));
    execute_with_context( (err=cudaStreamSynchronize(ctx->cudaStream[pos-1])), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("synchronize stream error.\n");
    }
}

static void cuda_stream_wait_event(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint64_t pos;
    cudaStream_t    stream = 0;
    cudaEvent_t     event = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    init_primary_context(ctx);
    pos = arg->src;
    if (__get_bit(&ctx->cudaStreamBitmap, pos-1)) {
        error("No such stream, pos=%ld\n", pos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    stream = ctx->cudaStream[pos-1];
    debug("stream 0x%lx\n", (uint64_t)stream);
    pos = arg->dst;
    if (__get_bit(ctx->cudaEventBitmap, pos-1)) {
        error("No such event, pos=%ld\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    event = ctx->cudaEvent[pos-1];
    // event = (cudaEvent_t)arg->dst;
    debug("wait for event 0x%lx\n", (uint64_t)event);    
    execute_with_context(err = cudaStreamWaitEvent(stream, event, 0), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("failed to wait event for stream.\n");
    }
}

static void cuda_event_create(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    // cudaEvent_t event;
    func();
    init_primary_context(ctx);
    /*
    execute_with_context(err = cudaEventCreate(&event), ctx->context);
    arg->cmd = err;
    arg->flag = (uint64_t)event;
    debug("create event 0x%lx\n", (uint64_t)event);
    */
    
    for(int i=0; i<CudaEventMapMax; i++) {
        // debug("i=%d, ctx->cudaEventBitmap[i]=0x%x\n", i, ctx->cudaEventBitmap[i]);
        pos = ffs(ctx->cudaEventBitmap[i]);
        if(pos) {
            pos = i * BITS_PER_WORD + pos;
            break;
        }
    }
    if(!pos) {
        error("event number is up to %d\n", CudaEventMaxNum);
        return;
    }
    __clear_bit(ctx->cudaEventBitmap, pos-1);
    execute_with_context(err = cudaEventCreate(&ctx->cudaEvent[pos-1]), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("create event error.\n");
        return;
    }
    arg->flag = (uint64_t)pos;
    debug("tid %d create event 0x%lx, pos(arg->flag) is 0x%lx\n",
          arg->tid, (uint64_t)ctx->cudaEvent[pos-1], arg->flag);
    
}

static void cuda_event_create_with_flags(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    unsigned int flag=0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    init_primary_context(ctx);
    
    for(int i=0; i<CudaEventMapMax; i++) {
        pos = ffs(ctx->cudaEventBitmap[i]);
        if(pos) {
            pos = i * BITS_PER_WORD + pos;
            break;
        }
    }
    if(!pos) {
        error("event number is up to %d\n", CudaEventMaxNum);
        return;
    }
    __clear_bit(ctx->cudaEventBitmap, pos-1);
    flag = arg->flag;
    execute_with_context((err=cudaEventCreateWithFlags(&ctx->cudaEvent[pos-1], flag)), ctx->context);
    
    // execute_with_context( (err=cudaEventCreateWithFlags(&event, flag)), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("create event with flags error.\n");
        return;
    }
    arg->dst = (uint64_t)pos;
    debug("create event 0x%lx with flag %u, pos is %u\n",
          (uint64_t)ctx->cudaEvent[pos-1], flag, pos);
}

static void cuda_event_destroy(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    int pos = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cudaEvent_t event;
    func();
    init_primary_context(ctx);
    // event = (cudaEvent_t)arg->flag;
    // debug("destroy event 0x%lx\n", (uint64_t)event);
    // execute_with_context( (err=cudaEventDestroy(event)), ctx->context);
    // arg->cmd = err;
    
    pos = (int)arg->flag;
    if (__get_bit(ctx->cudaEventBitmap, pos-1)) {
        error("No such event, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    event = ctx->cudaEvent[pos-1];
    debug("tid %d destroy event [pos=%d] 0x%lx\n", arg->tid, pos, (uint64_t)event);
    __set_bit(ctx->cudaEventBitmap, pos-1);
    execute_with_context( (err=cudaEventDestroy(event)), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("destroy event error.\n");
        return;
    }
    
}

static void cuda_event_query(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cudaEvent_t event;
    func();
    init_primary_context(ctx);
    // event = (cudaEvent_t)arg->flag;
    
    pos = arg->flag;
    if (__get_bit(ctx->cudaEventBitmap, pos-1)) {
        error("No such event, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    event = ctx->cudaEvent[pos-1];
    debug("query event 0x%lx\n", (uint64_t)event);
    cuErrorExit(cuCtxPushCurrent(ctx->context));
    err=cudaEventQuery(event);
    cuErrorExit(cuCtxPopCurrent(&ctx->context));
    arg->cmd = err;
}

static void cuda_event_record(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint64_t epos = 0;
    uint64_t spos = 0;
    cudaStream_t stream;
    cudaEvent_t event;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    init_primary_context(ctx);
    epos = arg->src;
    spos = arg->dst;
    debug("event pos = 0x%lx\n", epos);
    // event = (cudaEvent_t)arg->src;
    
    if (epos<=0 || __get_bit(ctx->cudaEventBitmap, epos-1)) {
        error("No such event, pos=0x%lx\n", epos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    event = ctx->cudaEvent[epos-1];
    debug("stream pos = 0x%lx\n", spos);
    if (spos==0) {
        stream=0;
    } else if (!__get_bit(&ctx->cudaStreamBitmap, spos-1)) {
        stream = ctx->cudaStream[spos-1];
    } else {
        error("No such stream, pos=0x%lx\n", spos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    debug("record event 0x%lx, stream=0x%lx\n", 
        (uint64_t)event, (uint64_t)stream);
    execute_with_context((err=cudaEventRecord(event, stream)), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("record event error.\n");
    }
}

static void cuda_event_synchronize(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cudaEvent_t event;
    func();
    init_primary_context(ctx);
    // event = (cudaEvent_t)arg->flag;
    // debug("sync event 0x%lx\n", (uint64_t)event);
    // execute_with_context( (err=cudaEventSynchronize(event)), ctx->context);
    
    pos = arg->flag;
    if (__get_bit(ctx->cudaEventBitmap, pos-1)) {
        error("No such event, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    event = ctx->cudaEvent[pos-1];
    debug("sync event 0x%lx\n", (uint64_t)event);
    execute_with_context( (err=cudaEventSynchronize(event)), ctx->context);
    
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("synchronize event error.\n");
        return;
    }
}

static void cuda_event_elapsedtime(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    int start_pos, stop_pos;
    float time = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cudaEvent_t start, stop;

    func();
    init_primary_context(ctx);
    // start = (cudaEvent_t)arg->src;
    // stop = (cudaEvent_t)arg->dst;
    
    start_pos = (int)arg->src;
    stop_pos = (int)arg->dst;
    if (__get_bit(ctx->cudaEventBitmap, start_pos-1)) {
        error("No such event, pos=%d\n", start_pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    if (__get_bit(ctx->cudaEventBitmap, stop_pos-1)) {
        error("No such event, pos=%d\n", stop_pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    start = ctx->cudaEvent[start_pos-1];
    stop = ctx->cudaEvent[stop_pos-1];
    debug("start event 0x%lx\n", (uint64_t)start);
    debug("stop event 0x%lx\n", (uint64_t)stop);
    execute_with_context( (err=cudaEventElapsedTime(&time, 
                                 start, stop)), ctx->context);    
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("event calc elapsed time error.\n");
        return;
    }
    debug("elapsed time=%g\n", time);
    cpu_physical_memory_write((hwaddr)arg->param2, &time, arg->paramSize);
}

static void cuda_device_synchronize(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    func();
    init_primary_context(ctx);
    // cudaError(err = cudaDeviceSynchronize());
    execute_with_context(err = cudaDeviceSynchronize(), ctx->context);
    arg->cmd = err;
}

static void cuda_thread_synchronize(VirtIOArg *arg, ThreadContext *tctx)
{
    func();
    /*
    * cudaThreadSynchronize is deprecated
    * cudaError( (err=cudaThreadSynchronize()) );
    */
    cuda_device_synchronize(arg, tctx);
}

static void cuda_get_last_error(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    func();
    init_primary_context(ctx);
    execute_with_context(err = cudaGetLastError(), ctx->context);
    arg->cmd = err;
}

static void cuda_peek_at_last_error(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    func();
    init_primary_context(ctx);
    execute_with_context(err = cudaPeekAtLastError(), ctx->context);
    arg->cmd = err;
}

static void cuda_mem_get_info(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    size_t freeMem, totalMem;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    execute_with_context(err = cudaMemGetInfo(&freeMem, &totalMem), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("get mem info error!\n");
        return;
    }
    arg->srcSize = freeMem;
    arg->dstSize = totalMem;
    debug("free memory = %lu, total memory = %lu.\n", freeMem, totalMem);
}

static void cublas_create(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    cublasHandle_t handle;
    hwaddr gpa = (hwaddr)(arg->dst);
    func();
    cublasCheck(status = cublasCreate_v2(&handle));
    cpu_physical_memory_write(gpa, &handle, arg->srcSize);
    arg->cmd = status;
    debug("cublas handle 0x%lx\n", (uint64_t)handle);
}

static void cublas_destroy(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    cublasHandle_t handle;
    hwaddr gpa = (hwaddr)(arg->dst);
    func();
    cpu_physical_memory_read(gpa, &handle, arg->srcSize);
    cublasCheck(status = cublasDestroy_v2(handle));
    arg->cmd = status;
    debug("cublas handle 0x%lx\n", (uint64_t)handle);
}

static void cublas_set_vector(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int size = 0;
    int n;
    int elemSize;
    int incx, incy;
    void *buf = NULL;
    void *src, *dst;
    int int_size = sizeof(int);
    int idx = 0;
    func();

    // device address
    if( (dst = (void*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
        return;
    }
    size = arg->srcSize;
    if(arg->flag) {
        if( (src = (void*)map_host_addr_by_vaddr(arg->src, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->src);
            arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
            return;
        }
    } else {
        if((src= gpa_to_hva((hwaddr)arg->src, size)) == NULL) {
            error("No such physical address 0x%lx.\n", arg->src);
            arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
            return;
        }
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
        return;
    }
    memcpy(&n, buf+idx, int_size);
    idx += int_size;
    memcpy(&elemSize, buf+idx, int_size);
    idx += int_size;
    memcpy(&incx, buf+idx, int_size);
    idx += int_size;
    memcpy(&incy, buf+idx, int_size);
    cublasCheck(status = cublasSetVector(n, elemSize, src, incx, dst, incy));
    debug("n=%d, elemSize=%d, src=%p, incx=%d, dst=%p, incy=%d\n",
        n, elemSize, src, incx, dst, incy);
    arg->cmd = status;
}

static void cublas_get_vector(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int size = 0;
    int n;
    int elemSize;
    int incx, incy;
    void *buf = NULL;
    void *src, *dst;
    int int_size = sizeof(int);
    int idx = 0;
    func();

    // device address
    if( (src = (void*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
        return;
    }
    size = arg->srcSize;
    if(arg->flag) {
        if( (dst = (void*)map_host_addr_by_vaddr(arg->param2, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->param2);
            arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
            return;
        }
    } else {
        if((dst= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
            error("No such physical address 0x%lx.\n", arg->param2);
            arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
            return;
        }
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
        return;
    }
    memcpy(&n, buf+idx, int_size);
    idx += int_size;
    memcpy(&elemSize, buf+idx, int_size);
    idx += int_size;
    memcpy(&incx, buf+idx, int_size);
    idx += int_size;
    memcpy(&incy, buf+idx, int_size);
    cublasCheck(status = cublasGetVector(n, elemSize, src, incx, dst, incy));
    debug("n=%d, elemSize=%d, src=%p, incx=%d, dst=%p, incy=%d\n",
        n, elemSize, src, incx, dst, incy);
    arg->cmd = status;
}

static void cublas_set_matrix(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int size = 0;
    int rows, cols;
    int elemSize;
    int lda, ldb;
    void *buf = NULL;
    void *src, *dst;
    int int_size = sizeof(int);
    int idx = 0;
    func();

    // device address
    if( (dst = (void*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
        return;
    }
    size = arg->srcSize;
    if(arg->flag) {
        if( (src = (void*)map_host_addr_by_vaddr(arg->src, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->src);
            arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
            return;
        }
    } else {
        if((src= gpa_to_hva((hwaddr)arg->src, size)) == NULL) {
            error("No such physical address 0x%lx.\n", arg->src);
            arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
            return;
        }
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
        return;
    }
    memcpy(&rows, buf+idx, int_size);
    idx += int_size;
    memcpy(&cols, buf+idx, int_size);
    idx += int_size;
    memcpy(&elemSize, buf+idx, int_size);
    idx += int_size;
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldb, buf+idx, int_size);
    cublasCheck(status = cublasSetMatrix(rows, cols, elemSize, 
                                            src, lda, 
                                            dst, ldb));
    arg->cmd = status;
}

static void cublas_get_matrix(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int size = 0;
    int rows, cols;
    int elemSize;
    int lda, ldb;
    void *buf = NULL;
    void *src, *dst;
    int int_size = sizeof(int);
    int idx = 0;
    func();

    // device address
    if( (src = (void*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
        return;
    }
    size = arg->srcSize;
    if(arg->flag) {
        if( (dst = (void*)map_host_addr_by_vaddr(arg->param2, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->param2);
            arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
            return;
        }
    } else {
        if((dst= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
            error("No such physical address 0x%lx.\n", arg->param2);
            arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
            return;
        }
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_MAPPING_ERROR;
        return;
    }
    memcpy(&rows, buf+idx, int_size);
    idx += int_size;
    memcpy(&cols, buf+idx, int_size);
    idx += int_size;
    memcpy(&elemSize, buf+idx, int_size);
    idx += int_size;
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldb, buf+idx, int_size);
    cublasCheck(status = cublasGetMatrix (rows, cols, elemSize, 
                                             src, lda, 
                                             dst, ldb));
    arg->cmd = status;
}

static void cublas_sgemm(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    void *buf = NULL;
    int int_size = sizeof(int);
    int idx = 0;
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m, n, k;
    int lda, ldb, ldc;
    float *A, *B, *C;
    float alpha; /* host or device pointer */
    float beta; /* host or device pointer */

    func();
    // device address
    if( (A = (float*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if( (B = (float*)map_device_addr_by_vaddr(arg->src2, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src2);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if( (C = (float*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    memcpy(&handle, buf+idx, sizeof(cublasHandle_t));
    idx += sizeof(cublasHandle_t);
    memcpy(&transa, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&transb, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&m, buf+idx, int_size);
    idx += int_size;
    memcpy(&n, buf+idx, int_size);
    idx += int_size;
    memcpy(&k, buf+idx, int_size);
    idx += int_size;
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldb, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldc, buf+idx, int_size);
    idx += int_size;
    memcpy(&alpha, buf+idx, sizeof(float));
    idx += sizeof(float);
    memcpy(&beta, buf+idx, sizeof(float));
    cublasCheck(status = cublasSgemm_v2(handle, transa, transb,
                                        m, n, k,
                                        &alpha, A, lda,
                                        B, ldb,
                                        &beta, C, ldc));
    debug("handle=%lx, transa=0x%lx, transb=0x%lx, m=%d, n=%d, k=%d,"
            " alpha=%g, A=%p, lda=%d,"
            " B=%p, ldb=%d, beta=%g, C=%p, ldc=%d\n",
            (uint64_t)handle, (uint64_t)transa, (uint64_t)transb, 
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    arg->cmd = status;
}

static void cublas_dgemm(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    void *buf = NULL;
    int int_size = sizeof(int);
    int idx = 0;
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m, n, k;
    int lda, ldb, ldc;
    double *A, *B, *C;
    double alpha; /* host or device pointer */
    double beta; /* host or device pointer */

    func();
    // device address
    if( (A = (double*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if( (B = (double*)map_device_addr_by_vaddr(arg->src2, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src2);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if( (C = (double*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    memcpy(&handle, buf+idx, sizeof(cublasHandle_t));
    idx += sizeof(cublasHandle_t);
    memcpy(&transa, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&transb, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&m, buf+idx, int_size);
    idx += int_size;
    memcpy(&n, buf+idx, int_size);
    idx += int_size;
    memcpy(&k, buf+idx, int_size);
    idx += int_size;
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldb, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldc, buf+idx, int_size);
    idx += int_size;
    memcpy(&alpha, buf+idx, sizeof(double));
    idx += sizeof(double);
    memcpy(&beta, buf+idx, sizeof(double));
    cublasCheck(status = cublasDgemm_v2(handle, transa, transb,
                                        m, n, k,
                                        &alpha, A, lda,
                                        B, ldb,
                                        &beta, C, ldc));
    debug("handle=%lx, transa=0x%lx, transb=0x%lx, m=%d, n=%d, k=%d,"
            " alpha=%g, A=%p, lda=%d,"
            " B=%p, ldb=%d, beta=%g, C=%p, ldc=%d\n",
            (uint64_t)handle, (uint64_t)transa, (uint64_t)transb, 
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    arg->cmd = status;
}

static void cublas_set_stream(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    cublasHandle_t handle;
    int pos = (int)arg->dst;
    cudaStream_t stream = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    debug("stream pos = 0x%x\n", pos);
    if (pos==0) {
        stream=0;
    } else if (!__get_bit(&ctx->cudaStreamBitmap, pos-1)) {
        stream = ctx->cudaStream[pos-1];
    } else {
        error("No such stream, pos=0x%x\n", pos);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    handle = (cublasHandle_t)arg->src;
    debug("handle 0x%lx, stream %lx\n", (uint64_t)handle, (uint64_t)stream);
    cublasCheck(status = cublasSetStream_v2(handle, stream));
    arg->cmd = status;
}

static void cublas_get_stream(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    cublasHandle_t handle;
    int pos = 0;
    cudaStream_t stream = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    handle = (cublasHandle_t)arg->src;
    cublasCheck(status = cublasGetStream_v2(handle, &stream));
    arg->cmd = status;
    debug("handle 0x%lx, stream %lx\n", (uint64_t)handle, (uint64_t)stream);
    if (stream == NULL)
        pos = 0;
    else {
        for(int i=0; i<sizeof(BITS_PER_WORD); i++) {
            if(!__get_bit(&ctx->cudaStreamBitmap, i)) {
                pos = i+1;
                break;
            }
        }
        if(pos==0) {
            error("No such stream\n");
            arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
            return;
        }
    }
    debug("stream pos = 0x%x\n", pos);
}

static void cublas_sasum(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx;
    float *x;
    float result; /* host or device pointer */

    func();
    // device address
    if( (x = (float*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_ALLOC_FAILED;
        return;
    }
    handle  = (cublasHandle_t)arg->dst;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    cublasCheck(status = cublasSasum_v2(handle, n, x, incx, &result));
    debug("handle=%lx, n=%d, x=%p, incx=%d, result=%g\n",
            (uint64_t)handle, n, x, incx, result);
    cpu_physical_memory_write((hwaddr)arg->param2, &result, arg->paramSize);
    arg->cmd = status;
}

static void cublas_dasum(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx;
    double *x;
    double result; /* host or device pointer */

    func();
    // device address
    if( (x = (double*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_ALLOC_FAILED;
        return;
    }
    handle  = (cublasHandle_t)arg->dst;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    cublasCheck(status = cublasDasum_v2(handle, n, x, incx, &result));
    debug("handle=%lx, n=%d, x=%p, incx=%d, result=%g\n",
            (uint64_t)handle, n, x, incx, result);
    cpu_physical_memory_write((hwaddr)arg->param2, &result, arg->paramSize);
    arg->cmd = status;
}

static void cublas_scopy(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    float *x, *y;

    func();
    // device address
    if( (x = (float*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if( (y = (float*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    handle  = (cublasHandle_t)arg->src2;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    cublasCheck(status = cublasScopy_v2 (handle, n, x, incx, y, incy));
    debug("handle=%lx, n=%d, x=%p, incx=%d, y=%p, incy=%d\n",
            (uint64_t)handle, n, x, incx, y, incy);
    arg->cmd = status;
}

static void cublas_dcopy(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    double *x, *y;

    func();
    // device address
    if( (x = (double*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if( (y = (double*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    handle  = (cublasHandle_t)arg->src2;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    cublasCheck(status = cublasDcopy_v2 (handle, n, x, incx, y, incy));
    debug("handle=%lx, n=%d, x=%p, incx=%d, y=%p, incy=%d\n",
            (uint64_t)handle, n, x, incx, y, incy);
    arg->cmd = status;
}

static void cublas_sdot(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    float *x, *y;
    float result;

    func();
    // device address
    if( (x = (float*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if( (y = (float*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    handle  = (cublasHandle_t)arg->src2;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    cublasCheck(status = cublasSdot_v2(handle, n, x, incx, y, incy, &result));
    debug("handle=%lx, n=%d, x=%p, incx=%d, y=%p, incy=%d, result=%g\n",
            (uint64_t)handle, n, x, incx, y, incy, result);
    cpu_physical_memory_write((hwaddr)arg->param2, &result, arg->paramSize);
    arg->cmd = status;
}

static void cublas_ddot(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    double *x, *y;
    double result;

    func();
    // device address
    if( (x = (double*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if( (y = (double*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    handle  = (cublasHandle_t)arg->src2;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    cublasCheck(status = cublasDdot_v2(handle, n, x, incx, y, incy, &result));
    debug("handle=%lx, n=%d, x=%p, incx=%d, y=%p, incy=%d, result=%g\n",
            (uint64_t)handle, n, x, incx, y, incy, result);
    cpu_physical_memory_write((hwaddr)arg->param2, &result, arg->paramSize);
    arg->cmd = status;
}

static void cublas_saxpy(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    float *x, *y;
    float alpha; /* host or device pointer */
    uint8_t *buf = NULL;
    int idx = 0;

    func();
    // device address
    if( (x = (float*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if( (y = (float*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    handle  = (cublasHandle_t)arg->src2;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    memcpy(&alpha, buf+idx, sizeof(float));
    idx += sizeof(float);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasSaxpy_v2 (handle, n, &alpha, x, incx, y, incy));
    debug("handle=%lx, n=%d, alpha=%g, x=%p, incx=%d, y=%p, incy=%d\n",
            (uint64_t)handle, n, alpha, x, incx, y, incy);
    arg->cmd = status;
}

static void cublas_daxpy(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    double *x, *y;
    double alpha; /* host or device pointer */
    uint8_t *buf = NULL;
    int idx = 0;

    func();
    // device address
    if( (x = (double*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if( (y = (double*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    handle  = (cublasHandle_t)arg->src2;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    memcpy(&alpha, buf+idx, sizeof(double));
    idx += sizeof(double);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasDaxpy_v2 (handle, n, &alpha, x, incx, y, incy));
        debug("handle=%lx, n=%d, alpha=%g, x=%p, incx=%d, y=%p, incy=%d\n",
            (uint64_t)handle, n, alpha, x, incx, y, incy);
    arg->cmd = status;
}

static void cublas_sscal(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx;
    float *x;
    float alpha;
    uint8_t *buf = NULL;
    int idx = 0;

    func();
    // device address
    if( (x = (float*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    handle  = (cublasHandle_t)arg->src2;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    memcpy(&alpha, buf+idx, sizeof(float));
    idx += sizeof(float);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasSscal_v2(handle, n, &alpha, x, incx));
    debug("handle=%lx, n=%d, alpha=%g, x=%p, incx=%d\n",
            (uint64_t)handle, n, alpha, x, incx);
    arg->cmd = status;
}

static void cublas_dscal(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx;
    double *x;
    double alpha;
    uint8_t *buf = NULL;
    int idx = 0;

    func();
    // device address
    if( (x = (double*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_NOT_INITIALIZED;
        return;
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    handle  = (cublasHandle_t)arg->src2;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    memcpy(&alpha, buf+idx, sizeof(double));
    idx += sizeof(double);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasDscal_v2(handle, n, &alpha, x, incx));
    debug("handle=%lx, n=%d, alpha=%g, x=%p, incx=%d\n",
            (uint64_t)handle, n, alpha, x, incx);
    arg->cmd = status;
}

static void cublas_sgemv(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    uint8_t *buf = NULL;
    int int_size = sizeof(int);
    int idx = 0;
    cublasHandle_t handle;
    cublasOperation_t trans;
    int m, n;
    int lda;
    int incx, incy;
    float *x, *y, *A;
    float alpha, beta; /* host or device pointer */

    func();
    // device address
    if( (A = (float*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if( (x = (float*)map_device_addr_by_vaddr(arg->src2, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src2);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if( (y = (float*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    n       = (int)arg->srcSize2;
    m       = (int)arg->dstSize;
    memcpy(&handle, buf+idx, sizeof(cublasHandle_t));
    idx += sizeof(cublasHandle_t);
    memcpy(&trans, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&incx, buf+idx, int_size);
    idx += int_size;
    memcpy(&incy, buf+idx, int_size);
    idx += int_size;
    memcpy(&alpha, buf+idx, sizeof(float));
    idx += sizeof(float);
    memcpy(&beta, buf+idx, sizeof(float));
    idx += sizeof(float);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasSgemv_v2(handle, trans, m, n, 
                                        &alpha, A, lda, x, incx,
                                        &beta, y, incy));
    debug("handle=%lx, trans=0x%lx, m=%d, n=%d, alpha=%g, A=%p, lda=%d,"
            " x=%p, incx=%d, beta=%g, y=%p, incy=%d\n",
            (uint64_t)handle, (uint64_t)trans, m, n, 
            alpha, A, lda, x, incx, beta, y, incy);
    arg->cmd = status;
}

static void cublas_dgemv(VirtIOArg *arg, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    uint8_t *buf = NULL;
    int int_size = sizeof(int);
    int idx = 0;
    cublasHandle_t handle;
    cublasOperation_t trans;
    int m, n;
    int lda;
    int incx, incy;
    double *x, *y, *A;
    double alpha, beta; /* host or device pointer */

    func();
    // device address
    if( (A = (double*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if( (x = (double*)map_device_addr_by_vaddr(arg->src2, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->src2);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if( (y = (double*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CUBLAS_STATUS_EXECUTION_FAILED;
        return;
    }
    n       = (int)arg->srcSize2;
    m       = (int)arg->dstSize;
    memcpy(&handle, buf+idx, sizeof(cublasHandle_t));
    idx += sizeof(cublasHandle_t);
    memcpy(&trans, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&incx, buf+idx, int_size);
    idx += int_size;
    memcpy(&incy, buf+idx, int_size);
    idx += int_size;
    memcpy(&alpha, buf+idx, sizeof(double));
    idx += sizeof(double);
    memcpy(&beta, buf+idx, sizeof(double));
    idx += sizeof(double);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasDgemv_v2(handle, trans, m, n, 
                                        &alpha, A, lda, x, incx,
                                        &beta, y, incy));
    debug("handle=%lx, trans=0x%lx, m=%d, n=%d, alpha=%g, A=%p, lda=%d,"
            " x=%p, incx=%d, beta=%g, y=%p, incy=%d\n",
            (uint64_t)handle, (uint64_t)trans, m, n, 
            alpha, A, lda, x, incx, beta, y, incy);
    arg->cmd = status;
}

static void curand_create_generator(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    curandRngType_t rng_type = (curandRngType_t)arg->dst;
    func();
    curandCheck(status = curandCreateGenerator(&generator, rng_type));
    arg->flag = (uint64_t)generator;
    arg->cmd = status;
    debug("curand generator 0x%lx\n", (uint64_t)generator);
}

static void curand_create_generator_host(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    curandRngType_t rng_type = (curandRngType_t)arg->dst;
    func();
    curandCheck(status = curandCreateGeneratorHost(&generator, rng_type));
    arg->flag = (uint64_t)generator;
    arg->cmd = status;
    debug("curand generator 0x%lx\n", (uint64_t)generator);
}

static void curand_generate(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    unsigned int *ptr;
    size_t n;

    func();
    // device address
    if (arg->flag == 0) {
        if( (ptr = (unsigned int *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (unsigned int *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address %p.\n", (void *)arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if( (ptr = (unsigned int *)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    generator   = (curandGenerator_t)arg->src;
    n           = (size_t)arg->param;
    curandCheck(status=curandGenerate(generator, ptr, n));
    debug("generator=%lx, ptr=%p, n=%ld\n", (uint64_t)generator, ptr, n);
    arg->cmd = status;
}

static void curand_generate_normal(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    float *ptr;
    size_t n;
    float mean, stddev;
    uint8_t *buf = NULL;
    int idx = 0;

    func();
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such param physical address 0x%lx.\n", arg->param);
        arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
        return;
    }
    // device address
    if (arg->flag == 0) {
        if( (ptr = (float *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (float *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address %p.\n", (void *)arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if( (ptr = (float*)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    generator   = (curandGenerator_t)arg->src;
    n           = (size_t)arg->src2;
    memcpy(&mean, buf+idx, sizeof(float));
    idx += sizeof(float);
    memcpy(&stddev, buf+idx, sizeof(float));
    idx += sizeof(float);
    assert(idx == arg->paramSize);
    curandCheck(status=curandGenerateNormal(generator, ptr, n, mean, stddev));
    debug("generator=%lx, ptr=%p, n=%ld, mean=%g, stddev=%g\n",
            (uint64_t)generator, ptr, n, mean, stddev);
    arg->cmd = status;
}

static void curand_generate_normal_double(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    double *ptr;
    size_t n;
    double mean, stddev;
    uint8_t *buf = NULL;
    int idx = 0;

    func();
    if (arg->flag == 0) {
        if( (ptr = (double *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (double *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address %p.\n", (void *)arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if((ptr = (double *)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
        return;
    }
    generator   = (curandGenerator_t)arg->src;
    n           = (size_t)arg->src2;
    memcpy(&mean, buf+idx, sizeof(double));
    idx += sizeof(double);
    memcpy(&stddev, buf+idx, sizeof(double));
    idx += sizeof(double);
    assert(idx == arg->paramSize);
    curandCheck(status=curandGenerateNormalDouble(generator, ptr, n, mean, stddev));
    debug("generator=%lx, ptr=%p, n=%ld, mean=%g, stddev=%g\n",
            (uint64_t)generator, ptr, n, mean, stddev);
    arg->cmd = status;
}

static void curand_generate_uniform(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    float *ptr;
    size_t num;
    func();
    generator   = (curandGenerator_t)arg->src;
    num         = (size_t)arg->param;
    // device address
    if (arg->flag == 0) {
        if( (ptr = (float *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (float *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address 0x%lx.\n", arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if( (ptr = (float*)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    curandCheck(status = curandGenerateUniform(generator, ptr, num));
    arg->cmd = status;
    debug("curand generator 0x%lx, ptr=%p, num=0x%lx\n", 
            (uint64_t)generator, ptr, num);
}

static void curand_generate_uniform_double(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    double *ptr;
    size_t num;
    func();
    generator   = (curandGenerator_t)arg->src;
    num         = (size_t)arg->param;
    // device address
    if (arg->flag == 0) {
        if( (ptr = (double *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (double *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address 0x%lx.\n", arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if((ptr = (double *)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    curandCheck(status = curandGenerateUniformDouble(generator, ptr, num));
    arg->cmd = status;
    debug("curand generator 0x%lx, ptr=%p, num=0x%lx\n", 
            (uint64_t)generator, ptr, num);
}

static void curand_destroy_generator(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    func();
    generator   = (curandGenerator_t)arg->src;
    curandCheck(status = curandDestroyGenerator(generator));
    arg->cmd = status;
    debug("curand destroy generator 0x%lx\n", (uint64_t)generator);
}

static void curand_set_generator_offset(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    unsigned long long offset;
    func();
    generator   = (curandGenerator_t)arg->src;
    offset      = (unsigned long long)arg->param;
    curandCheck(status = curandSetGeneratorOffset(generator, offset));
    arg->cmd = status;
    debug("curand set offset generator 0x%lx, offset 0x%llx\n", 
        (uint64_t)generator, offset);
}

static void curand_set_pseudorandom_seed(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    unsigned long long seed;
    func();
    generator   = (curandGenerator_t)arg->src;
    seed      = (unsigned long long)arg->param;
    curandCheck(status = curandSetPseudoRandomGeneratorSeed(generator, seed));
    arg->cmd = status;
    debug("curand set seed generator 0x%lx, seed 0x%llx\n", 
        (uint64_t)generator, seed);
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
    VirtIOArg *msg = NULL;
    debug("port->id=%d, len=%ld\n", port->id, len);
    /*if (len != sizeof(VirtIOArg)) {
        error("buf len should be %lu, not %ld\n", sizeof(VirtIOArg), len);
        return 0;
    }*/
    msg = (VirtIOArg *)malloc(len);
    memcpy((void *)msg, (void *)buf, len);
    chan_send(port->thread_context->worker_queue, (void *)msg);
    return 0;
}


static void unload_module(ThreadContext *tctx)
{
    int i=0, idx=0;
    CudaContext *ctx = NULL;
    CudaModule *mod = NULL;

    func();
    for (idx = 0; idx < tctx->deviceCount; idx++) {
        if (idx==0 || tctx->deviceBitmap & 1<<idx) {
            ctx = &tctx->contexts[idx];
            for (i=0; i < ctx->moduleCount; i++) {
                mod = &ctx->modules[i];
                // debug("Unload module 0x%lx\n", mod->handle);
                for(int j=0; j<mod->cudaKernelsCount; j++) {
                    free(mod->cudaKernels[j].func_name);
                }
                for(int j=0; j<mod->cudaVarsCount; j++) {
                    free(mod->cudaVars[j].addr_name);
                }
                if (ctx->initialized)
                    cuErrorExit(cuModuleUnload(mod->module));
                free(mod->fatbin);
                memset(mod, 0, sizeof(CudaModule));
            }
            ctx->moduleCount = 0;
            if(ctx->initialized) {
                cuErrorExit(cuDevicePrimaryCtxRelease(ctx->dev));
                deinit_primary_context(ctx);
            }
        }
    }
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
    if(guest_connected) {
        gettimeofday(&port->start_time, NULL);
    } else {
        ThreadContext *tctx = port->thread_context;
        unload_module(tctx);
        if (tctx->deviceBitmap != 0) {
            tctx->deviceCount   = total_device;
            tctx->cur_dev       = DEFAULT_DEVICE;
            memset(&tctx->deviceBitmap, 0, sizeof(tctx->deviceBitmap));
        }
        gettimeofday(&port->end_time, NULL);
        // double time_spent = (double)(port->end_time.tv_usec - port->start_time.tv_usec)/1000000 +
        //             (double)(port->end_time.tv_sec - port->start_time.tv_sec);
        // printf("port %d spends %f seconds\n", port->id, time_spent);
    }
}

/* 
 * Enable/disable backend for virtio serial port
 * default enable is whether vm is running.
 * When vm is running, enable backend, otherwise disable backend.
 */
static void virtconsole_enable_backend(VirtIOSerialPort *port, bool enable)
{
    func();
    debug("port id=%d, enable=%d\n", port->id, enable);

    if(!enable && global_deinitialized) {
        if (!total_port)
            return;
        /*
        * kill tid thread
        */
        debug("Closing thread %d !\n", port->id);
        chan_close(port->thread_context->worker_queue);
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

/*
* worker process of thread
*/
static void *worker_processor(void *arg)
{
    VirtIOArg *msg = (VirtIOArg *)malloc(sizeof(VirtIOArg));;
    int ret=0;
    VirtIOSerialPort *port = (VirtIOSerialPort*)arg;
    ThreadContext *tctx = port->thread_context;
    int port_id = port->id;
    int tid = port_id;
    debug("port id = %d\n", port_id);
    while (chan_recv(tctx->worker_queue, (void**)&msg) == 0)
    {
        switch(msg->cmd) {
            case VIRTIO_CUDA_HELLO:
                cuda_gpa_to_hva(msg);
                break;
            case VIRTIO_CUDA_REGISTERFATBINARY:
                cuda_register_fatbinary(msg, tctx);
                break;
            case VIRTIO_CUDA_UNREGISTERFATBINARY:
                cuda_unregister_fatbinary(msg, tctx);
                break;
            case VIRTIO_CUDA_REGISTERFUNCTION:
                cuda_register_function(msg, tctx);
                break;
            case VIRTIO_CUDA_REGISTERVAR:
                cuda_register_var(msg, tctx);
                break;
            case VIRTIO_CUDA_LAUNCH:
                cuda_launch(msg, port);
                break;
            case VIRTIO_CUDA_MALLOC:
                cuda_malloc(msg, tctx);
                break;
            case VIRTIO_CUDA_HOSTREGISTER:
                cuda_host_register(msg, port);
                break;
            case VIRTIO_CUDA_HOSTUNREGISTER:
                cuda_host_unregister(msg, port);
                break;
            case VIRTIO_CUDA_MEMCPY:
                cuda_memcpy(msg, tctx);
                break;
            case VIRTIO_CUDA_FREE:
                cuda_free(msg, tctx);
                break;
            case VIRTIO_CUDA_GETDEVICE:
                cuda_get_device(msg, tid);
                break;
            case VIRTIO_CUDA_GETDEVICEPROPERTIES:
                cuda_get_device_properties(msg);
                break;
            case VIRTIO_CUDA_MMAPCTL:
                mmapctl(msg, port);
                break;
            case VIRTIO_CUDA_MUNMAPCTL:
                munmapctl(msg, port);
                break;
            case VIRTIO_CUDA_GETDEVICECOUNT:
                cuda_get_device_count(msg);
                break;
            case VIRTIO_CUDA_SETDEVICE:
                cuda_set_device(msg, tctx);
                break;
            case VIRTIO_CUDA_SETDEVICEFLAGS:
                cuda_set_device_flags(msg, tctx);
                break;
            case VIRTIO_CUDA_DEVICERESET:
                cuda_device_reset(msg, tctx);
                break;
            case VIRTIO_CUDA_STREAMCREATE:
                cuda_stream_create(msg, tctx);
                break;
            case VIRTIO_CUDA_STREAMCREATEWITHFLAGS:
                cuda_stream_create_with_flags(msg, tctx);
                break;
            case VIRTIO_CUDA_STREAMDESTROY:
                cuda_stream_destroy(msg, tctx);
                break;
            case VIRTIO_CUDA_STREAMWAITEVENT:
                cuda_stream_wait_event(msg, tctx);
                break;
            case VIRTIO_CUDA_STREAMSYNCHRONIZE:
                cuda_stream_synchronize(msg, tctx);
                break;
            case VIRTIO_CUDA_EVENTCREATE:
                cuda_event_create(msg, tctx);
                break;
            case VIRTIO_CUDA_EVENTCREATEWITHFLAGS:
                cuda_event_create_with_flags(msg, tctx);
                break;
            case VIRTIO_CUDA_EVENTDESTROY:
                cuda_event_destroy(msg, tctx);
                break;
            case VIRTIO_CUDA_EVENTRECORD:
                cuda_event_record(msg, tctx);
                break;
            case VIRTIO_CUDA_EVENTQUERY:
                cuda_event_query(msg, tctx);
                break;
            case VIRTIO_CUDA_EVENTSYNCHRONIZE:
                cuda_event_synchronize(msg, tctx);
                break;
            case VIRTIO_CUDA_EVENTELAPSEDTIME:
                cuda_event_elapsedtime(msg, tctx);
                break;
            case VIRTIO_CUDA_THREADSYNCHRONIZE:
                cuda_thread_synchronize(msg, tctx);
                break;
            case VIRTIO_CUDA_GETLASTERROR:
                cuda_get_last_error(msg, tctx);
                break;
            case VIRTIO_CUDA_PEEKATLASTERROR:
                cuda_peek_at_last_error(msg, tctx);
                break;
            case VIRTIO_CUDA_MEMCPY_ASYNC:
                cuda_memcpy_async(msg, tctx);
                break;
            case VIRTIO_CUDA_MEMSET:
                cuda_memset(msg, tctx);
                break;
            case VIRTIO_CUDA_DEVICESYNCHRONIZE:
                cuda_device_synchronize(msg, tctx);
                break;
            case VIRTIO_CUDA_MEMGETINFO:
                cuda_mem_get_info(msg, tctx);
                break;
            case VIRTIO_CUDA_MEMCPYTOSYMBOL:
                cuda_memcpy_to_symbol(msg, tctx);
                break;
            case VIRTIO_CUDA_MEMCPYFROMSYMBOL:
                cuda_memcpy_from_symbol(msg, tctx);
                break;
            case VIRTIO_CUBLAS_CREATE:
                cublas_create(msg, tctx);
                break;
            case VIRTIO_CUBLAS_DESTROY:
                cublas_destroy(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SETVECTOR:
                cublas_set_vector(msg, tctx);
                break;
            case VIRTIO_CUBLAS_GETVECTOR:
                cublas_get_vector(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SETMATRIX:
                cublas_set_matrix(msg, tctx);
                break;
            case VIRTIO_CUBLAS_GETMATRIX:
                cublas_get_matrix(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SGEMM:
                cublas_sgemm(msg, tctx);
                break;
            case VIRTIO_CUBLAS_DGEMM:
                cublas_dgemm(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SETSTREAM:
                cublas_set_stream(msg, tctx);
                break;
            case VIRTIO_CUBLAS_GETSTREAM:
                cublas_get_stream(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SASUM:
                cublas_sasum(msg, tctx);
                break;
            case VIRTIO_CUBLAS_DASUM:
                cublas_dasum(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SAXPY:
                cublas_saxpy(msg, tctx);
                break;
            case VIRTIO_CUBLAS_DAXPY:
                cublas_daxpy(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SCOPY:
                cublas_scopy(msg, tctx);
                break;
            case VIRTIO_CUBLAS_DCOPY:
                cublas_dcopy(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SGEMV:
                cublas_sgemv(msg, tctx);
                break;
            case VIRTIO_CUBLAS_DGEMV:
                cublas_dgemv(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SDOT:
                cublas_sdot(msg, tctx);
                break;
            case VIRTIO_CUBLAS_DDOT:
                cublas_ddot(msg, tctx);
                break;
            case VIRTIO_CUBLAS_SSCAL:
                cublas_sscal(msg, tctx);
                break;
            case VIRTIO_CUBLAS_DSCAL:
                cublas_dscal(msg, tctx);
                break;
            case VIRTIO_CURAND_CREATEGENERATOR:
                curand_create_generator(msg, tctx);
                break;
            case VIRTIO_CURAND_CREATEGENERATORHOST:
                curand_create_generator_host(msg, tctx);
                break;
            case VIRTIO_CURAND_GENERATE:
                curand_generate(msg, tctx);
                break;
            case VIRTIO_CURAND_GENERATENORMAL:
                curand_generate_normal(msg, tctx);
                break;
            case VIRTIO_CURAND_GENERATENORMALDOUBLE:
                curand_generate_normal_double(msg, tctx);
                break;
            case VIRTIO_CURAND_GENERATEUNIFORM:
                curand_generate_uniform(msg, tctx);
                break;
            case VIRTIO_CURAND_GENERATEUNIFORMDOUBLE:
                curand_generate_uniform_double(msg, tctx);
                break;
            case VIRTIO_CURAND_DESTROYGENERATOR:
                curand_destroy_generator(msg, tctx);
                break;
            case VIRTIO_CURAND_SETGENERATOROFFSET:
                curand_set_generator_offset(msg, tctx);
                break;
            case VIRTIO_CURAND_SETPSEUDORANDOMSEED:
                curand_set_pseudorandom_seed(msg, tctx);
                break;
            default:
                error("[+] header.cmd=%u, nr= %u \n",
                      msg->cmd, _IOC_NR(msg->cmd));
            return NULL;
        }
        debug("writing back tid %d\n", msg->tid);
        ret = virtio_serial_write(port, (const uint8_t *)msg, 
                                  sizeof(VirtIOArg));
        if (ret < sizeof(VirtIOArg)) {
            error("write error.\n");
            virtio_serial_throttle_port(port, true);
        }
        debug("[+] WRITE BACK\n");
    }
    free(msg);
    debug("Shutting the thread %d\n", tid);
    return NULL;
}

static void spawn_thread_by_port(VirtIOSerialPort *port)
{
    char thread_name[16];
    int port_id = port->id;
    int thread_id = port_id%total_port;
    debug("Starting thread %d computing workloads and queue!\n",thread_id);
    // spawn_thread_by_port(port);
    sprintf(thread_name, "thread_%d", thread_id);
    qemu_thread_create(&port->thread_context->worker_thread, thread_name, 
            worker_processor, port, QEMU_THREAD_JOINABLE);
}

/* 
* Guest is now ready to accept data (virtqueues set up). 
* When the guest has asked us for this information it means
* the guest is all setup and has its virtqueues
* initialized. If some app is interested in knowing about
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
    spawn_thread_by_port(port);
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
    total_port = 0;
    qemu_mutex_init(&total_port_mutex);
    cuError( cuInit(0));
    if(!vser->gcount) {
        cuError(cuDeviceGetCount(&total_device));
    }
    else {
        total_device = vser->gcount;
    }
    debug("vser->gcount(or total_device)=%d\n", total_device);
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
    ThreadContext *tctx = NULL;
    port->thread_context = malloc(sizeof(ThreadContext));
    CudaContext *ctx = NULL;
    tctx = port->thread_context;
    tctx->deviceCount = total_device;
    tctx->cur_dev = DEFAULT_DEVICE;
    memset(&tctx->deviceBitmap, 0, sizeof(tctx->deviceBitmap));
    tctx->contexts = malloc(sizeof(CudaContext) * tctx->deviceCount);
    for (int i = 0; i < tctx->deviceCount; i++)
    {
        ctx = &tctx->contexts[i];
        memset(&ctx->cudaStreamBitmap,  ~0, sizeof(ctx->cudaStreamBitmap));
        memset(&ctx->cudaEventBitmap,   ~0, sizeof(ctx->cudaEventBitmap));
        memset(ctx->cudaEvent,           0,  sizeof(cudaEvent_t) *CudaEventMaxNum);
        memset(ctx->cudaStream,          0,  sizeof(cudaStream_t)*CudaStreamMaxNum);
        INIT_LIST_HEAD(&ctx->vol);
        INIT_LIST_HEAD(&ctx->host_vol);
        ctx->moduleCount = 0;
        ctx->initialized = 0;
        ctx->tctx        = tctx;
    }
    tctx->worker_queue  = chan_init(100);
}

static void deinit_port(VirtIOSerialPort *port)
{
    ThreadContext *tctx = port->thread_context;
    /*delete elements in struct list_head*/
    /*
    */
    free(tctx->contexts);
    tctx->contexts = NULL;
    chan_dispose(tctx->worker_queue);
    debug("Ending thread %d computing workloads and queue!\n", port->id);
    qemu_thread_join(&tctx->worker_thread);
    free(tctx);
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
