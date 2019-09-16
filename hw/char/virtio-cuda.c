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
static uint64_t map_addr_by_vaddr(uint64_t vaddr, struct list_head *header);

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
    if (! (tctx->deviceBitmap & 1<<DEFAULT_DEVICE)) {
        debug("Initialize DEFAULT_DEVICE\n");
        tctx->deviceBitmap |= 1<<DEFAULT_DEVICE;
        cuErrorExit(cuDeviceGet(&ctx->dev, DEFAULT_DEVICE));
        // cuErrorExit(cuDevicePrimaryCtxRetain(&ctx->context, ctx->dev));
        cuErrorExit(cuCtxCreate(&ctx->context, 0, ctx->dev));
        ctx->moduleCount = 0;
        memset(ctx->modules, 0, sizeof(ctx->modules));
        cuErrorExit(cuCtxSetCurrent(ctx->context));
    }
    fat_bin = malloc(fatbin_size);
    cpu_physical_memory_read((hwaddr)arg->dst, fat_bin, fatbin_size);

    m_idx = ctx->moduleCount++;
    debug("fat_bin gva is 0x%lx\n", (uint64_t)arg->src);
    debug("fat_bin gpa is 0x%lx\n", (uint64_t)arg->dst);
    debug("fat_bin size is 0x%x\n", fatbin_size);
    debug("fat_bin hva is 0x%lx\n", (uint64_t)fat_bin);
    debug("fat_bin hva is at %p\n", fat_bin);
    debug("module = %d\n", m_idx);
    ctx->modules[m_idx].handle              = (size_t)arg->src;
    ctx->modules[m_idx].fatbin_size         = fatbin_size;
    ctx->modules[m_idx].cudaKernelsCount    = 0;
    ctx->modules[m_idx].cudaVarsCount       = 0;
    ctx->modules[m_idx].fatbin              = fat_bin;
    cuErrorExit(cuModuleLoadData(&ctx->modules[m_idx].module, fat_bin));
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
    VOL *vol, *vol2;
    HVOL *hvol, *hvol2;
    int i=0, idx=0;
    CudaContext *ctx = NULL;
    CudaModule *mod = NULL;
    func();

    for (idx = 0; idx < tctx->deviceCount; idx++) {
        if (tctx->deviceBitmap & 1<<idx) {
            ctx = &tctx->contexts[idx];
            for (i=0; i < ctx->moduleCount; i++) {
                debug("Unload module\n");
                mod = &ctx->modules[i];
                for(int j=0; j<mod->cudaKernelsCount; j++) {
                    free(mod->cudaKernels[j].func_name);
                }
                for(int j=0; j<mod->cudaVarsCount; j++) {
                    free(mod->cudaVars[j].addr_name);
                }
                cuErrorExit(cuModuleUnload(mod->module));
                free(mod->fatbin);
                memset(mod, 0, sizeof(CudaModule));
            }
            ctx->moduleCount=0;
            memset(&ctx->cudaStreamBitmap, ~0, sizeof(ctx->cudaStreamBitmap));
            memset(ctx->cudaEventBitmap, ~0, sizeof(ctx->cudaEventBitmap));
            memset(ctx->cudaStream, 0, sizeof(cudaStream_t)*CudaStreamMaxNum);
            memset(ctx->cudaEvent, 0, sizeof(cudaEvent_t)*CudaEventMaxNum);
            // free struct list
            list_for_each_entry_safe(vol, vol2, &ctx->vol, list) {
                list_del(&vol->list);
            }
            list_for_each_entry_safe(hvol, hvol2, &ctx->host_vol, list) {
                list_del(&hvol->list);
            }
            // cuErrorExit(cuDevicePrimaryCtxRelease(ctx->dev));
            cuErrorExit(cuCtxDestroy(ctx->context));
        }
    }
    memset(&tctx->deviceBitmap, 0, sizeof(tctx->deviceBitmap));
    tctx->cur_dev = DEFAULT_DEVICE;
    arg->cmd = cudaSuccess;
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
    cuErrorExit(cuModuleGetFunction(&kernel->kernel_func, cuda_module->module, kernel->func_name));
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
    CudaMemVar   *var;
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

    cuErrorExit(cuModuleGetGlobal(&var->device_ptr, &var->mem_size, cuda_module->module, var->addr_name));
    arg->cmd = cudaSuccess;
}

static void cuda_setup_argument(VirtIOArg *arg)
{
    func();
}

static void cuda_launch(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    uint32_t para_num=0, para_idx=0;
    uint32_t para_size=0, conf_size=0;
    cudaStream_t stream_kernel = 0;
    int i = 0;
    int j = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int m_num       = ctx->moduleCount;
    CudaModule *cuda_module = NULL;
    CudaKernel *kernel  = NULL;
    size_t func_handle  = 0;
    void **para_buf     = NULL;
    uint64_t addr       = 0;

    func();
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
    para_num = *((uint32_t*)para);
    debug(" para_num = %u\n", para_num);
    para_buf = malloc(para_num * sizeof(void*));
    para_idx = sizeof(uint32_t);
    for(i=0; i<para_num; i++) {
        para_buf[i] = &para[para_idx + sizeof(uint32_t)];
        addr = map_addr_by_vaddr( *(uint64_t*)para_buf[i], &ctx->vol);
        if(addr!=0) {
            debug("Found 0x%lx\n", addr);
            memcpy(&para_buf[i], &addr, sizeof(uint64_t));
        }
        debug("arg %d = 0x%llx , size=%u byte\n", i, 
              *(unsigned long long*)para_buf[i], 
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
    cudaError(err = cudaLaunchKernel( kernel->kernel_func, 
                                    conf->gridDim, conf->blockDim,
                                    para_buf, conf->sharedMem, 
                                    stream_kernel));
    arg->cmd = err;
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

// static HVOL *find_hvol_by_vaddr(uint64_t vaddr, struct list_head *header)
// {
//     HVOL *hvol;
//     list_for_each_entry(hvol, header, list) {
//         if(hvol->virtual_addr <= vaddr && vaddr < (hvol->virtual_addr+hvol->size) )
//             goto out;
//     }
//     hvol = NULL;
// out:
//     return hvol;
// }

static uint64_t map_addr_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    VOL *vol = find_vol_by_vaddr(vaddr, header);
    if(vol != NULL)
        return vol->addr + (vaddr - vol->v_addr);
    return 0;
}

static void cuda_memcpy(VirtIOArg *arg, int tid)
{
/*    cudaError_t err;
    uint32_t size;
    void *src, *dst;
    uint64_t *gpa_array=NULL;
    int      i = 0;
    uint64_t addr = 0;
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
    }*/
}

static void cuda_memcpy_async(VirtIOArg *arg, int tid)
{
/*    cudaError_t err=-1;
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
        
        if((addr=map_addr_by_vaddr(arg->dst, &cudaDevices[tid]))==0) {
            error("Failed to find dst virtual addr %p in vol\n",
                  (void *)arg->dst);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        dst = (void *)addr;
        // try to fetch dst guest physical address
        gpa_array = (uint64_t*)gpa_to_hva((hwaddr)arg->param2, blocks);
        if(!gpa_array) {
            error("No such dst physical address 0x%lx.\n", arg->param2);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        src = NULL;
        void *gsrc=NULL;
        HVOL *hvol = find_hvol_by_vaddr(arg->src, &cudaDevices[tid]);
        if (hvol) {
            int offsets = arg->src - hvol->virtual_addr;
            src = (void*)(hvol->actual_addr + offsets);
            gsrc = (void*)(hvol->native_addr + offsets);
        }
        if (src==NULL) {
            gsrc = malloc(size);
        }

        uint32_t start_offset = init_offset % KMALLOC_SIZE;
        int len = min(size, KMALLOC_SIZE - start_offset);
        cpu_physical_memory_read((hwaddr)gpa_array[0], gsrc, len);
        int rsize=size;
        int offset=len;
        rsize-=len;
        i=1;
        while(rsize) {
            len=min(rsize, KMALLOC_SIZE);
            cpu_physical_memory_read((hwaddr)gpa_array[i++],
                                     gsrc+offset, len);
            offset+=len;
            rsize-=len;
        }
        assert(i == blocks);
        
        // cuError( (err= cuMemcpyHtoDAsync((CUdeviceptr)dst, (void *)src, size, 
                                         // stream)));
        int cmd = VIRTIO_CUDA_MEMCPY_ASYNC;
        write(pfd[tid][WRITE], &cmd, 4);
        write(pfd[tid][WRITE], &arg->flag, 4);
        write(pfd[tid][WRITE], &stream, sizeof(cudaStream_t));
        write(pfd[tid][WRITE], &size, 4);
        write(pfd[tid][WRITE], &src, sizeof(void *));
        if(src==NULL)
            write(pfd[tid][WRITE], &gsrc, size);
        write(pfd[tid][WRITE], &dst, sizeof(void *));

        while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
        arg->cmd = err;
        if(err != cudaSuccess) {
            error("memcpy async HtoD error!\n");
        }
        if (src == NULL)
            free(gsrc);
    } else if (arg->flag == cudaMemcpyDeviceToHost) {
        if((src=(void*)map_addr_by_vaddr(arg->src, &cudaDevices[tid]))==0) {
            error("Failed to find virtual addr %p in vol or hvol\n",
                  (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        // try to fetch dst guest physical address
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
            int offsets = arg->dst - hvol->virtual_addr;
            dst = (void*)(hvol->actual_addr+offsets);
            gdst = (void*)(hvol->native_addr+ offsets);
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
        write(pfd[tid][WRITE], &size, 4);
        write(pfd[tid][WRITE], &src, sizeof(void *));
        write(pfd[tid][WRITE], &dst, sizeof(void *));

        while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
        if (err != cudaSuccess) {
            error("memcpy async DtoD error!\n");
            arg->cmd = err;
            return;
        }
    } else {
        error("No such memcpy direction.\n");
        arg->cmd= cudaErrorInvalidMemcpyDirection;
    }*/
}

static void cuda_memcpy_to_symbol(VirtIOArg *arg, int tid)
{
/*    cudaError_t err=-1;
    size_t size;
    void *src;
    uint64_t *gpa_array=NULL;
    uint32_t init_offset=0, blocks=0;
    size_t var_handle=0;
    size_t var_offset=0;
    int m_num       = cudaModuleNum[tid];
    int found       = 0;
    int i           = 0;
    int j           = 0;
    int var_count   = 0;
    CudaMemVar   *var;
    CudaModule      *cuda_module;
    func();
    
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=%lu, "
        "param=0x%lx, param2=0x%lx \n",
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
        arg->param, arg->param2);
    size = arg->srcSize;
    src = malloc(size);
    if (arg->flag != 0) {
        blocks = arg->flag >> 32;
        if(blocks==0) {
            error("Failed to get blocks\n");
            arg->cmd = cudaErrorInvalidValue;
            return ;
        }
        init_offset = arg->flag & 0xffffffff;
        debug("blocks=0x%x, offset=0x%x\n", blocks, init_offset);
        // try to fetch dst guest physical address
        gpa_array = (uint64_t*)gpa_to_hva((hwaddr)arg->param2, blocks);
        if(!gpa_array) {
            error("No such dst physical address 0x%lx.\n", arg->param2);
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
    } else {
        cpu_physical_memory_read((hwaddr)arg->param2, src, size);
    }

    var_handle = (size_t)arg->dst;
    for (i=0; i < m_num; i++) {
        cuda_module = &cudaModules[tid][i];
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
    var_offset = (size_t)arg->param;
    int cmd = VIRTIO_CUDA_MEMCPYTOSYMBOL;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &size, sizeof(size_t));
    write(pfd[tid][WRITE], src, size);
    write(pfd[tid][WRITE], &var_handle, sizeof(size_t));
    write(pfd[tid][WRITE], &var_offset, sizeof(size_t));
    read(cfd[tid][READ], &err, sizeof(cudaError_t));
    arg->cmd = err;
    if(err != cudaSuccess) {
        error("memcpy to symbol HtoD error!\n");
    }
    free(src);*/
}

static void cuda_memcpy_from_symbol(VirtIOArg *arg, int tid)
{
/*    cudaError_t err=-1;
    size_t  size;
    void    *dst;
    uint64_t *gpa_array=NULL;
    uint32_t init_offset=0, blocks=0;
    size_t  var_handle=0;
    size_t  var_offset=0;
    int m_num       = cudaModuleNum[tid];
    int found       = 0;
    int i           = 0;
    int j           = 0;
    int var_count   = 0;
    CudaMemVar   *var;
    CudaModule      *cuda_module;
    func();
    
    debug(  "src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=0x%lx, "
            "param=0x%lx, param2=0x%lx,\n",
            arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
            arg->param, arg->param2);
    size = arg->dstSize;
 
    var_handle = (size_t)arg->src;
    for (i=0; i < m_num; i++) {
        cuda_module = &cudaModules[tid][i];
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
    var_offset = (size_t)arg->param;

    int cmd = VIRTIO_CUDA_MEMCPYFROMSYMBOL;

    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &size, sizeof(size_t));
    write(pfd[tid][WRITE], &var_handle, sizeof(size_t));
    write(pfd[tid][WRITE], &var_offset, sizeof(size_t));
    read(cfd[tid][READ], &err, sizeof(cudaError_t));
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("memcpy from symbol DtoH error!\n");
        return;
    }
    // copy back to VM
    dst = malloc(size);
    read(cfd[tid][READ], dst, size);

    if (arg->flag != 0) {
        blocks = arg->flag >> 32;
        if(blocks==0) {
            error("Failed to get blocks\n");
            arg->cmd = cudaErrorInvalidValue;
            free(dst);
            return ;
        }
        init_offset = arg->flag & 0xffffffff;
        debug("blocks=0x%x, offset=0x%x\n", blocks, init_offset);
        // try to fetch dst guest physical address
        gpa_array = (uint64_t*)gpa_to_hva((hwaddr)arg->param2, blocks);
        if(!gpa_array) {
            error("No such dst physical address 0x%lx.\n", arg->param2);
            arg->cmd = cudaErrorInvalidValue;
            free(dst);
            return;
        }
        uint32_t start_offset = init_offset % KMALLOC_SIZE;
        int len = min(size, KMALLOC_SIZE - start_offset);
        cpu_physical_memory_write((hwaddr)gpa_array[0], dst, len);
        int rsize=size;
        uint32_t offset=len;
        rsize-=len;
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
        cpu_physical_memory_write((hwaddr)arg->param2, dst, size);
    }
    free(dst);*/
}


static void cuda_memset(VirtIOArg *arg, int tid)
{
/*    cudaError_t err=-1;
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
        error("memset memory error!\n");*/
}

static void cuda_malloc(VirtIOArg *arg, int tid)
{
/*    cudaError_t err=-1;
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
          "ret value=0x%x\n", (uint64_t)devPtr, arg->dst, size, err);*/
}

static void cuda_host_register(VirtIOArg *arg, int tid)
{
/*    cudaError_t err=-1;
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
    debug("file path = %s\n", file_path);
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
    list_add_tail(&hvol->list, &cudaDevices[tid].host_vol);*/
}

static void cuda_host_unregister(VirtIOArg *arg, int tid)
{
/*    cudaError_t err=-1;
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
    }*/

}
static void cuda_free(VirtIOArg *arg, int tid)
{
/*    cudaError_t err=-1;
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
    error("Failed to find ptr!\n");*/
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

static void cuda_set_device(VirtIOArg *arg, int tid)
{
/*    cudaError_t err = -1;
    func();
    int dev_id = (int)(arg->flag);
    if (dev_id < 0 || dev_id > total_device) {
        error("setting error device = %d\n", dev_id);
        arg->cmd = cudaErrorInvalidDevice;
        return ;
    }
    
    debug("set devices=%d\n", (int)(arg->flag));
    cudaDevices[tid].device = (CUdevice)dev_id;

    int cmd = VIRTIO_CUDA_SETDEVICE;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &dev_id, 4);
    read(cfd[tid][READ], &err, sizeof(cudaError_t));
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("set device error.\n");
        return;
    }*/
    /* clear kernel function addr in parent process, 
    because cudaSetDevice will clear all resources related with host thread.
    */
}

static void cuda_set_device_flags(VirtIOArg *arg, int tid)
{
/*    cudaError_t err = -1;
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
    }*/
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
/*    cudaError_t err = -1;
    func();
    // free memory
    VOL *vol, *vol2;
    list_for_each_entry_safe(vol, vol2, &cudaDevices[tid].vol, list) {
        // cudaError( cudaFree((void*)(vol->addr))) ;
        list_del(&vol->list);
    }

    memset(&cudaStreamBitmap[tid], ~0, sizeof(cudaStreamBitmap[tid]));
    memset(cudaStream[tid], 0, sizeof(cudaStream_t)*CudaStreamMaxNum);
    memset(cudaEventBitmap[tid], ~0, sizeof(cudaEventBitmap[tid]));
    memset(cudaEvent[tid], 0, sizeof(cudaEvent_t)*CudaEventMaxNum);
    int cmd = VIRTIO_CUDA_DEVICERESET;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("create stream error.\n");
    }
    debug("reset devices\n");*/
}

static void cuda_stream_create(VirtIOArg *arg, CudaContext *ctx)
{
    cudaError_t err = -1;
    uint32_t pos = 0;
    func();
    pos = ffsll(ctx->cudaStreamBitmap);
    if (!pos) {
        error("stream number is up to %d\n", CudaStreamMaxNum);
        return;
    }
    cudaError(err = cudaStreamCreate(&ctx->cudaStream[pos-1]));
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

static void cuda_stream_destroy(VirtIOArg *arg, CudaContext *ctx)
{
    cudaError_t err = -1;
    uint32_t pos;
    func();
    pos = arg->flag;
    if (__get_bit(&ctx->cudaStreamBitmap, pos-1)) {
        error("No such stream, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    debug("destroy stream 0x%lx\n", (uint64_t)ctx->cudaStream[pos-1]);
    cudaError((err=cudaStreamDestroy(ctx->cudaStream[pos-1]) ));
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("destroy stream error.\n");
        return;
    }
    __set_bit(&ctx->cudaStreamBitmap, pos-1);
}

static void cuda_stream_synchronize(VirtIOArg *arg, int tid)
{
 /*   cudaError_t err = -1;
    uint32_t pos;
    func();
    pos = arg->flag;
    if (__get_bit(&cudaStreamBitmap[tid], pos-1)) {
        error("No such stream, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    debug("destroy stream 0x%lx\n", (uint64_t)cudaStream[tid][pos-1]);
    // cudaError( (err=cudaStreamSynchronize(cudaStream[tid][pos-1]) ));
    int cmd = VIRTIO_CUDA_STREAMSYNCHRONIZE;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &cudaStream[tid][pos-1], sizeof(cudaStream_t));
    read(cfd[tid][READ], &err, sizeof(cudaError_t));
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("synchronize stream error.\n");
        return;
    }*/
}

static void cuda_stream_wait_event(VirtIOArg *arg, int tid)
{
    /*cudaError_t err = -1;
    uint64_t pos;
    cudaStream_t    stream = 0;
    cudaEvent_t     event = 0;
    func();
    pos = arg->src;
    if (__get_bit(&cudaStreamBitmap[tid], pos-1)) {
        error("No such stream, pos=%ld\n", pos);
        arg->cmd=cudaErrorInvalidResourceHandle;
        return;
    }
    stream = cudaStream[tid][pos-1];
    debug("stream 0x%lx\n", (uint64_t)stream);
    pos = arg->dst;
    if (__get_bit(cudaEventBitmap[tid], pos-1)) {
        error("No such event, pos=%ld\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    event = cudaEvent[tid][pos-1];
    debug("wait for event 0x%lx\n", (uint64_t)event);
    
    int cmd = VIRTIO_CUDA_STREAMWAITEVENT;
    write(pfd[tid][WRITE], &cmd, 4);
    write(pfd[tid][WRITE], &stream, sizeof(cudaStream_t));
    write(pfd[tid][WRITE], &event, sizeof(cudaEvent_t));
    read(cfd[tid][READ], &err, sizeof(cudaError_t));
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("failed to wait event for stream.\n");
    }*/
}

static void cuda_event_create(VirtIOArg *arg, int tid)
{
    /*cudaError_t err = -1;
    uint32_t pos = 0;
    func();
    for(int i=0; i<CudaEventMapMax; i++) {    
        pos = ffsll(cudaEventBitmap[tid][i]);
        if(pos) break;
    }
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
    __clear_bit(cudaEventBitmap[tid], pos-1);
    debug("create event 0x%lx, idx is %u\n",
          (uint64_t)cudaEvent[tid][pos-1], pos-1);*/
}

static void cuda_event_create_with_flags(VirtIOArg *arg, int tid)
{
    /*cudaError_t err = -1;
    uint32_t pos = 0;
    unsigned int flag=0;
    func();
    for(int i=0; i<CudaEventMapMax; i++) {
        pos = ffsll(cudaEventBitmap[tid][i]);
        if(pos) break;
    }
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
    __clear_bit(cudaEventBitmap[tid], pos-1);
    debug("create event 0x%lx with flag %u, idx is %u\n",
          (uint64_t)cudaEvent[tid][pos-1], flag, pos-1);*/
}

static void cuda_event_destroy(VirtIOArg *arg, int tid)
{
 /*   cudaError_t err = -1;
    uint32_t pos = 0;
    func();
    pos = arg->flag;
    if (__get_bit(cudaEventBitmap[tid], pos-1)) {
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
    __set_bit(cudaEventBitmap[tid], pos-1);*/
}

static void cuda_event_record(VirtIOArg *arg, int tid)
{
 /*   cudaError_t err = -1;
    uint64_t epos = 0, spos = 0;
    cudaStream_t stream;
    func();
    epos = arg->src;
    spos = arg->dst;
    debug("event pos = 0x%lx\n", epos);
    if (epos<=0 || __get_bit(cudaEventBitmap[tid], epos-1)) {
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
    }*/
}

static void cuda_event_synchronize(VirtIOArg *arg, int tid)
{
   /* cudaError_t err = -1;
    uint32_t pos = 0;
    func();
    pos = arg->flag;
    if (__get_bit(cudaEventBitmap[tid], pos-1)) {
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
    }*/
}

static void cuda_event_elapsedtime(VirtIOArg *arg, int tid)
{
  /*  cudaError_t err = -1;
    int start_pos, stop_pos;
    float time = 0;
    func();
    start_pos = arg->src;
    stop_pos = arg->dst;
    if (__get_bit(cudaEventBitmap[tid], start_pos-1)) {
        error("No such event, pos=%d\n", start_pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    if (__get_bit(cudaEventBitmap[tid], stop_pos-1)) {
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
    arg->flag = (uint64_t)time;*/
}

static void cuda_device_synchronize(VirtIOArg *arg, int tid)
{
  /*  cudaError_t err = -1;
    func();
    int cmd =VIRTIO_CUDA_DEVICESYNCHRONIZE;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;*/
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
/*    cudaError_t err = -1;
    func();
    // communicate with pipe
    int cmd =VIRTIO_CUDA_GETLASTERROR;
    write(pfd[tid][WRITE], &cmd, 4);
    while(read(cfd[tid][READ], &err, sizeof(cudaError_t))==0);
    arg->cmd = err;*/
}

static void cuda_mem_get_info(VirtIOArg *arg, int tid)
{
/*    cudaError_t err = -1;
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
    debug("free memory = %lu, total memory = %lu.\n", freeMem, totalMem);*/
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
    if (len != sizeof(VirtIOArg)) {
        error("buf len should be %lu, not %ld\n", sizeof(VirtIOArg), len);
        return 0;
    }
    msg = (VirtIOArg *)malloc(sizeof(VirtIOArg));
    memcpy((void *)msg, (void *)buf, len);
    chan_send(port->thread_context->worker_queue, (void *)msg);
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
        /*
        * kill tid thread
        */
        debug("Closing thread %d !\n",thread_id);
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

// static void init_device_module(CudaContext *ctx)
// {
//     int i=0, j=0;
//     CudaModule *module = NULL;
//     CudaKernel *kernel = NULL;
//     CudaMemVar *var = NULL;
//     debug("sub module number = %d\n", ctx->moduleCount);
//     for (i=0; i < ctx->moduleCount; i++) {
//         module = &ctx->modules[i];
//         if (!module)
//             return;
//         cuError(cuModuleLoadData(&module->module, module->fatbin));
//         debug("kernel count = %d\n", module->cudaKernelsCount);
//         for(j=0; j<module->cudaKernelsCount; j++) {
//             kernel = &module->cudaKernels[j];
//             cuError(cuModuleGetFunction(&kernel->kernel_func, module->module, kernel->func_name));
//         }
//         debug("var count = %d\n", module->cudaVarsCount);
//         for(j=0; j<module->cudaVarsCount; j++) {
//             var = &module->cudaVars[j];
//             cuError(cuModuleGetGlobal(&var->device_ptr, &var->mem_size, module->module, var->addr_name));
//         }
//     }
// }

/*
* worker process of thread
*/
static void *worker_processor(void *arg)
{
    VirtIOArg *msg = (VirtIOArg *)malloc(sizeof(VirtIOArg));;
    int ret=0;
    VirtIOSerialPort *port = (VirtIOSerialPort*)arg;
    ThreadContext *tctx = port->thread_context;
    CudaContext *cudaContexts = tctx->contexts;
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
                cuda_launch(msg, tctx);
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
                cuda_stream_create(msg, cudaContexts);
                break;
            case VIRTIO_CUDA_STREAMDESTROY:
                cuda_stream_destroy(msg, cudaContexts);
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
            case VIRTIO_CUDA_MEMCPYTOSYMBOL:
                cuda_memcpy_to_symbol(msg, tid);
                break;
            case VIRTIO_CUDA_MEMCPYFROMSYMBOL:
                cuda_memcpy_from_symbol(msg, tid);
                break;
            case VIRTIO_CUDA_STREAMWAITEVENT:
                cuda_stream_wait_event(msg, tid);
                break;
            case VIRTIO_CUDA_STREAMSYNCHRONIZE:
                cuda_stream_synchronize(msg, tid);
                break;
            default:
                error("[+] header.cmd=%u, nr= %u \n",
                      msg->cmd, _IOC_NR(msg->cmd));
            return NULL;
        }
        ret = virtio_serial_write(port, (const uint8_t *)msg, 
                                  sizeof(VirtIOArg));
        if (ret < sizeof(VirtIOArg)) {
            error("write error.\n");
            virtio_serial_throttle_port(port, true);
        }
        debug("[+] WRITE BACK\n");
    }
    free(msg);
    // Notify that all jobs were received.
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
    // debug("vser->gpus[i].name=%s\n", vser->gpus[vser->gcount-1]->prop.name);
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
        // cuErrorExit(cuDeviceGet(&ctx->dev, i));
        /*reserved index 0 for thread*/
        memset(&ctx->cudaStreamBitmap,   ~0, sizeof(ctx->cudaStreamBitmap));
        memset(ctx->cudaEventBitmap,     ~0, sizeof(ctx->cudaEventBitmap));
        memset(ctx->cudaEvent,           0,  sizeof(cudaEvent_t) *CudaEventMaxNum);
        memset(ctx->cudaStream,          0,  sizeof(cudaStream_t)*CudaStreamMaxNum);
        INIT_LIST_HEAD(&ctx->vol);
        INIT_LIST_HEAD(&ctx->host_vol);
        ctx->moduleCount = 0;
    }
    tctx->worker_queue  = chan_init(0);
}

static void deinit_port(VirtIOSerialPort *port)
{
    ThreadContext *tctx = port->thread_context;
    int tid = port->id;
    /*delete elements in struct list_head*/
    /*
    */
    free(tctx->contexts);
    tctx->contexts = NULL;
    chan_dispose(tctx->worker_queue);
    debug("Ending thread %d computing workloads and queue!\n",tid);
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
