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
#include "virtio-ioc.h"
#include "hw/virtio/virtio-serial.h"
#include "qapi/error.h"
#include "qapi/qapi-events-char.h"
#include "exec/cpu-common.h"    // cpu_physical_memory_rw

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h> //PATH: /usr/local/cuda/include/builtin_types.h

#include <openssl/hmac.h> // hmac EVP_MAX_MD_SIZE
/*Encodes Base64 */
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <stdint.h>

#define func() printf("[FUNC]%s\n",__FUNCTION__)
#define error(fmt, arg...) printf("[ERROR]In file %s, line %d, "fmt, \
            __FILE__, __LINE__, ##arg)
#define debug(fmt, arg...) printf("[DEBUG] "fmt, ##arg)
#define cudaError(err) __cudaErrorCheck(err, __LINE__)

#define TYPE_VIRTIO_CONSOLE_SERIAL_PORT "virtcudaport"
#define VIRTIO_CONSOLE(obj) \
    OBJECT_CHECK(VirtConsole, (obj), TYPE_VIRTIO_CONSOLE_SERIAL_PORT)

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

typedef struct CudaDev {
    CUdevice device;
    CUcontext context;
    unsigned int cudaFunctionID[CudaFunctionMaxNum];
    CUfunction cudaFunction[CudaFunctionMaxNum];
    CUmodule module;
    int kernelsLoaded;
}CudaDev;

typedef struct KernelInfo {
    void *fatBin;
    char functionName[512];
    uint32_t funcID;
} KernelInfo;

typedef struct KernelConf {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
} KernelConf_t ;

int totalDevice;
int cudaFunctionNum;
CudaDev *cudaDevices;
CudaDev zeroedDevice;
CUdevice cudaDeviceCurrent[16];
KernelInfo devicesKernels[CudaFunctionMaxNum];
cudaEvent_t cudaEvent[CudaEventMaxNum];
uint32_t cudaEventNum;

cudaStream_t cudaStream[CudaStreamMaxNum];
uint32_t cudaStreamNum;
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

/*
 * from /hw/virtio/virtio-balloon.c
*/
/*
static void* gpa_to_hva(uint64_t pa)
{
    ram_addr_t addr;
    MemoryRegionSection section;
    void *p;

    // FIXME: remove get_system_memory(), but how? 
    section = memory_region_find(get_system_memory(), (ram_addr_t)pa, 1);
    if (!int128_nz(section.size) ||!memory_region_is_ram(section.mr) ) {
        memory_region_unref(section.mr);
        return NULL;
    }

    debug("0x%lx name: %s\n", pa, memory_region_name(section.mr));
    // Using memory_region_get_ram_ptr is bending the rules a bit, but
    // should be OK because we only want a single page.  
    addr = section.offset_within_region;
    p = memory_region_get_ram_ptr(section.mr) + addr;
    memory_region_unref(section.mr);
    return p;
}
*/

static void load_module_kernel(int devID, void *fatBin, 
                char *funcName, unsigned int funcID, int funcIndex)
{
    func();
    debug("Loading module... fatBin=%16p, name=%s, funcID=%d\n", 
            fatBin, funcName, funcID);
    cuError( cuModuleLoadData(&cudaDevices[devID].module, fatBin) );
    cuError( cuModuleGetFunction(&cudaDevices[devID].cudaFunction[funcIndex], \
        cudaDevices[devID].module, funcName) );
    cudaDevices[devID].cudaFunctionID[funcIndex] = funcID;
    cudaDevices[devID].kernelsLoaded = 1;
}
/*
static void reload_all_kernels(unsigned int id)
{
    func();
    int i=0;
    void *fatBin;
    char *functionName;
    unsigned int funcID;
    for(i=0; i<cudaFunctionNum; i++) {
        fatBin = devicesKernels[i].fatBin;
        functionName = devicesKernels[i].functionName;
        funcID = devicesKernels[i].funcID;
        load_module_kernel( cudaDeviceCurrent[id], fatBin, 
                functionName, funcID, i);
    }
}
*/

static cudaError_t initialize_device(unsigned int id)
{
    func();
    int dev = cudaDeviceCurrent[id];
    if (dev >= totalDevice) {
        error("setting device = %d\n", dev);
        return cudaErrorInvalidDevice;
    } else {
        // device reset
        if ( !memcmp(&zeroedDevice, &cudaDevices[dev], sizeof(CudaDev)) ) {
            cuError( cuDeviceGet(&cudaDevices[dev].device, dev) );
            cuError( cuCtxCreate(&cudaDevices[dev].context, 0, cudaDevices[dev].device) );
            debug("Device was reset therefore no context\n");
        } else {
            cuError( cuCtxSetCurrent(cudaDevices[dev].context) );
            debug("Cuda device %d\n", cudaDevices[dev].device);
        }
        //if( cudaDevices[dev].kernelsLoaded == 0)
        //  reload_all_kernels(id);
        return cudaSuccess;
    }
}

static unsigned int get_current_id(unsigned int tid)
{
    return tid%totalDevice;
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
static void cuda_register_fatbinary(void *buf, ssize_t len)
{
    func();
    unsigned int i;
    void *fat_bin;
    hwaddr gpa;
    VirtIOArg *arg = (VirtIOArg*)buf;
    uint32_t src_size = arg->srcSize;
    // check out hmac
    gpa = (hwaddr)(arg->src);
    // try cpu_physical_memory_map next time
    // fat_bin = malloc(arg->srcSize);
    // cpu_physical_memory_read(gpa, fat_bin, arg->srcSize);
    hwaddr src_len = (hwaddr)src_size;
    fat_bin = cpu_physical_memory_map(gpa, &src_len, 1);
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
    for(i =0; i<CudaEventMaxNum; i++)
        memset(&cudaEvent[i], 0, sizeof(cudaEvent_t));
    cuError(cuInit(0));
    cuError( cuDeviceGetCount(&totalDevice) );
    cudaDevices = (CudaDev *)malloc(totalDevice * sizeof(CudaDev));
    memset(&zeroedDevice, 0, sizeof(CudaDev));
    i = totalDevice;
    while(i-- != 0) {
        debug("[+] Create context for device %d\n", i);
        memset(&cudaDevices[i], 0, sizeof(CudaDev));
        cuError( cuDeviceGet(&cudaDevices[i].device, i) );
        cuError( cuCtxCreate(&cudaDevices[i].context, 0, cudaDevices[i].device) );
        memset(&cudaDevices[i].cudaFunction, 0, sizeof(CUfunction) * CudaFunctionMaxNum);
        cudaDevices[i].kernelsLoaded = 0;
    }

    cudaFunctionNum = 0;
    cudaDeviceCurrent[0] = cudaDevices[0].device;
    cudaEventNum = 0;
    cudaStreamNum = 0;
    arg->cmd = cudaSuccess;
}

static void cuda_unregister_fatinary(void *buf, ssize_t len)
{
    int i;
    VirtIOArg *arg = (VirtIOArg*)buf;
    func();
    for(i=0; i<totalDevice; i++) {
        if ( memcmp(&zeroedDevice, &cudaDevices[i], sizeof(CudaDev)) != 0 )
            cuError( cuCtxDestroy(cudaDevices[i].context) );
        free(devicesKernels[i].fatBin);
        memset(devicesKernels[i].functionName, 0, sizeof(devicesKernels[i].functionName));
    }
    free(cudaDevices);
    arg->cmd = cudaSuccess;
}

static void cuda_register_function(uint8_t *buf, ssize_t len)
{
    hwaddr fat_bin_gpa;
    hwaddr func_name_gpa;
    uint32_t func_id;
    uint32_t fat_size, name_size;
    VirtIOArg *arg = (VirtIOArg*)buf;
    func();

    fat_bin_gpa = (hwaddr)(arg->src);
    func_name_gpa = (hwaddr)(arg->dst);
    func_id = arg->flag;
    fat_size = arg->srcSize;
    name_size = arg->dstSize;
    // initialize the KernelInfo
    // fatSize = next_p2(header->srcSize);
    // malloc 最大申请可达到4G内存，应该够用了
    devicesKernels[cudaFunctionNum].fatBin = malloc(fat_size);

    cpu_physical_memory_read(fat_bin_gpa, \
        devicesKernels[cudaFunctionNum].fatBin, fat_size);
    cpu_physical_memory_read(func_name_gpa, \
        devicesKernels[cudaFunctionNum].functionName, name_size);

    devicesKernels[cudaFunctionNum].funcID = func_id;
    debug("fatBin = %16p, name='%s', cudaFunctionNum = %d\n", 
        devicesKernels[cudaFunctionNum].fatBin, \
        (char*)devicesKernels[cudaFunctionNum].functionName, \
        cudaFunctionNum);
    load_module_kernel(cudaFunctionNum, \
        devicesKernels[cudaFunctionNum].fatBin, \
        (char*)devicesKernels[cudaFunctionNum].functionName, \
        func_id, cudaFunctionNum);
    cudaFunctionNum++;
    arg->cmd = cudaSuccess;
}

static void cuda_configure_call(uint8_t *buf, ssize_t *len)
{
    func();
    cudaError_t err;
    VirtIOArg *header = (VirtIOArg*)buf;
    // unsigned int id = get_current_id( (unsigned int)header->tid );
    KernelConf_t *kernelConf = (KernelConf_t*)(buf+sizeof(VirtIOArg));

    debug("gridDim=%u %u %u\n", kernelConf->gridDim.x, 
        kernelConf->gridDim.y, kernelConf->gridDim.z);
    debug("blockDim=%u %u %u\n", kernelConf->blockDim.x, 
        kernelConf->blockDim.y, kernelConf->blockDim.z);
    debug("sharedMem=%ld\n", kernelConf->sharedMem);
    cudaError( (err= cudaConfigureCall(kernelConf->gridDim, 
        kernelConf->blockDim, kernelConf->sharedMem, kernelConf->stream)));
    debug(" return value=%d\n", err);
    header->cmd = err;
    *len = sizeof(VirtIOArg);
}

static void cuda_setup_argument(void *buf, ssize_t len)
{
    func();
}

static void cuda_launch(uint8_t *buf, ssize_t *len)
{
    func();
    cudaError_t err;
    int i=0;
    uint32_t funcID, paraNum, paraIdx, funcIdx;
    void **paraBuf;
    VirtIOArg *header = (VirtIOArg*)buf;
    unsigned int id = get_current_id( (unsigned int)header->tid );
    KernelConf_t *para = (KernelConf_t*)(buf+sizeof(VirtIOArg));
    KernelConf_t *kernelConf = (KernelConf_t*)(buf+sizeof(VirtIOArg)+header->srcSize);
    funcID = header->flag;
    paraNum = *((uint32_t*)para);
    debug(" paraNum = %d\n", paraNum);
    
    paraBuf = malloc(paraNum * sizeof(void*));
    paraIdx = sizeof(uint32_t);
    for(i=0; i<paraNum; i++) {
        paraBuf[i] = &para[paraIdx + sizeof(uint32_t)];
        debug("arg %d = 0x%llx size=%u byte\n", i, 
            *(unsigned long long*)paraBuf[i], *(unsigned int*)&para[paraIdx]);
        paraIdx += *((uint32_t*)&para[paraIdx]) + sizeof(uint32_t);
    }

    for(funcIdx=0; funcIdx < cudaFunctionNum; funcIdx++) {
        if( cudaDevices[cudaDeviceCurrent[id]].cudaFunctionID[funcIdx] == funcID)
            break;
    }

    debug("gridDim=%u %u %u\n", kernelConf->gridDim.x, 
        kernelConf->gridDim.y, kernelConf->gridDim.z);
    debug("blockDim=%u %u %u\n", kernelConf->blockDim.x, 
        kernelConf->blockDim.y, kernelConf->blockDim.z);
    debug("sharedMem=%ld\n", kernelConf->sharedMem);
    /*
    cudaError( (err= cudaConfigureCall(kernelConf->gridDim, 
        kernelConf->blockDim, kernelConf->sharedMem, kernelConf->stream)));
    */
    cuError( (err = cuLaunchKernel(
        cudaDevices[cudaDeviceCurrent[id]].cudaFunction[funcIdx], \
        kernelConf->gridDim.x, kernelConf->gridDim.y, kernelConf->gridDim.z,\
        kernelConf->blockDim.x, kernelConf->blockDim.y, kernelConf->blockDim.z,\
        kernelConf->sharedMem, kernelConf->stream, paraBuf, NULL) ) );
    header->cmd = err;
    *len = sizeof(VirtIOArg);
    free(paraBuf);
    paraBuf = NULL;
}

static void cuda_memcpy(uint8_t *buf, ssize_t len)
{
    cudaError_t err;
    uint32_t size;
    VirtIOArg *arg = (VirtIOArg*)buf;
    void *src, *dst;
    unsigned int id = get_current_id( (unsigned int)arg->tid );
    func();
    // in case cudaReset was the previous call
    initialize_device(id);
    debug("src=0x%lx, srcSize=%d, dst=9x%lx, dstSize=%d, kind=%lu\n", \
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag);
    size = arg->srcSize;
    if (arg->flag == cudaMemcpyHostToDevice) {
        hwaddr src_len = (hwaddr)size;
        src = cpu_physical_memory_map(arg->src, &src_len, 1);
        if (!src || src_len != size) {
            error("Failed to map MMIO memory for"
                              " gpa 0x%lx element size %u\n",
                                arg->src, size);
            return ;
        }
        dst = (void *)arg->dst;
        cuError( (err= cuMemcpyHtoD((CUdeviceptr)dst, (void *)src, size)));
    } else if (arg->flag == cudaMemcpyDeviceToHost) {
        src = (void *)(arg->src);
        hwaddr dst_len = (hwaddr)size;
        // gpa => hva
        dst = cpu_physical_memory_map(arg->dst, &dst_len, 1);
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
        src = (void *)(arg->src);
        dst = (void *)(arg->dst);
        cuError( (err=cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, size)) );
    }
    arg->cmd = err;
    debug(" return value=%d\n", err);
}

static void cuda_malloc(uint8_t *buf, ssize_t len)
{
    cudaError_t err;
    void *devPtr;
    uint32_t size;
    VirtIOArg *arg = (VirtIOArg*)buf;
    func();

    unsigned int id = get_current_id( (unsigned int)arg->tid );
    // in case cudaReset was the previous call
    initialize_device(id);
    size = arg->srcSize;
    cudaError( (err= cudaMalloc(&devPtr, size)));
    arg->dst = (uint64_t)devPtr;
    arg->cmd = err;
    debug(" devPtr=0x%lx, size=%d, return value=%d\n", arg->dst, size, err);
}

static void cuda_free(uint8_t *buf, ssize_t len)
{
    cudaError_t err;
    void *src;
    VirtIOArg *arg = (VirtIOArg*)buf;
    unsigned int id = get_current_id( (unsigned int)arg->tid );
    func();
    // in case of cudaReset
    initialize_device(id);
    src = (void*)(arg->src);
    cudaError( (err= cudaFree(src)) );
    arg->cmd = err;
    debug(" ptr = 0x%lx, return value=%d\n", arg->src, err);
}

static void cuda_get_device(void *buf, ssize_t len)
{
    int dev = 0;
    VirtIOArg *arg = (VirtIOArg*)buf;
    hwaddr gpa = (hwaddr)(arg->dst);
    func();
    unsigned int id = get_current_id( (unsigned int)arg->tid );
    initialize_device(id);
    arg->cmd = cudaSuccess;
    dev = (int)(cudaDevices[cudaDeviceCurrent[id]].device);
    cpu_physical_memory_write(gpa, &dev, sizeof(int));
}

static void cuda_get_device_properties(void *buf, ssize_t len)
{
    cudaError_t err;
    int devID;
    struct cudaDeviceProp prop;
    VirtIOArg *arg = (VirtIOArg*)buf;
    hwaddr gpa = (hwaddr)(arg->dst);
    func();
    devID = (int)(arg->flag);
    debug("Get prop for device %d\n", devID);
    cudaError( (err=cudaGetDeviceProperties(&prop, devID)) );
    debug("Device %d : \"%s\" with compute %d.%d capability.\n", devID, prop.name, prop.major, prop.minor);
    arg->cmd = err;
    cpu_physical_memory_write(gpa, &prop, arg->dstSize);
}

static void cuda_set_device(void *buf, ssize_t len)
{
    func();
    cudaError_t err = 0;
    VirtIOArg *header = (VirtIOArg*)buf;
    unsigned int id = get_current_id( (unsigned int)header->tid );
    cudaDeviceCurrent[id] = (int)(header->flag);
    cudaError( (err=initialize_device(id)) );
    header->cmd = err;
    debug("set devices=%d\n", (int)(header->flag));
}

static void cuda_get_device_count(void *buf, ssize_t len)
{
    func();
    VirtIOArg *header = (VirtIOArg*)buf;
    unsigned int id = get_current_id( (unsigned int)header->tid );
    initialize_device(id);
    debug("Device count=%d.\n", totalDevice);
    header->cmd = (int32_t)cudaSuccess;
    header->flag = (uint64_t)totalDevice;
}

static void cuda_device_reset(void *buf, ssize_t len)
{
    func();
    cudaError_t err = 0;
    VirtIOArg *header = (VirtIOArg*)buf;
    unsigned int id = get_current_id( (unsigned int)header->tid );
    // should get rid of events for current devices
    cuCtxDestroy(cudaDevices[cudaDeviceCurrent[id]].context ) ;
    memset( &cudaDevices[cudaDeviceCurrent[id]], 0, sizeof(CudaDev) );
    cudaError( (err= cudaDeviceReset()) );
    header->cmd = err;
    debug("reset devices\n");
}

static void cuda_stream_create(void *buf, ssize_t len)
{
    cudaError_t err = 0;
    VirtIOArg *header = (VirtIOArg*)buf;
    func();

    cudaError( (err = cudaStreamCreate(&cudaStream[cudaStreamNum]) ));
    header->flag = cudaStreamNum;
    memcpy(buf+sizeof(VirtIOArg), &cudaStream[cudaStreamNum], header->srcSize);
    cudaStreamNum++;
    header->cmd = err;
}

static void cuda_stream_destroy(void *buf, ssize_t *len)
{
    cudaError_t err = 0;
    VirtIOArg *header = (VirtIOArg*)buf;
    func();
    // uint32_t idx;
    // idx = header->flag;
    cudaStream_t stream = (cudaStream_t)(buf+sizeof(VirtIOArg));
//  cudaError( (err=cudaStreamDestroy(cudaStream[idx]) ));
    cudaError( (err=cudaStreamDestroy(stream) ));
    header->cmd = err;
//  memset(&cudaStream[idx], 0, sizeof(cudaStream_t));
    *len = sizeof(VirtIOArg);
}

static void cuda_event_create(void *buf, ssize_t len)
{
    func();
    cudaError_t err = 0;
    uint64_t idx = 0;
    VirtIOArg *header = (VirtIOArg*)buf;
    idx = cudaEventNum;
    cudaError( (err=cudaEventCreate(&cudaEvent[idx]) ));
    header->flag = (uint64_t)idx;
    header->cmd = err;
    cudaEventNum = (cudaEventNum+1)%CudaEventMaxNum;
    // debug("create event %u, idx is %u\n", cudaEvent[idx], idx);

}

static void cuda_event_destroy(void *buf, ssize_t len)
{
    func();
    cudaError_t err = 0;
    uint32_t idx = 0;
    VirtIOArg *header = (VirtIOArg*)buf;
    idx = header->flag;
    cudaEvent_t event = cudaEvent[idx];
    cudaError( (err=cudaEventDestroy(event)) );
    header->cmd = err;
    // debug("destroy event %u\n", event);
}

static void cuda_event_record(void *buf, ssize_t *len)
{
    func();
    cudaError_t err = 0;
    uint64_t idx = 0;
    VirtIOArg *header = (VirtIOArg*)buf;
    idx = header->param;
    debug("idx = %lu\n", idx);   
    if(header->flag == (uint64_t)-1)
    {
        // debug("record event= %u , streams = 0\n", cudaEvent[idx]);
        cudaError( (err=cudaEventRecord(cudaEvent[idx], 0)) );
    }
    else
    {
        cudaStream_t stream = (cudaStream_t)(buf+sizeof(VirtIOArg)+header->srcSize);
//      debug("record event %u, stream=%u\n", event, stream);
        cudaError( (err=cudaEventRecord(cudaEvent[idx], stream)) );
    }
    header->cmd = err;
    *len = sizeof(VirtIOArg);
}

static void cuda_event_synchronize(void *buf, ssize_t len)
{
    func();
    cudaError_t err = 0;
    uint64_t idx = 0;
    VirtIOArg *header = (VirtIOArg*)buf;
    idx = header->flag;
    // debug("record event %u , idx = %lu\n", cudaEvent[idx], idx);
    cudaError( (err=cudaEventSynchronize(cudaEvent[idx])) );
    header->cmd = err;
}

static void cuda_event_elapsedtime(void *buf, ssize_t len)
{
    cudaError_t err = 0;
    uint64_t start_idx, stop_idx;
    float        time = 0;
    func();
    VirtIOArg *header = (VirtIOArg*)buf;
    // unsigned int id = get_current_id( (unsigned int)header->tid );
    start_idx = header->flag;
    stop_idx = header->param;
    debug("start_idx = %lu , stop_idx = %lu\n", start_idx, stop_idx);
    cudaError( (err=cudaEventElapsedTime(&time, cudaEvent[start_idx], cudaEvent[stop_idx])) );
    header->cmd = err;
    memcpy(buf+sizeof(VirtIOArg), &time, sizeof(float));
    // debug("event start %d to end %d, elapsedtime %f\n", (int)cudaEvent[start_idx], (int)cudaEvent[stop_idx], time);
}

static void cuda_thread_synchronize(void *buf, ssize_t len)
{
    cudaError_t err = 0;
    func();
    VirtIOArg *header = (VirtIOArg*)buf;
    cudaError( (err=cudaThreadSynchronize()) );
    header->cmd = err;
}

static void cuda_get_last_error(void *buf, ssize_t len)
{
    cudaError_t err = 0;
    func();
    VirtIOArg *header = (VirtIOArg*)buf;
    // unsigned int id = get_current_id( (unsigned int)header->tid );
    cudaError( (err=cudaGetLastError()) );
    header->cmd = err;
}
/*
static inline void cpu_physical_memory_read(hwaddr addr,
                                            void *buf, int len)
{
    cpu_physical_memory_rw(addr, buf, len, 0);
}
static inline void cpu_physical_memory_write(hwaddr addr,
                                             const void *buf, int len)
{
    cpu_physical_memory_rw(addr, (void *)buf, len, 1);
}
*/
static void cuda_gpa_to_hva(void *buf, ssize_t len)
{
    int a;
    hwaddr gpa;
    VirtIOArg *arg = (VirtIOArg*)buf;
    gpa = (hwaddr)(arg->src);
    //a = gpa_to_hva(arg->src);

    cpu_physical_memory_read(gpa, &a, sizeof(int));
    debug("a=%d\n", a);
    a++;
    cpu_physical_memory_write(gpa, &a, sizeof(int));
    arg->cmd = 1;//(int32_t)cudaSuccess;
}

/* 
*   Callback function that's called when the guest sends us data 
*/
static ssize_t flush_buf(VirtIOSerialPort *port,
                         const uint8_t *buf, ssize_t len)
{
    ssize_t ret;
    VirtIOArg header;
    void *out;
    func();

    debug("port->id=%d, buf=%6s, len=%ld\n", port->id, buf, len);
    memcpy((void *)&header, (char *)buf, sizeof(VirtIOArg));
    out = (uint8_t *)malloc(len);
    memcpy((void *)out, (void *)buf, len);
    debug("[+] header.cmd=%u, nr= %u \n", \
        header.cmd, _IOC_NR(header.cmd));
    debug("[+] header.tid = %d\n", header.tid);
    switch(header.cmd) {
        case VIRTIO_CUDA_HELLO:
            cuda_gpa_to_hva(out, len);
            break;
        case VIRTIO_CUDA_REGISTERFATBINARY:
            cuda_register_fatbinary(out, len);
            break;
        case VIRTIO_CUDA_UNREGISTERFATBINARY:
            cuda_unregister_fatinary(out, len);
            break;
        case VIRTIO_CUDA_REGISTERFUNCTION:
            cuda_register_function(out, len);
            break;

        case VIRTIO_CUDA_LAUNCH:
            cuda_launch(out, &len);
            break;
        case VIRTIO_CUDA_MALLOC:
            cuda_malloc(out, len);
            break;
        case VIRTIO_CUDA_MEMCPY:
            cuda_memcpy(out, len);
            break;
        case VIRTIO_CUDA_FREE:
            cuda_free(out, len);
            break;
        case VIRTIO_CUDA_GETDEVICE:
            cuda_get_device(out, len);
            break;
        case VIRTIO_CUDA_GETDEVICEPROPERTIES:
            cuda_get_device_properties(out, len);
            break;
        case VIRTIO_CUDA_CONFIGURECALL:
            cuda_configure_call(out, &len);
            break;
        case VIRTIO_CUDA_SETUPARGUMENT:
            cuda_setup_argument(out, len);
            break;
        case VIRTIO_CUDA_GETDEVICECOUNT:
            cuda_get_device_count(out, len);
            break;
        case VIRTIO_CUDA_SETDEVICE:
            cuda_set_device(out, len);
            break;
        case VIRTIO_CUDA_DEVICERESET:
            cuda_device_reset(out, len);
            break;
        case VIRTIO_CUDA_STREAMCREATE:
            cuda_stream_create(out, len);
            break;
        case VIRTIO_CUDA_STREAMDESTROY:
            cuda_stream_destroy(out, &len);
            break;
        case VIRTIO_CUDA_EVENTCREATE:
            cuda_event_create(out, len);
            break;
        case VIRTIO_CUDA_EVENTDESTROY:
            cuda_event_destroy(out, len);
            break;
        case VIRTIO_CUDA_EVENTRECORD:
            cuda_event_record(out, &len); // fuck you for forgetting &
            break;
        case VIRTIO_CUDA_EVENTSYNCHRONIZE:
            cuda_event_synchronize(out, len);
            break;
        case VIRTIO_CUDA_EVENTELAPSEDTIME:
            cuda_event_elapsedtime(out, len);
            break;
        case VIRTIO_CUDA_THREADSYNCHRONIZE:
            cuda_thread_synchronize(out, len);
            break;
        case VIRTIO_CUDA_GETLASTERROR:
            cuda_get_last_error(out, len);
            break;
        default:
            error("[+] header.cmd=%u, nr= %u \n", \
                header.cmd, _IOC_NR(header.cmd));
            return -1;
    }
    ret = virtio_serial_write(port, out, len);
    if (ret < len) {
        /*
         * Ideally we'd get a better error code than just -1, but
         * that's what the chardev interface gives us right now.  If
         * we had a finer-grained message, like -EPIPE, we could close
         * this connection.
         */
        if (ret < 0)
            ret = 0;

        /* XXX we should be queuing data to send later for the
         * console devices too rather than silently dropping
         * console data on EAGAIN. The Linux virtio-console
         * hvc driver though does sends with spinlocks held,
         * so if we enable throttling that'll stall the entire
         * guest kernel, not merely the process writing to the
         * console.
         *
         * While we could queue data for later write without
         * enabling throttling, this would result in the guest
         * being able to trigger arbitrary memory usage in QEMU
         * buffering data for later writes.
         *
         * So fixing this problem likely requires fixing the
         * Linux virtio-console hvc driver to not hold spinlocks
         * while writing, and instead merely block the process
         * that's writing. QEMU would then need some way to detect
         * if the guest had the fixed driver too, before we can
         * use throttling on host side.
         */
        
        virtio_serial_throttle_port(port, true);
    }
    out=NULL;
    debug("[+] WRITE BACK\n");
    return ret;
}

/* Callback function that's called when the guest opens/closes the port */
static void set_guest_connected(VirtIOSerialPort *port, int guest_connected)
{
    func();
    // VirtConsole *vcon = VIRTIO_CONSOLE(port);
    DeviceState *dev = DEVICE(port);
    //VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_GET_CLASS(port);

    if (dev->id) {
        qapi_event_send_vserport_change(dev->id, guest_connected,
                                        &error_abort);
    }
}

static void virtconsole_enable_backend(VirtIOSerialPort *port, bool enable)
{
    func();
    return ;
}

static void virtconsole_realize(DeviceState *dev, Error **errp)
{
    func();
    VirtIOSerialPort *port = VIRTIO_SERIAL_PORT(dev);
    VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_GET_CLASS(dev);
    debug("port->id == %d\n", port->id == 0);
    if (port->id == 0 && !k->is_console) {
        error_setg(errp, "Port number 0 on virtio-serial devices reserved "
                   "for virtconsole devices for backward compatibility.");
        return;
    }

    virtio_serial_open(port);
}

static void virtconsole_unrealize(DeviceState *dev, Error **errp)
{
    func();
    VirtConsole *vcon = VIRTIO_CONSOLE(dev);

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
