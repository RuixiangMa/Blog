---

---
### 1. 介绍 
**cuFile** API 是 NVIDIA 为支持 GPUDirect Storage (GDS) 而提供的接口集，它是 CUDA Driver C API 的一部分。

### 2. 基本接口及其功能描述
**cuFileDriverOpen**:
初始化驱动程序会话，以支持后续的GDS I/O操作。成功调用此函数后，将建立起与内核驱动的通信

**cuFileDriverClose**: 关闭驱动程序会话并释放所有与GDS相关的资源。这个步骤通常是在进程结束时隐式完成的，但在某些情况下也可能需要显式地调用它来确保资源的及时释放

**cuFileHandleRegister**: 将操作系统级别的文件句柄注册到CUDA环境中，这使得应用程序能够通过GPU直接访问文件数据

**cuFileHandleDeregister**: 一旦完成了对特定文件的操作，应该调用此函数来释放与该文件相关的句柄资源，从而允许系统回收这些资源供其他用途使用

**cuFileBufRegister**: 注册一个内存缓冲区，以便它可以在GDS操作中被引用。这一步骤确保了指定的缓冲区可以用于直接从存储设备向GPU内存或反之进行数据传输

**cuFileBufDeregister**: 当不再需要某个已经注册的内存缓冲区时，可以通过此函数释放相关资源

**cuFileWrite**: 此函数允许应用程序将数据从GPU内存直接写入到注册的文件句柄所指向的存储设备中。通过使用此函数，可以绕过传统的CPU内存路径，从而减少延迟并提高I/O吞吐量

**cuFileRead**: 此函数用于将数据从存储设备读取到GPU显存中。与cuFileWrite类似，它也避免了CPU内存作为中间介质的需求，实现更高效的直接数据路径。

### 3.工作流程

**初始化**: 使用cuFileDriverOpen初始化驱动程序，并确保CUDA环境已经准备好进行GDS操作。

**注册文件句柄**: 通过cuFileHandleRegister将操作系统级别的文件句柄注册到CUDA环境中，使后续的I/O操作可以直接访问该文件。

**注册缓冲区**: 如果尚未完成，使用cuFileBufRegister注册参与I/O操作的GPU缓冲区。

**执行读/写操作**: 根据需求调用cuFileWrite或cuFileRead来进行数据传输。

**清理资源**: 在完成所有I/O操作后，依次调用cuFileBufDeregister释放缓冲区，cuFileHandleDeregister注销文件句柄，并最终调用cuFileDriverClose关闭驱动程序会话。

### 4. 示例代码
    
```
#define MAX_BUFFER_SIZE (4 * 1024 * 1024) 

int main() {
    int fd = -1;
    ssize_t ret = -1;
    void *writePtr = NULL, *readPtr = NULL; // 分别用于写入和读取的设备指针
    const size_t size = MAX_BUFFER_SIZE;
    CUfileError_t status;
    const char *TESTFILE = "/path/test.file"; 
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;

    // 初始化CUDA驱动
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileDriverOpen failed: " << cuFileGetErrorString(status) << std::endl;
        return -1;
    }

    // 分配设备内存
    cudaError_t cudaStatus = cudaMalloc(&writePtr, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc for write buffer failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cuFileDriverClose();
        return -1;
    }
    
    // 分配另一个设备内存区域用于读取
    cudaStatus = cudaMalloc(&readPtr, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc for read buffer failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(writePtr);
        cuFileDriverClose();
        return -1;
    }

    // 打开文件
    fd = open(TESTFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
    if (fd == -1) {
        std::cerr << "open failed: " << strerror(errno) << std::endl;
        cudaFree(writePtr);
        cudaFree(readPtr);
        cuFileDriverClose();
        return -1;
    }

    // 注册文件句柄到cuFile
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileHandleRegister failed: " << cuFileGetErrorString(status) << std::endl;
        close(fd);
        cudaFree(writePtr);
        cudaFree(readPtr);
        cuFileDriverClose();
        return -1;
    }

    // 注册设备内存缓冲区
    status = cuFileBufRegister(writePtr, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileBufRegister for write buffer failed: " << cuFileGetErrorString(status) << std::endl;
        cuFileHandleDeregister(cf_handle);
        close(fd);
        cudaFree(writePtr);
        cudaFree(readPtr);
        cuFileDriverClose();
        return -1;
    }

    status = cuFileBufRegister(readPtr, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileBufRegister for read buffer failed: " << cuFileGetErrorString(status) << std::endl;
        cuFileBufDeregister(writePtr);
        cuFileHandleDeregister(cf_handle);
        close(fd);
        cudaFree(writePtr);
        cudaFree(readPtr);
        cuFileDriverClose();
        return -1;
    }

    // 执行写入操作
    ret = cuFileWrite(cf_handle, writePtr, size, 0, 0);
    if (ret < 0 || ret != size) {
        std::cerr << "cuFileWrite failed or incomplete write: " << ret << " vs " << size << std::endl;
        goto cleanup;
    }

    // 执行读取操作
    ret = cuFileRead(cf_handle, readPtr, size, 0, 0);
    if (ret < 0 || ret != size) {
        std::cerr << "cuFileRead failed or incomplete read: " << ret << " vs " << size << std::endl;
        goto cleanup;
    }

cleanup:
    // 清理资源
    cuFileBufDeregister(writePtr);
    cuFileBufDeregister(readPtr);
    cuFileHandleDeregister(cf_handle);
    close(fd);
    cudaFree(writePtr);
    cudaFree(readPtr);

    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileDriverClose failed: " << cuFileGetErrorString(status) << std::endl;
    }

    return (ret == size) ? 0 : -1;
}
```