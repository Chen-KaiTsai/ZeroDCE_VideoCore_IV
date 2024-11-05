#include "main.h"

unique_ptr<cl_platform_id[]> ocl::getPlatformID(void)
{
	cl_int error;
	cl_uint nPlatform = 0;
	error = clGetPlatformIDs(0, nullptr, &nPlatform);
	if (error != CL_SUCCESS || nPlatform == 0) {
        getErrMsg(error);
		return nullptr;
	}

	unique_ptr<cl_platform_id[]> platforms = make_unique<cl_platform_id[]>(nPlatform);

	error = clGetPlatformIDs(nPlatform, platforms.get(), nullptr);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		return nullptr;
	}

	return std::move(platforms);
}

unique_ptr<cl_device_id[]> ocl::getDeviceID(const cl_platform_id platform, const cl_device_type type)
{
	cl_int error;
	cl_uint nDevice = 0;
	error = clGetDeviceIDs(platform, type, 0, nullptr, &nDevice);
	if (error != CL_SUCCESS || nDevice == 0) {
        getErrMsg(error);
		return nullptr;
	}

	unique_ptr<cl_device_id[]> devices = make_unique<cl_device_id[]>(nDevice);

	error = clGetDeviceIDs(platform, type, nDevice, devices.get(), nullptr);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		return nullptr;
	}

	return std::move(devices);
}

unique_ptr<char[]> ocl::getPlatformInfo(const cl_platform_id platform, const cl_platform_info info)
{
	cl_int error;
	size_t pSize = 0;
	error = clGetPlatformInfo(platform, info, 0, nullptr, &pSize);
	if (error != CL_SUCCESS || pSize == 0) {
        getErrMsg(error);
		return nullptr;
	}

	unique_ptr<char[]> pInfo = make_unique<char[]>(pSize + 1);
	pInfo[pSize] = '\0';

	error = clGetPlatformInfo(platform, info, pSize, pInfo.get(), nullptr);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		return nullptr;
	}

	return std::move(pInfo);
}

unique_ptr<char[]> ocl::getDeviceInfo(const cl_device_id device, const cl_device_info info)
{
	cl_int error;
	size_t dSize = 0;
	error = clGetDeviceInfo(device, info, 0, nullptr, &dSize);
	if (error != CL_SUCCESS || dSize == 0) {
        getErrMsg(error);
		return nullptr;
	}

	unique_ptr<char[]> dInfo = make_unique<char[]>(dSize);

	error = clGetDeviceInfo(device, info, dSize, dInfo.get(), nullptr);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		return nullptr;
	}

	return std::move(dInfo);
}

cl_context ocl::createContext(cl_platform_id platform, cl_device_id device)
{
	cl_int error;
	cl_context context = nullptr;
	cl_context_properties contextProp[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0 };

	context = clCreateContext(contextProp, 1, &device, nullptr, nullptr, &error);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		return nullptr;
	}
	
	return context;
}

cl_command_queue ocl::createQueue(cl_context context, cl_device_id device)
{
	cl_int error;
	cl_command_queue cmdQueue = nullptr;
#if CL_TARGET_OPENCL_VERSION < 200
    cl_command_queue_properties queueProp;
    queueProp = CL_QUEUE_PROFILING_ENABLE;
	cmdQueue = clCreateCommandQueue(context, device, queueProp, &error);
#else
	cl_command_queue_properties queueProp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	cmdQueue = clCreateCommandQueueWithProperties(context, device, queueProp, &error);
#endif
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		return nullptr;
	}
	
	return cmdQueue;
}

cl_program ocl::createProgramFromSource(cl_context context, const std::vector<const char*> fileNames)
{
	cl_int error;
	cl_program program = nullptr;
	size_t nProgram = fileNames.size();
	char** programSrcs = new char* [nProgram];
	size_t* lengths = new size_t[nProgram];
	
	printf("Reading cl files...\n");

	for (size_t i = 0; i < nProgram; ++i)
	{
		std::ifstream ifsFile;
		ifsFile.open(fileNames[i], std::ios::in | std::ios::binary);
		if (!ifsFile.is_open()) {
			perror("Unable to open file");
			return nullptr;
		}

		ifsFile.seekg(0, std::ios::end);
		lengths[i] = ifsFile.tellg();
		ifsFile.seekg(0, std::ios::beg);

		programSrcs[i] = new char[lengths[i]];
		ifsFile.read(programSrcs[i], lengths[i]);
		ifsFile.close();
	}

	program = clCreateProgramWithSource(context, static_cast<cl_uint>(nProgram), (const char**)programSrcs, lengths, &error);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		return nullptr;
	}

	for (int i = 0; i < nProgram; ++i)
		delete[] programSrcs[i];
	delete[] programSrcs;
	delete[] lengths;

	printf("Finish\n");

	return program;
}

cl_kernel ocl::createKernel(cl_program program, const char* kernelName)
{
	cl_int error;
	cl_kernel kernel = nullptr;
	kernel = clCreateKernel(program, kernelName, &error);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		return nullptr;
	}

	return kernel;
}

void ocl::buildProgram(cl_program program, cl_device_id device, const char* options)
{
	cl_int error;
	error = clBuildProgram(program, NUM_DEVICE, &device, options, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		size_t nInfo;
		error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &nInfo);
		if (error != CL_SUCCESS) {
            getErrMsg(error);
			return;
		}

		std::unique_ptr<char[]> pInfo = std::make_unique<char[]>(nInfo);

		error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, nInfo, pInfo.get(), nullptr);
		if (error != CL_SUCCESS) {
            getErrMsg(error);
			return;
		}
		else {
			printf("--------------- Compiler Log ---------------\n\n%s", static_cast<const char*>(pInfo.get()));
		}
		return;
	}
}

cl_mem ocl::createBuffer(cl_context context, cl_mem_flags flags, size_t size, void* hBuffer)
{
	cl_int error;
	cl_mem dBuffer = nullptr;
	dBuffer = clCreateBuffer(context, flags, size, hBuffer, &error);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		return nullptr;
	}

	return dBuffer;
}

void ocl::launchOneKernelAndWait(cl_command_queue dQueue, cl_kernel dKernel, cl_int dim, const size_t* gws, const size_t* lws)
{
	cl_int error;
	cl_event launchEvent;
	error = clEnqueueNDRangeKernel(dQueue, dKernel, dim, nullptr, gws, lws, 0, nullptr, &launchEvent);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		exit(EXIT_FAILURE);
	}
	error = clWaitForEvents(1, &launchEvent);
	if (error != CL_SUCCESS) {
        getErrMsg(error);
		exit(EXIT_FAILURE);
	}
}

void ocl::launchOneKernelAndProfile(cl_command_queue dQueue, cl_kernel dKernel, cl_int dim, const size_t* gws, const size_t* lws)
{
    cl_int error;
    cl_event launchEvent;
    cl_ulong start, end;
    error = clEnqueueNDRangeKernel(dQueue, dKernel, dim, nullptr, gws, lws, 0, nullptr, &launchEvent);
    if (error != CL_SUCCESS) {
        getErrMsg(error);
        exit(EXIT_FAILURE);
    }
    error = clWaitForEvents(1, &launchEvent);
    if (error != CL_SUCCESS) {
        getErrMsg(error);
        exit(EXIT_FAILURE);
    }

    clGetEventProfilingInfo(launchEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(launchEvent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    clReleaseEvent(launchEvent);

    printf("Execution Time : %ld.%ld s\n", (end - start) / 1000000000, (end  - start) % 1000000000);
}

void ocl::readBufferBlockNoOffset(cl_command_queue dQueue, cl_mem dBuffer, size_t size, void* hBuffer)
{
	cl_int error;
	error = clEnqueueReadBuffer(dQueue, dBuffer, CL_TRUE, 0, size, hBuffer, 0, nullptr, nullptr);

	if (error != CL_SUCCESS) {
        getErrMsg(error);
		exit(EXIT_FAILURE);
	}
}

void ocl::getErrMsg(cl_int error)
{
    switch (error)
    {
    case CL_SUCCESS:
        printf("Success.\n");
        break;
    case CL_DEVICE_NOT_FOUND:
        printf("Device not found.\n");
        break;
    case CL_DEVICE_NOT_AVAILABLE:
        printf("Device not available.\n");
        break;
    case CL_COMPILER_NOT_AVAILABLE:
        printf("Compiler not available.\n");
        break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        printf("Memory object allocation failed.\n");
        break;
    case CL_OUT_OF_RESOURCES:
        printf("Out of resource.\n");
        break;
    case CL_OUT_OF_HOST_MEMORY:
        printf("Out of host memory.\n");
        break;
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        printf("Profiling information not available.\n");
        break;
    case CL_MEM_COPY_OVERLAP:
        printf("Memory copy overlap.\n");
        break;
    case CL_IMAGE_FORMAT_MISMATCH:
        printf("Image format mismatch.\n");
        break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        printf("Image format not supported.\n");
        break;
    case CL_BUILD_PROGRAM_FAILURE:
        printf("Build program failed.\n");
        break;
    case CL_MAP_FAILURE:
        printf("Mapping failed.\n");
        break;
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        printf("Misaligned sub-buffer offset.\n");
        break;
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        printf("Execution status error for events in wait list.\n");
        break;
    case CL_COMPILE_PROGRAM_FAILURE:
        printf("Compile program failed.\n");
        break;
    case CL_LINKER_NOT_AVAILABLE:
        printf("Linker not available.\n");
        break;
    case CL_LINK_PROGRAM_FAILURE:
        printf("Link program failed.\n");
        break;
    case CL_DEVICE_PARTITION_FAILED:
        printf("Device partition failed.\n");
        break;
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        printf("Kernel argument information not available.\n");
        break;
    case CL_INVALID_VALUE:
        printf("Invalid value.\n");
        break;
    case CL_INVALID_DEVICE_TYPE:
        printf("Invalid device type.\n");
        break;
    case CL_INVALID_PLATFORM:
        printf("Invalid platform.\n");
        break;
    case CL_INVALID_DEVICE:
        printf("Invalid device.\n");
        break;
    case CL_INVALID_CONTEXT:
        printf("Invalid context.\n");
        break;
    case CL_INVALID_QUEUE_PROPERTIES:
        printf("Invalid queue properties.\n");
        break;
    case CL_INVALID_COMMAND_QUEUE:
        printf("Invalid command queue.\n");
        break;
    case CL_INVALID_HOST_PTR:
        printf("Invalid host pointer.\n");
        break;
    case CL_INVALID_MEM_OBJECT:
        printf("Invalid memory object.\n");
        break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        printf("Invalid image format descriptor.\n");
        break;
    case CL_INVALID_IMAGE_SIZE:
        printf("Invalid image size.\n");
        break;
    case CL_INVALID_SAMPLER:
        printf("Invalid sampler.\n");
        break;
    case CL_INVALID_BINARY:
        printf("Invalid binary.\n");
        break;
    case CL_INVALID_BUILD_OPTIONS:
        printf("Invalid build options.\n");
        break;
    case CL_INVALID_PROGRAM:
        printf("Invalid program.\n");
        break;
    case CL_INVALID_PROGRAM_EXECUTABLE:
        printf("Invalid program executable.\n");
        break;
    case CL_INVALID_KERNEL_NAME:
        printf("Invalid kernel name.\n");
        break;
    case CL_INVALID_KERNEL_DEFINITION:
        printf("Invalid kernel definition.\n");
        break;
    case CL_INVALID_KERNEL:
        printf("Invalid kernel.\n");
        break;
    case CL_INVALID_ARG_INDEX:
        printf("Invalid argument index.\n");
        break;
    case CL_INVALID_ARG_VALUE:
        printf("Invalid argument value.\n");
        break;
    case CL_INVALID_ARG_SIZE:
        printf("Invalid argument size.\n");
        break;
    case CL_INVALID_KERNEL_ARGS:
        printf("Invalid kernel arguments.\n");
        break;
    case CL_INVALID_WORK_DIMENSION:
        printf("Invalid work dimension.\n");
        break;
    case CL_INVALID_WORK_GROUP_SIZE:
        printf("Invalid work group size.\n");
        break;
    case CL_INVALID_WORK_ITEM_SIZE:
        printf("Invalid work time size.\n");
        break;
    case CL_INVALID_GLOBAL_OFFSET:
        printf("Invalid global offset.\n");
        break;
    case CL_INVALID_EVENT_WAIT_LIST:
        printf("Invalid event wait list.\n");
        break;
    case CL_INVALID_EVENT:
        printf("Invalid event.\n");
        break;
    case CL_INVALID_OPERATION:
        printf("Invalid operation.\n");
        break;
    case CL_INVALID_GL_OBJECT:
        printf("Invalid GL object.\n");
        break;
    case CL_INVALID_BUFFER_SIZE:
        printf("Invalid buffer size.\n");
        break;
    case CL_INVALID_MIP_LEVEL:
        printf("Invalid map level.\n");
        break;
    case CL_INVALID_GLOBAL_WORK_SIZE:
        printf("Invalid global work size.\n");
        break;
    case CL_INVALID_PROPERTY:
        printf("Invalid property.\n");
        break;
    case CL_INVALID_IMAGE_DESCRIPTOR:
        printf("Invalid image descriptor.\n");
        break;
    case CL_INVALID_COMPILER_OPTIONS:
        printf("Invalid compiler options.\n");
        break;
    case CL_INVALID_LINKER_OPTIONS:
        printf("Invalid linker options.\n");
        break;
    case CL_INVALID_DEVICE_PARTITION_COUNT:
        printf("Invalid device partition count.\n");
        break;
    default:
        printf("Unknown error.\n");
        break;
    }
}
