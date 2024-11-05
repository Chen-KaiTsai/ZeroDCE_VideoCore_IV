/*
 * @author Chen-Kai Tsai
 * @version 0.1
 * @brief First porting to OpenCL
 * @note Notice that this ported version should be able to run on Raspberry Pi 3 with VideoCore IV
*/

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <chrono>

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <cmath>

#include <omp.h>

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define  CL_TARGET_OPENCL_VERSION 120

// OpenCL Header
#include <CL/cl.h>

// OpenCV Header
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
//#include <opencv2/core/utils/logger.hpp>

using std::unique_ptr;
using std::make_unique;

constexpr unsigned int DSRATE = 12;
constexpr unsigned int IMG_HEIGHT = 1080;
constexpr unsigned int IMG_WIDTH = 1920;
constexpr unsigned int IMG_CHANNEL = 3;
constexpr unsigned int DCE_HEIGHT = 1080 / DSRATE;
constexpr unsigned int DCE_WIDTH = 1920 / DSRATE;
constexpr unsigned int DCE_CHANNEL = 32;

#define PLATFORM_ID 0
#define DEVICE_ID 0
#define NUM_DEVICE 1

#define QX 16384
#define QW 16384
#define QB (QX * QW)
#define QI 1024
#define QA 1024

#define NTHREAD 3

extern cl_platform_id dPlatform;
extern cl_device_id  dDevice;
extern cl_context dContext;
extern cl_command_queue dQueue;
extern cl_program dProgram;

// OpenCL Wrapper Functions ONLY provided as default use. Please fall back to OpenCL functions for advanced usage.
namespace ocl
{
	unique_ptr<cl_platform_id[]> getPlatformID(void);
	unique_ptr<cl_device_id[]> getDeviceID(const cl_platform_id platform, const cl_device_type type);
	/*
	* @brief [WARNING] This only support infomation that is returned in string format [TODO].
	*/
	unique_ptr<char[]> getPlatformInfo(const cl_platform_id platform, const cl_platform_info info);
	/*
	* @brief [WARNING] This only support infomation that is returned in string format [TODO].
	*/
	unique_ptr<char[]> getDeviceInfo(const cl_device_id device, const cl_device_info info);

	cl_context createContext(cl_platform_id platform, cl_device_id device);
	cl_command_queue createQueue(cl_context context, cl_device_id device);
	cl_program createProgramFromSource(cl_context context, const std::vector<const char*> fileNames);
	cl_kernel createKernel(cl_program program, const char* kernelName);

	void buildProgram(cl_program program, cl_device_id device, const char* option);
	
	cl_mem createBuffer(cl_context context, cl_mem_flags flags, size_t size, void* buffer);
	/*
	* @brief [WARNING] Launch one kernel and wait for kernel to finish.
	*/
	void launchOneKernelAndWait(cl_command_queue dQueue, cl_kernel dKernel, cl_int dim, const size_t* gws, const size_t* lws);
	void launchOneKernelAndProfile(cl_command_queue dQueue, cl_kernel dKernel, cl_int dim, const size_t* gws, const size_t* lws);

	void readBufferBlockNoOffset(cl_command_queue dQueue, cl_mem dBuffer, size_t size, void* hBuffer);

	void getErrMsg(cl_int error);
}

namespace DCE
{
    void initOpenCL();
    void initMem();    
    void cleanMem();
    void qNormNDownSample();
    void qConv1st();
    void qConv2nd();
    void qConv3rd();
    void qUpSample();
    void qEnhance();
}

namespace cvf
{
    void cvReadImg(char* filename);
    void cvOutputImg(char* filename);
}

void loadWeight();

/**
 * Grouping Data to Structure for Readability
 */

using RGBIOData_t = struct RGBIOData
{
    uint8_t data[IMG_HEIGHT * IMG_WIDTH * IMG_CHANNEL];
};

using qNormImg_t = struct qNormImg
{
    short data[IMG_HEIGHT * IMG_WIDTH * IMG_CHANNEL];
};

using qNetIO_t = struct qNetIO
{
    short data[DCE_HEIGHT * DCE_WIDTH * IMG_CHANNEL];
};

using qNetFeature_t = struct qNetFeature
{
    short data[DCE_HEIGHT * DCE_WIDTH * DCE_CHANNEL];
};

using qEnhancedParam_t = struct qEnhancedParam
{
    short data[IMG_HEIGHT * IMG_WIDTH * IMG_CHANNEL];
};

using qWConv1st_t = struct qWConv1st
{
    short data[DCE_CHANNEL * IMG_CHANNEL * 3 * 3];
};
using qBConv1st_t = struct qBConv1st
{
    int data[DCE_CHANNEL];
};

using qWConv2nd_t =  struct qWConv2nd
{
    short data[DCE_CHANNEL * DCE_CHANNEL * 3 * 3];
};
using qBConv2nd_t = struct qBConv2nd
{
    int data[DCE_CHANNEL];
};

using qWConv3rd_t = struct qWConv3rd
{
    short data[IMG_CHANNEL * DCE_CHANNEL * 3 * 3];
};
using qBConv3rd_t = struct qBConv3rd
{
    int data[IMG_CHANNEL];
};

/**
 * Initialize all data space
 */

extern RGBIOData_t* INDATA;
extern cl_mem dINDATA;

extern RGBIOData_t* OUTDATA;
extern cl_mem dOUTDATA;

extern cl_mem dNORM;

extern qNetIO_t* NETIO;
extern cl_mem dNETIO;

extern cl_mem dFEATURE1;
extern cl_mem dFEATURE2;

extern qEnhancedParam_t* PARAM;
extern cl_mem dPARAM;
extern qEnhancedParam_t* UPSBUFFER;
extern cl_mem dUPSBUFFER;

extern qWConv1st_t* CONVW01;
extern qBConv1st_t* CONVB01;
extern qWConv2nd_t* CONVW02;
extern qBConv2nd_t* CONVB02;
extern qWConv3rd_t* CONVW03;
extern qBConv3rd_t* CONVB03;

extern cl_mem dCONVW01;
extern cl_mem dCONVB01;
extern cl_mem dCONVW02;
extern cl_mem dCONVB02;
extern cl_mem dCONVW03;
extern cl_mem dCONVB03;
