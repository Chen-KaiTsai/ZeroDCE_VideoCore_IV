#include "main.h"
#include "qWeight.h"

RGBIOData_t* INDATA = nullptr;
cl_mem dINDATA = nullptr;

RGBIOData_t* OUTDATA = nullptr;
cl_mem dOUTDATA = nullptr;

cl_mem dNORM = nullptr;

qNetIO_t* NETIO = nullptr;
cl_mem dNETIO = nullptr;

qNetFeature_t* dFEATURE1 = nullptr;
cl_mem dFEATURE2 = nullptr;

qEnhancedParam_t* PARAM = nullptr;
cl_mem dPARAM = nullptr;

qEnhancedParam_t* UPSBUFFER = nullptr;
cl_mem dUPSBUFFER = nullptr;

qWConv1st_t* CONVW01 = nullptr;
qBConv1st_t* CONVB01 = nullptr;
qWConv2nd_t* CONVW02 = nullptr;
qBConv2nd_t* CONVB02 = nullptr;
qWConv3rd_t* CONVW03 = nullptr;
qBConv3rd_t* CONVB03 = nullptr;

cl_mem dCONVW01 = nullptr;
cl_mem dCONVB01 = nullptr;
cl_mem dCONVW02 = nullptr;
cl_mem dCONVB02 = nullptr;
cl_mem dCONVW03 = nullptr;
cl_mem dCONVB03 = nullptr;

cl_platform_id dPlatform = nullptr;
cl_device_id  dDevice = nullptr;
cl_context dContext = nullptr;
cl_command_queue dQueue = nullptr;
cl_program dProgram = nullptr;

void DCE::initOpenCL() 
{
    if (dPlatform == nullptr) {
        auto dPlatforms = ocl::getPlatformID();
        dPlatform = dPlatforms[PLATFORM_ID];
        auto dPlatformName = ocl::getPlatformInfo(dPlatform, CL_PLATFORM_NAME);
        printf("Platform Name : %s\n\n", dPlatformName.get());
    }
    if (dDevice == nullptr) {
        auto dDevices = ocl::getDeviceID(dPlatform, CL_DEVICE_TYPE_GPU);
        dDevice = dDevices[DEVICE_ID];
	    auto dDeviceName = ocl::getDeviceInfo(dDevice, CL_DEVICE_NAME);
	    printf("Device Name : %s\n\n", dDeviceName.get());
    }
    if (dContext == nullptr) {
        dContext = ocl::createContext(dPlatform, dDevice);
    }
    if (dQueue == nullptr) {
        dQueue = ocl::createQueue(dContext, dDevice);
    }
    if (dProgram == nullptr) {
        cl_program dProgram = ocl::createProgramFromSource(dContext, { "qKernel.cl" });
    }

    printf("Start compile OpenCL kernels...\n\n");
    const char options[] = "-cl-fast-relaxed-math";
    ocl::buildProgram(dProgram, dDevice, options);
    printf("Finish compiling\n\n");

    return;
}

void loadWeight() {
    memcpy((void*)CONVW01, conv1_w, 2 * 864);
    memcpy((void*)CONVB01, conv1_b, 4 * 32);
    memcpy((void*)CONVW02, conv2_w, 2 * 9216);
    memcpy((void*)CONVB02, conv2_b, 4 * 32);
    memcpy((void*)CONVW03, conv3_w, 2 * 864);
    memcpy((void*)CONVB03, conv3_b, 4 * 3);
}

void cvf::cvReadImg(char* filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    #pragma omp parallel for schedule(guided) num_threads(NTHREAD)
    for (int c = 0; c < IMG_CHANNEL; c++) {
        for (int y = 0; y < IMG_HEIGHT; ++y) {
            for(int x = 0; x < IMG_WIDTH; ++x) {
                INDATA->data[y * IMG_WIDTH * IMG_CHANNEL + x * IMG_CHANNEL + c] = image.at<cv::Vec3b>(y, x)[c];
            }
        }
    }

    //cv::waitKey(0);
}

void cvf::cvOutputImg(char* filename) {
    cv::Mat image(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);

    #pragma omp parallel for schedule(guided) num_threads(NTHREAD)
    for (int c = 0; c < IMG_CHANNEL; c++)
        for (int y = 0; y < IMG_HEIGHT; ++y)
            for (int x = 0; x < IMG_WIDTH; ++x) {
                image.at<cv::Vec3b>(y, x)[c] = OUTDATA->data[y * IMG_WIDTH * IMG_CHANNEL + x * IMG_CHANNEL + c];
            }
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, image);

    //cv::waitKey(0);
}
