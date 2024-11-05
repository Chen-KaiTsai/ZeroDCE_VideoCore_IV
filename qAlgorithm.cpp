#include "main.h"

void DCE::initMem() {
    INDATA = (RGBIOData*)malloc(sizeof(RGBIOData_t));
    
    OUTDATA = (RGBIOData*)malloc(sizeof(RGBIOData_t));

    NETIO = (qNetIO_t*)malloc(sizeof(qNetIO_t));

    CONVW01 = (qWConv1st_t*)malloc(sizeof(qWConv1st_t));
    CONVB01 = (qBConv1st_t*)malloc(sizeof(qBConv1st_t));
    CONVW02 = (qWConv2nd_t*)malloc(sizeof(qWConv2nd_t));
    CONVB02 = (qBConv2nd_t*)malloc(sizeof(qBConv2nd_t));
    CONVW03 = (qWConv3rd_t*)malloc(sizeof(qWConv3rd_t));
    CONVB03 = (qBConv3rd_t*)malloc(sizeof(qBConv3rd_t));
}

void DCE::cleanMem() {
    if (INDATA != nullptr) {
        free(INDATA);
        INDATA = nullptr;
    }
    if (OUTDATA != nullptr) {
        free(OUTDATA);
        OUTDATA = nullptr;
    }
    if (NETIO != nullptr) {
        free(NETIO);
        NETIO = nullptr;
    }

    if (CONVW01 != nullptr) {
        free(CONVW01);
        CONVW01 = nullptr;
    }
    if (CONVB01 != nullptr) {
        free(CONVB01);
        CONVB01 = nullptr;
    }
    if (CONVW02 != nullptr) {
        free(CONVW02);
        CONVW02 = nullptr;
    }
    if (CONVB02 != nullptr) {
        free(CONVB02);
        CONVB02 = nullptr;
    }
    if (CONVW03 != nullptr) {
        free(CONVW03);
        CONVW03 = nullptr;
    }
    if (CONVB03 != nullptr) {
        free(CONVB03);
        CONVB03 = nullptr;
    }
}

void DCE::qNormNDownSample() {
    cl_kernel kNorm = ocl::createKernel(dProgram, "dNorm");

    // Allocate & Copy dINDATA dNORM

    dINDATA = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(RGBIOData), (void*)INDATA->data);
    dNORM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS , sizeof(qNormImg_t), nullptr);

    // Run kNorm

    clSetKernelArg(kNorm, 0, sizeof(cl_mem), static_cast<void*>(dINDATA));
    clSetKernelArg(kNorm, 1, sizeof(cl_mem), static_cast<void*>(dNORM));

    size_t gws[3]{IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL};
    size_t lws[3]{12, 12, 3};

    ocl::launchOneKernelAndWait(dQueue, kNorm, 3, gws, lws);

    clReleaseMemObject(dINDATA);
    dINDATA = nullptr;
    
    cl_kernel kDownSample = ocl::createKernel(dProgram, "dDownSample");

    // Allocate & Copy dNETIO

    dNETIO = ocl::createBuffer(dContext, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(qNetIO_t), nullptr);

    // Run kDownSample

    clSetKernelArg(kDownSample, 0, sizeof(cl_mem), static_cast<void*>(dNORM));
    clSetKernelArg(kDownSample, 1, sizeof(cl_mem), static_cast<void*>(dNETIO));

    size_t gws[3]{DCE_HEIGHT, DCE_WIDTH, 1};
    size_t lws[3]{10, 10, 1};

    ocl::launchOneKernelAndWait(dQueue, kDownSample, 2, gws, lws);
}

void DCE::qConv1st() {
    cl_kernel kConv1st = ocl::createKernel(dProgram, "kConv1st");

    // Allocate & Copy CONVW01 CONVB01

    dCONVW01 = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qWConv1st_t), CONVW01->data);
    dCONVB01 = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qBConv1st_t), CONVB01->data);

    // Allocate FEATURE1

    dFEATURE1 = ocl::createBuffer(dContext, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(qNetFeature_t), nullptr);

    // Run kConv1st

    clSetKernelArg(kConv1st, 0, sizeof(cl_mem), static_cast<void*>(dNETIO));
    clSetKernelArg(kConv1st, 1, sizeof(cl_mem), static_cast<void*>(dCONVW01));
    clSetKernelArg(kConv1st, 2, sizeof(cl_mem), static_cast<void*>(dCONVB01));
    clSetKernelArg(kConv1st, 3, sizeof(cl_mem), static_cast<void*>(dFEATURE1));

    size_t gws[3]{DCE_HEIGHT, DCE_WIDTH, 1};
    size_t lws[3]{10, 10, 1};

    ocl::launchOneKernelAndWait(dQueue, kConv1st, 2, gws, lws);

    // Free CONVW01 CONVB01

    if (CONVW01 != nullptr) {
        free(CONVW01);
        CONVW01 = nullptr;
    }

    if (CONVB01 != nullptr) {
        free(CONVB01);
        CONVB01 = nullptr;
    }

    // Free dCONVW01, dCONVB01

    clReleaseMemObject(dCONVW01);
    dCONVW01 = nullptr;
    clReleaseMemObject(dCONVB01);
    dCONVB01 = nullptr;
}

void DCE::qConv2nd() {
    cl_kernel kConv2nd = ocl::createKernel(dProgram, "kConv2nd");

    // Allocate & Copy CONVW02 CONVB02

    dCONVW02 = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qWConv2nd_t), CONVW02->data);
    dCONVB02 = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qBConv2nd_t), CONVB02->data);

    // Allocate FEATURE2

    dFEATURE2 = ocl::createBuffer(dContext, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(qNetFeature_t), nullptr);

    // Run kConv2nd

    clSetKernelArg(kConv2nd, 0, sizeof(cl_mem), static_cast<void*>(dFEATURE1));
    clSetKernelArg(kConv2nd, 1, sizeof(cl_mem), static_cast<void*>(dCONVW02));
    clSetKernelArg(kConv2nd, 2, sizeof(cl_mem), static_cast<void*>(dCONVB02));
    clSetKernelArg(kConv2nd, 3, sizeof(cl_mem), static_cast<void*>(dFEATURE2));

    size_t gws[3]{DCE_HEIGHT, DCE_WIDTH, 1};
    size_t lws[3]{10, 10, 1};

    ocl::launchOneKernelAndWait(dQueue, kConv2nd, 2, gws, lws);

    // Free CONVW02 CONVB02

    if (CONVW02 != nullptr) {
        free(CONVW02);
        CONVW02 = nullptr;
    }

    if (CONVB02 != nullptr) {
        free(CONVB02);
        CONVB02 = nullptr;
    }

    // Free dCONVW02, dCONVB02

    clReleaseMemObject(dCONVW02);
    dCONVW02 = nullptr;
    clReleaseMemObject(dCONVB02);
    dCONVB02 = nullptr;
}

void DCE::qConv3rd() {
    cl_kernel kConv3rd = ocl::createKernel(dProgram, "kConv3rd");

    // Allocate & Copy CONVW03 CONVB03

    dCONVW03 = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qWConv3rd_t), CONVW03->data);
    dCONVB03 = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qBConv3rd_t), CONVB03->data);

    // Run kConv3rd

    clSetKernelArg(kConv3rd, 0, sizeof(cl_mem), static_cast<void*>(dFEATURE1));
    clSetKernelArg(kConv3rd, 1, sizeof(cl_mem), static_cast<void*>(dFEATURE2));
    clSetKernelArg(kConv3rd, 2, sizeof(cl_mem), static_cast<void*>(dCONVW03));
    clSetKernelArg(kConv3rd, 3, sizeof(cl_mem), static_cast<void*>(dCONVB03));
    clSetKernelArg(kConv3rd, 4, sizeof(cl_mem), static_cast<void*>(dNETIO));

    size_t gws[3]{DCE_HEIGHT, DCE_WIDTH, 1};
    size_t lws[3]{10, 10, 1};

    ocl::launchOneKernelAndWait(dQueue, kConv3rd, 2, gws, lws);

    // Free CONVW03 CONVB03

    if (CONVW03 != nullptr) {
        free(CONVW03);
        CONVW03 = nullptr;
    }

    if (CONVB03 != nullptr) {
        free(CONVB03);
        CONVB03 = nullptr;
    }

    // Free dCONVW03, dCONVB03

    clReleaseMemObject(dCONVW03);
    dCONVW03 = nullptr;
    clReleaseMemObject(dCONVB03);
    dCONVB03 = nullptr;

    clReleaseMemObject(dFEATURE1);
    dFEATURE1 = nullptr;
    clReleaseMemObject(dFEATURE2);
    dFEATURE2 = nullptr;
}

void DCE::qUpSample() {
    cl_kernel kUpSample_x = ocl::createKernel(dProgram, "kUpSample_x");
    cl_kernel kUpSample_y = ocl::createKernel(dProgram, "kUpSample_y");
    
    // Allocate dUPSBUFFER, dPARAM

    dUPSBUFFER = ocl::createBuffer(dContext, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(qEnhancedParam_t), nullptr);
    dPARAM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(qEnhancedParam_t), nullptr);

    // Run kUpSample_x

    clSetKernelArg(kUpSample_x, 0, sizeof(cl_mem), static_cast<void*>(dNETIO));
    clSetKernelArg(kUpSample_x, 1, sizeof(cl_mem), static_cast<void*>(dUPSBUFFER));
    clSetKernelArg(kUpSample_x, 2, sizeof(short) * 12, nullptr);
    
    size_t gws[3]{DCE_HEIGHT, IMG_WIDTH, 1};
    size_t lws[3]{10, 12, 1};

    ocl::launchOneKernelAndWait(dQueue, kUpSample_x, 2, gws, lws);

    // Run kUpSample_y
    clSetKernelArg(kUpSample_x, 0, sizeof(cl_mem), static_cast<void*>(dUPSBUFFER));
    clSetKernelArg(kUpSample_x, 1, sizeof(cl_mem), static_cast<void*>(dPARAM));
    clSetKernelArg(kUpSample_x, 2, sizeof(short) * 12, nullptr);
    
    size_t gws[3]{IMG_HEIGHT, IMG_WIDTH, 1};
    size_t lws[3]{24, 30, 1};

    ocl::launchOneKernelAndWait(dQueue, kUpSample_x, 2, gws, lws);

    clReleaseMemObject(dUPSBUFFER);
    dUPSBUFFER = nullptr;
}

void DCE::qEnhance() {
    cl_kernel kEnhance = ocl::createKernel(dProgram, "kEnhance");

    clSetKernelArg(kEnhance, 0, sizeof(cl_mem), static_cast<void*>(dNORM));
    clSetKernelArg(kEnhance, 1, sizeof(cl_mem), static_cast<void*>(dPARAM));
    clSetKernelArg(kEnhance, 2, sizeof(cl_mem), static_cast<void*>(dOUTDATA));

    size_t gws[3]{IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL};
    size_t lws[3]{12, 12, 3};

    ocl::launchOneKernelAndWait(dQueue, kEnhance, 3, gws, lws);

    ocl::readBufferBlockNoOffset(dQueue, dOUTDATA, sizeof(RGBIOData_t), OUTDATA);

    clReleaseMemObject(dOUTDATA);
    dOUTDATA = nullptr;
    clReleaseMemObject(dPARAM);
    dPARAM = nullptr;
    clReleaseMemObject(dNORM);
    dNORM = nullptr;

    if (PARAM != nullptr) {
        free(PARAM);
        PARAM = nullptr;
    }
}
