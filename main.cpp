#include "main.h"
#include <chrono>

int main(int argc, char** argv)
{
    auto start = std::chrono::steady_clock::now();
    DCE::initMem();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("initMem Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    loadWeight();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("loadWeight Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    cvf::cvReadImg("testInput.png");
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("cvReadImg Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE::qNormNDownSample();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qNormNDownSample Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE::qConv1st();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qConv1st Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE::qConv2nd();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qConv2nd Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE::qConv3rd();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qConv3rd Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE::qUpSample();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qUpSample Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE::qEnhance();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qEnhance Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    cvf::cvOutputImg("Enhanced_CPP_output.png");
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("cvOutputImg Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    DCE::cleanMem();
    
    return 0;
}
