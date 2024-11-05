#define NULL 0

#define DSRATE = 12;
#define IMG_HEIGHT = 1080;
#define IMG_WIDTH = 1920;
#define IMG_CHANNEL = 3;
#define DCE_HEIGHT = 90;
#define DCE_WIDTH = 160;
#define DCE_CHANNEL = 32;

__kernel void kNorm(__global uint8* dINDATA, __global short* dNORM)
{
    unsigned int globalIdx_x = get_global_id(0); // IMG_HEIGHT
    unsigned int globalIdx_y = get_global_id(1); // IMG_WIDTH
    unsigned int globalIdx_z = get_global_id(2); // IMG_CHANNEL

    unsigned int idx = globalIdx_x * IMG_WIDTH * IMG_CHANNEL + globalIdx_y * IMG_CHANNEL + globalIdx_z;
    dNORM[idx] = (short)dINDATA[idx];
}

__kernel void kDownSample(__global short* dNORM, __global short* dNETIO)
{
    unsigned int globalIdx_x = get_global_id(0); // DCE_HEIGHT
    unsigned int globalIdx_y = get_global_id(1); // DCE_WIDTH

    unsigned int idx = globalIdx_x * DCE_WIDTH * IMG_CHANNEL + globalIdx_y * IMG_CHANNEL;
    unsigned int idx_DownSample = (globalIdx_x * DSRATE) * DCE_WIDTH * IMG_CHANNEL + (globalIdx_y * DSRATE) * IMG_CHANNEL;
    dNETIO[idx] = dNORM[idx_DownSample];
    dNETIO[idx + 1] = dNORM[idx_DownSample + 1];
    dNETIO[idx + 2] = dNORM[idx_DownSample + 2];
}

__kernel void kConv1st(__global short* dNETIO, __global short* dCONVW01, __global int* dCONVB01, __global short* dFEATURE1) 
{
    int h = get_global_id(0);
    int w = get_global_id(1);

    int sum;
    for (int cout = 0; cout < DCE_CHANNEL; ++cout) {
        sum = 0;
        for (int cin = 0; cin < IMG_CHANNEL; ++cin) {
            for (int kh = -1; kh <= 1; ++kh)
                for(int kw = -1; kw <= 1; ++kw) {
                    if((unsigned int)(h + kw) < DCE_HEIGHT && (unsigned int)(w + kw) < DCE_WIDTH) {
                        sum += dNETIO[(h + kw) * DCE_WIDTH * DCE_CHANNEL + (w + kw) * DCE_CHANNEL + cin] * dCONVW01[cout * IMG_CHANNEL * 9 + cin * 9 + (kh + 1) * 3 + (kw + 1)];
                    }
                }
        }
        sum += dCONVB01[cout];
        sum = max(0, sum);
        dFEATURE1[h * DCE_WIDTH * DCE_CHANNEL + w * DCE_CHANNEL + cout] = sum >> 14;
    }
}

__kernel void kConv2nd(__global short* dFEATURE1, __global short* dCONVW02, __global int* dCONVB02, dFEATURE2)
{
    int h = get_global_id(0);
    int w = get_global_id(1);

    int sum;
    for (int cout = 0; cout < DCE_CHANNEL; ++cout) {
        sum = 0;
        for (int cin = 0; cin < DCE_CHANNEL; ++cin) {
            for (int kh = -1; kh <= 1; ++kh)
                for(int kw = -1; kw <= 1; ++kw) {
                    if((unsigned int)(h + kw) < DCE_HEIGHT && (unsigned int)(w + kw) < DCE_WIDTH) {
                        sum += dFEATURE1[(h + kw) * DCE_WIDTH * DCE_CHANNEL + (w + kw) * DCE_CHANNEL + cin] * dCONVW02[cout * IMG_CHANNEL * 9 + cin * 9 + (kh + 1) * 3 + (kw + 1)];
                    }
                }
        }
        sum += dCONVB02[cout];
        sum = max(0, sum);
        dFEATURE2[h * DCE_WIDTH * DCE_CHANNEL + w * DCE_CHANNEL + cout] = sum >> 14;
    }
}

int sigmoidMapping(int x)
{
    if (x >= -1 * QX && x <= QX)
        return (x * 7810 + 195) >> 18;
    else if ((x >= -2 * QX && x < -1 * QX) || (x <= 2 * QX && x > QX))
        return x > 0 ? ((x * 4899 + 47996260) >> 18) : ((x * 4899 - 47996260) >> 18);
    else if ((x >= -3 * QX && x < -2 * QX) || (x <= 3 * QX && x > 2 * QX))
        return x > 0 ? ((x * 2330 + 130915972) >> 18) : ((x * 2330 - 130915972) >> 18);
    else if ((x >= -4 * QX && x < -3 * QX) || (x <= 4 * QX && x > 3 * QX))
        return x > 0 ? ((x * 952 + 197514809) >> 18) : ((x * 952 - 197514809) >> 18);
    else if ((x >= -5 * QX && x < -4 * QX) || (x <= 5 * QX && x > 4 * QX))
        return x > 0 ? ((x * 364 + 235417895) >> 18) : ((x * 364 - 235417895) >> 18);
    else
        return (x > 0 ? QA : -QA);
}

__kernel void kConv3rd(__global short* dFEATURE1, __global short* FEATURE2, __global short* dCONVW03, __global int* dCONVB03, __global short* dNETIO)
{
    int h = get_global_id(0);
    int w = get_global_id(1);

    int sum;
    unsigned int fIdx;
    for (int cout = 0; cout < IMG_CHANNEL; ++cout) {
        sum = 0;
        for (int cin = 0; cin < DCE_CHANNEL; ++cin) {
            for (int kh = -1; kh <= 1; ++kh)
                for(int kw = -1; kw <= 1; ++kw) {
                    if((unsigned int)(h + kw) < DCE_HEIGHT && (unsigned int)(w + kw) < DCE_WIDTH) {
                        fIdx = (h + kw) * DCE_WIDTH * DCE_CHANNEL + (w + kw) * DCE_CHANNEL + cin;
                        sum += (dFEATURE1[fIdx] + dFEATURE2[fIdx]) * dCONVW03[cout * IMG_CHANNEL * 9 + cin * 9 + (kh + 1) * 3 + (kw + 1)];
                    }
                }
        }
        sum += dCONVB03[cout];
        dNETIO[h * DCE_WIDTH * DCE_CHANNEL + w * DCE_CHANNEL + cout] = (short)sigmoidMapping(sum >> 14);
    }
}

__kernel void kUpSample_x(__global short* dNETIO, __global short* dUPSBUFFER, __local short* coef) // TODO modified entire program to fit into this modification
{
    int h = get_global_id(0);
    int w = get_global_id(1);
    
    int lh = get_local_id(0);
    int lw = get_local_id(1);

    int pad = DSRATE / 2;

    if (lh == 0 && lw == 0) {
        coef[0] = 42;
        coef[1] = 128;
        coef[2] = 213;
        coef[3] = 298;
        coef[4] = 384;
        coef[5] = 469;
        coef[6] = 554;
        coef[7] = 640;
        coef[8] = 725;
        coef[9] = 810;
        coef[10] = 896;
        coef[11] = 981;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (w < pad) {
        unsigned int UPSbufferIdx = h * IMG_WIDTH * IMG_CHANNEL + w * IMG_CHANNEL;
        unsigned int NetIOIdx = h * IMG_WIDTH * IMG_CHANNEL;
        dUPSBUFFER[UPSbufferIdx] = dNETIO[NetIOIdx];
        dUPSBUFFER[UPSbufferIdx + 1] = dNETIO[NetIOIdx + 1];
        dUPSBUFFER[UPSbufferIdx + 2] = dNETIO[NetIOIdx + 2];
    }
    else if (w >= (IMG_WIDTH - pad)) {
        unsigned int UPSbufferIdx = h * IMG_WIDTH * IMG_CHANNEL + w * IMG_CHANNEL;
        unsigned int NetIOIdx = h * IMG_WIDTH * IMG_CHANNEL + (DCE_WIDTH - 1) * IMG_CHANNEL;
		dUPSBUFFER[UPSbufferIdx] = dNETIO[NetIOIdx];
		dUPSBUFFER[UPSbufferIdx + 1] = dNETIO[NetIOIdx + 1];
		dUPSBUFFER[UPSbufferIdx + 2] = dNETIO[NetIOIdx + 2];
    }
    else {
        int d = (w - pad) % DSRATE;
        int wi = (w - pad) / DSRATE + 1;

        unsigned int UPSbufferIdx = h * IMG_WIDTH * IMG_CHANNEL + w * IMG_CHANNEL;
        unsigned int NetIOIdx = h * IMG_WIDTH * IMG_CHANNEL + wi * IMG_CHANNEL;
        unsigned int NetIOIdx_prv = h * IMG_WIDTH * IMG_CHANNEL + (wi - 1) * IMG_CHANNEL;

        dUPSBUFFER[UPSbufferIdx] = (coef[d] * (dNETIO->data[NetIOIdx] - dNETIO[NetIOIdx_prv]) >> 10) + dNETIO[NetIOIdx_prv];
		dUPSBUFFER[UPSbufferIdx + 1] = (coef[d] * (dNETIO->data[NetIOIdx + 1] - dNETIO[NetIOIdx_prv + 1]) >> 10) + dNETIO[NetIOIdx_prv + 1];
		dUPSBUFFER[UPSbufferIdx + 2] = (coef[d] * (dNETIO->data[NetIOIdx + 2] - dNETIO[NetIOIdx_prv + 2]) >> 10) + dNETIO[NetIOIdx_prv + 2];
    }
}

__kernel void kUpSample_y(__global short* dUPSBUFFER, __global short* dPARAM, __local short* coef) // TODO modified entire program to fit into this modification
{
    int h = get_global_id(0);
    int w = get_global_id(1);
    
    int lh = get_local_id(0);
    int lw = get_local_id(1);

    int pad = DSRATE / 2;

    if (lh == 0 && lw == 0) {
        coef[0] = 42;
        coef[1] = 128;
        coef[2] = 213;
        coef[3] = 298;
        coef[4] = 384;
        coef[5] = 469;
        coef[6] = 554;
        coef[7] = 640;
        coef[8] = 725;
        coef[9] = 810;
        coef[10] = 896;
        coef[11] = 981;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (h < pad) {
        unsigned int ParamIdx = h * IMG_WIDTH * IMG_CHANNEL + w * IMG_CHANNEL;
        unsigned int UPSbufferIdx = w * IMG_CHANNEL;
        dPARAM[ParamIdx] = dUPSBUFFER[UPSbufferIdx];
        dPARAM[ParamIdx + 1] = dUPSBUFFER[UPSbufferIdx + 1];
        dPARAM[ParamIdx + 2] = dUPSBUFFER[UPSbufferIdx + 2];
    }
    else if (h >= (IMG_HEIGHT - pad)) {
        unsigned int ParamIdx = h * IMG_WIDTH * IMG_CHANNEL + w * IMG_CHANNEL;
        unsigned int UPSbufferIdx = (DCE_HEIGHT - 1) * IMG_WIDTH * IMG_CHANNEL + w * IMG_CHANNEL;
		dPARAM[ParamIdx] = dUPSBUFFER[UPSbufferIdx];
		dPARAM[ParamIdx + 1] = dUPSBUFFER[UPSbufferIdx + 1];
		dPARAM[ParamIdx + 2] = dUPSBUFFER[UPSbufferIdx + 2];
    }
    else {
        int d = (h - pad) % DSRATE;
        int hi = (h - pad) / DSRATE + 1;

        unsigned int ParamIdx = h * IMG_WIDTH * IMG_CHANNEL + w * IMG_CHANNEL;
        unsigned int UPSbufferIdx = hi * IMG_WIDTH * IMG_CHANNEL + w * IMG_CHANNEL;
        unsigned int UPSbufferIdx_prv = (hi - 1) * IMG_WIDTH * IMG_CHANNEL + w * IMG_CHANNEL;

        dPARAM[ParamIdx] = (coef[d] * (dUPSBUFFER[UPSbufferIdx] - dUPSBUFFER[UPSbufferIdx_prv]) >> 10) + dUPSBUFFER[UPSbufferIdx_prv];
		dPARAM[ParamIdx + 1] = (coef[d] * (dUPSBUFFER[UPSbufferIdx + 1] - dUPSBUFFER[UPSbufferIdx_prv + 1]) >> 10) + dUPSBUFFER[UPSbufferIdx_prv + 1];
		dPARAM[ParamIdx + 2] = (coef[d] * (dUPSBUFFER[UPSbufferIdx + 2] - dUPSBUFFER[UPSbufferIdx_prv + 2]) >> 10) + dUPSBUFFER[UPSbufferIdx_prv + 2];
    }
}

__kernel void kEnhance(__global short* dNORM, __global short* dPARAM, __global uint8* dOUTDATA)
{
    int globalIdx_x = get_global_id(0);
    int globalIdx_y = get_global_id(1);
    int globalIdx_z = get_global_id(2);
    
    int idx = globalIdx_x * IMG_WIDTH * IMG_CHANNEL + globalIdx_y * IMG_CHANNEL + globalIdx_z;

    int qX = dNORM[idx] >> 4;
    int qP = dPARAM[idx];

    int output;
    for (int i = 0; i < 8; ++i) {
        int qX2 = qX << 10;
        int qX3 = qX2 << 10;
        qX3 = qX3 + qP * (qX * qX - qX2);
        qX = (int)(qX3 >> 20);
    }
    output = qX >> 2;
    output = output > 255 ? 255 : output;

    dOUTDATA[idx] = (uint8)output;
}
