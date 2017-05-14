#include "calc_cost.h"

texture<float, cudaTextureType2D, cudaReadModeElementType> tex2Dleft;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex2Dright;

int iDivUp(int a, int b)
{
    return (a + b - 1) / b;
}

#define INDEX(dim0, dim1, dim2) (((dim0)*ALIGN_WIDTH + (dim1)) * DEP_CNT + (dim2))

__global__ void
ADCalcCostKernel(
    int measurement_cnt,
    float r11, float r12, float r13,
    float r21, float r22, float r23,
    float r31, float r32, float r33,
    float t1, float t2, float t3,
    float fx, float fy, float cx, float cy,
    unsigned char *img_l, size_t pitch_img_l,
    unsigned char *img_r, size_t pitch_r_warp,
    float *cost)
{
    const int tidx_base = blockDim.x * blockIdx.x;
    const int tidy = blockIdx.y;

    for (int k = 0, tidx = tidx_base; k < DEP_CNT; k++, tidx++)
        if (tidx >= 0 && tidx <= WIDTH - 1 && tidy >= 0 && tidy <= HEIGHT - 1)
        {
            float x = r11 * tidx + r12 * tidy + r13 * 1.0f;
            float y = r21 * tidx + r22 * tidy + r23 * 1.0f;
            float z = r31 * tidx + r32 * tidy + r33 * 1.0f;

            float xu = x - r12; //r11 * tidx + r12 * tidy + r13 * 1.0f;
            float yu = y - r22; //r21 * tidx + r22 * tidy + r23 * 1.0f;
            float zu = z - r32; //r31 * tidx + r32 * tidy + r33 * 1.0f;

            float xd = x + r12; //r11 * tidx + r12 * tidy + r13 * 1.0f;
            float yd = y + r22; //r21 * tidx + r22 * tidy + r23 * 1.0f;
            float zd = z + r32; //r31 * tidx + r32 * tidy + r33 * 1.0f;

            float xl = x - r11; //r11 * tidx + r12 * tidy + r13 * 1.0f;
            float yl = y - r21; //r21 * tidx + r22 * tidy + r23 * 1.0f;
            float zl = z - r31; //r31 * tidx + r32 * tidy + r33 * 1.0f;

            float xr = x + r11; //r11 * tidx + r12 * tidy + r13 * 1.0f;
            float yr = x + r21; //r21 * tidx + r22 * tidy + r23 * 1.0f;
            float zr = x + r31; //r31 * tidx + r32 * tidy + r33 * 1.0f;

            float xul = xu - r11;
            float yul = yu - r21;
            float zul = zu - r31;

            float xdr = xd + r11;
            float ydr = yd + r21;
            float zdr = zd + r31;

            float xld = xl + r12;
            float yld = yl + r22;
            float zld = zl + r32;

            float xru = xr - r12;
            float yru = xr - r22;
            float zru = xr - r32;

            //for (int i = 0; i < DEP_CNT; i++)
            int i = threadIdx.x;
            {
                float *cost_ptr = cost + INDEX(tidy, tidx, i);
                if (measurement_cnt == 1 && (tidx == 0 || tidx == WIDTH - 1 || tidy == 0 || tidy == HEIGHT - 1))
                {
                    *cost_ptr = -1.0f;
                    continue;
                }

                float last_cost = *cost_ptr;
                if (measurement_cnt != 1 && last_cost < 0)
                {
                    continue;
                }

                float tmp = 0.0f;
                float idep = i * DEP_SAMPLE;

                {
                    float w = z + t3 * idep;
                    float u = (x + t1 * idep) / w;
                    float v = (y + t2 * idep) / w;

                    if (w < 0 || u < 0 || u > WIDTH - 1 || v < 0 || v > HEIGHT - 1)
                    {
                        *cost_ptr = -1.0f;
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx+0.5f, tidy+0.5f) - tex2D(tex2Dright, u+0.5f, v+0.5f));
                }

                {
                    float wu = zu + t3 * idep;
                    float uu = (xu + t1 * idep) / wu;
                    float vu = (yu + t2 * idep) / wu;

                    if (wu < 0 || uu < 0 || uu > WIDTH - 1 || vu < 0 || vu > HEIGHT - 1)
                    {
                        *cost_ptr = -1.0f;
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx+0.5f, tidy + 1+0.5f) - tex2D(tex2Dright, uu+0.5f, vu+0.5f));
                }

                {
                    float wd = zd + t3 * idep;
                    float ud = (xd + t1 * idep) / wd;
                    float vd = (yd + t2 * idep) / wd;

                    if (wd < 0 || ud < 0 || ud > WIDTH - 1 || vd < 0 || vd > HEIGHT - 1)
                    {
                        *cost_ptr = -1.0f;
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx+0.5f, tidy - 1+0.5f) - tex2D(tex2Dright, ud+0.5f, vd+0.5f));
                }

                {
                    float wl = zl + t3 * idep;
                    float ul = (xl + t1 * idep) / wl;
                    float vl = (yl + t2 * idep) / wl;

                    if (wl < 0 || ul < 0 || ul > WIDTH - 1 || vl < 0 || vl > HEIGHT - 1)
                    {
                        *cost_ptr = -1.0f;
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx - 1+0.5f, tidy+0.5f) - tex2D(tex2Dright, ul+0.5f, vl+0.5f));
                }

                {
                    float wr = zr + t3 * idep;
                    float ur = (xr + t1 * idep) / wr;
                    float vr = (yr + t2 * idep) / wr;

                    if (wr < 0 || ur < 0 || ur > WIDTH - 1 || vr < 0 || vr > HEIGHT - 1)
                    {
                        *cost_ptr = -1.0f;
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx + 1+0.5f, tidy+0.5f) - tex2D(tex2Dright, ur+0.5f, vr+0.5f));
                }

                {

                    float wul = zul + t3 * idep;
                    float uul = (xul + t1 * idep) / wul;
                    float vul = (yul + t2 * idep) / wul;

                    if (wul < 0 || uul < 0 || uul > WIDTH - 1 || vul < 0 || vul > HEIGHT - 1)
                    {
                        *cost_ptr = -1.0f;
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx - 1+0.5f, tidy - 1+0.5f) - tex2D(tex2Dright, uul+0.5f, vul+0.5f));
                }

                {
                    float wdr = zdr + t3 * idep;
                    float udr = (xdr + t1 * idep) / wdr;
                    float vdr = (ydr + t2 * idep) / wdr;

                    if (wdr < 0 || udr < 0 || udr > WIDTH - 1 || vdr < 0 || vdr > HEIGHT - 1)
                    {
                        *cost_ptr = -1.0f;
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx + 1+0.5f, tidy + 1+0.5f) - tex2D(tex2Dright, udr+0.5f, vdr+0.5f));
                }

                {
                    float wld = zld + t3 * idep;
                    float uld = (xld + t1 * idep) / wld;
                    float vld = (yld + t2 * idep) / wld;

                    if (wld < 0 || uld < 0 || uld > WIDTH - 1 || vld < 0 || vld > HEIGHT - 1)
                    {
                        *cost_ptr = -1.0f;
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx - 1+0.5f, tidy + 1+0.5f) - tex2D(tex2Dright, uld+0.5f, vld+0.5f));
                }

                {
                    float wru = zru + t3 * idep;
                    float uru = (xru + t1 * idep) / wru;
                    float vru = (yru + t2 * idep) / wru;

                    if (wru < 0 || uru < 0 || uru > WIDTH - 1 || vru < 0 || vru > HEIGHT - 1)
                    {
                        *cost_ptr = -1.0f;
                        continue;
                    }

                    tmp += fabs(tex2D(tex2Dleft, tidx + 1+0.5f, tidy - 1+0.5f) - tex2D(tex2Dright, uru+0.5f, vru+0.5f));
                }

                if (measurement_cnt == 1)
                    *cost_ptr = tmp / 9.0f;
                else
                    *cost_ptr = (last_cost * (measurement_cnt - 1) + tmp / 9.0f) / measurement_cnt;
            }
        }
}

__global__ void
filterCostKernel(float *cost,
                 unsigned char *dep, size_t pitch_dep,
                 float var_scale, int var_width)
{
    const int tidx = blockIdx.x; // + threadIdx.x;
    const int tidy = blockIdx.y; // + threadIdx.y;
    const int d = threadIdx.x;

    if (tidx >= 0 && tidx < WIDTH && tidx >= 0 && tidy < HEIGHT)
    {
        float *p_dep = (float *)(dep + tidy * pitch_dep) + tidx;

        __shared__ float c[DEP_CNT], c_min[DEP_CNT];
        __shared__ int c_idx[DEP_CNT];

        c[d] = c_min[d] = cost[INDEX(tidy, tidx, d)];
        c_idx[d] = d;
        __syncthreads();
        for (int i = 64; i > 0; i /= 2)
        {
            if (d < i && d + i < DEP_CNT && c_min[d + i] < c_min[d])
            {
                c_min[d] = c_min[d + i];
                c_idx[d] = c_idx[d + i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            float min_cost = c_min[0];
            int min_idx = c_idx[0];

            if (min_cost == 0 || min_idx == 0 || min_idx == DEP_CNT - 1 || c[min_idx - 1] + c[min_idx + 1] < 2 * min_cost * var_scale)
                *p_dep = 1000.0f;
            else
            {
                //printf("%f %f %f\n", c[min_idx - 1], c[min_idx], c[min_idx + 1]);
                float cost_pre = c[min_idx - 1];
                float cost_post = c[min_idx + 1];
                float a = cost_pre - 2.0f * min_cost + cost_post;
                float b = -cost_pre + cost_post;
                float subpixel_idx = min_idx - b / (2.0f * a);
                *p_dep = 1.0f / (subpixel_idx * DEP_SAMPLE);
            }
        }
    }
}

void ad_calc_cost(
    int measurement_cnt,
    float r11, float r12, float r13,
    float r21, float r22, float r23,
    float r31, float r32, float r33,
    float t1, float t2, float t3,
    float fx, float fy, float cx, float cy,
    unsigned char *img_l, size_t pitch_img_l,
    unsigned char *img_r, size_t pitch_img_r,
    unsigned char *cost)
{
    checkCudaErrors(cudaUnbindTexture(tex2Dleft));
    checkCudaErrors(cudaUnbindTexture(tex2Dright));

    dim3 numThreads = dim3(DEP_CNT, 1, 1);
    dim3 numBlocks = dim3(iDivUp(WIDTH, DEP_CNT), HEIGHT);

    cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<float>();
    tex2Dleft.addressMode[0] = cudaAddressModeBorder;
    tex2Dleft.addressMode[1] = cudaAddressModeBorder;
    tex2Dleft.filterMode = cudaFilterModeLinear;
    tex2Dleft.normalized = false;
    tex2Dright.addressMode[0] = cudaAddressModeBorder;
    tex2Dright.addressMode[1] = cudaAddressModeBorder;
    tex2Dright.filterMode = cudaFilterModeLinear;
    tex2Dright.normalized = false;

    size_t offset = 0;
    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dleft, reinterpret_cast<float *>(img_l), ca_desc0, WIDTH, HEIGHT, ALIGN_WIDTH * sizeof(float)));
    assert(offset == 0);

    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dright, reinterpret_cast<float *>(img_r), ca_desc1, WIDTH, HEIGHT, ALIGN_WIDTH * sizeof(float)));
    assert(offset == 0);

    ADCalcCostKernel << <numBlocks, numThreads>>> (measurement_cnt,
                                                   r11, r12, r13,
                                                   r21, r22, r23,
                                                   r31, r32, r33,
                                                   t1, t2, t3,
                                                   fx, fy, cx, cy,
                                                   img_l, pitch_img_l,
                                                   img_r, pitch_img_r,
                                                   reinterpret_cast<float *>(cost));
    cudaDeviceSynchronize();
}

//void census_calc_cost(int k,
//        unsigned char *img_l, size_t pitch_img_l,
//        unsigned char *img_warp, size_t pitch_img_warp,
//        unsigned char *cost)
//{
//    dim3 numThreads = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
//    dim3 numBlocks = dim3(iDivUp(WIDTH, numThreads.x), iDivUp(HEIGHT, numThreads.y));
//
//    CensusCalcCostKernel << <numBlocks, numThreads>>> (k,
//            img_l, pitch_img_l,
//            img_warp, pitch_img_warp,
//            reinterpret_cast<float *>(cost));
//    cudaDeviceSynchronize();
//}

void filter_cost(
    unsigned char *cost,
    unsigned char *dep, size_t pitch_dep)
{
    dim3 numThreads = dim3(DEP_CNT, 1, 1);
    dim3 numBlocks = dim3(WIDTH, HEIGHT, 1);

    filterCostKernel << <numBlocks, numThreads>>> (reinterpret_cast<float *>(cost),
                                                   dep, pitch_dep,
                                                   var_scale, var_width);
    cudaDeviceSynchronize();
}

template <int idx, int start, int dx, int dy, int n>
__global__ void sgm2(
    unsigned char *x0, size_t pitch_x0,
    unsigned char *x1, size_t pitch_x1,
    float *input,
    float *output,
    float pi1, float pi2, float tau_so, float sgm_q1, float sgm_q2)
{
    int xy[2] = {blockIdx.x, blockIdx.x};
    xy[idx] = start;
    int x = xy[0], y = xy[1];
    int d = threadIdx.x;

    __shared__ float output_s[400], output_min[400];
    __shared__ float input_s[400], input_min[400];

    input_s[d] = input_min[d] = input[INDEX(y, x, d)];
    __syncthreads();
    for (int i = 32; i > 0; i /= 2)
    {
        if (d < i && d + i < DEP_CNT && input_min[d + i] < input_min[d])
        {
            input_min[d] = input_min[d + i];
        }
        __syncthreads();
    }
    if (input_min[0] < 0.0f)
    {
        input_s[d] = 0.0f;
        output[INDEX(y, x, d)] = input_s[d];
        output_s[d] = output_min[d] = input_s[d];
    }
    else
    {
        output[INDEX(y, x, d)] += input_s[d];
        output_s[d] = output_min[d] = input_s[d];
    }
    xy[0] += dx;
    xy[1] += dy;

    for (int k = 1; k < n; k++, xy[0] += dx, xy[1] += dy)
    {
        x = xy[0];
        y = xy[1];

        input_s[d] = input_min[d] = input[INDEX(y, x, d)];
        __syncthreads();
        for (int i = 64; i > 0; i /= 2)
        {
            if (d < i && d + i < DEP_CNT && output_min[d + i] < output_min[d])
            {
                output_min[d] = output_min[d + i];
            }
            if (d < i && d + i < DEP_CNT && input_min[d + i] < input_min[d])
            {
                input_min[d] = input_min[d + i];
            }
            __syncthreads();
        }
        if (input_min[0] < 0.0f)
        {
            input_s[d] = 0.0f;
            __syncthreads();
        }

        //float *i_cur = (float *)(x0 + y * pitch_x0) + x;
        //float *i_pre = (float *)(x0 + (y - dy) * pitch_x0) + (x - dx);
        //float D1 = fabs(*i_cur - *i_pre);
        //float D2 = fabs(*i_cur - *i_pre);
        float D1 = fabs(tex2D(tex2Dleft, x+0.5f, y+0.5f) - tex2D(tex2Dleft, x - dx+0.5f, y - dy+0.5f));
        //float D1 = fabs(tex2D( - *i_pre);
        //float D2 = D1;
        // int xx = x + d * direction;
        // if (xx < 0 || xx >= size2 || xx - dx < 0 || xx - dx >= size2)
        // {
        //     D2 = 10;
        // }
        // else
        // {
        //     D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * size2 - dx);
        // }
        float P1 = pi1, P2 = pi2;
        if (D1 < tau_so)
        {
            P1 /= sgm_q1;
            P2 /= sgm_q2;
        }
        //if (D1 < tau_so && D2 < tau_so)
        //{
        //    P1 = pi1;
        //    P2 = pi2;
        //}
        //else if (D1 > tau_so && D2 > tau_so)
        //{
        //    P1 = pi1 / (sgm_q1 * sgm_q1);
        //    P2 = pi2 / (sgm_q2 * sgm_q2);
        //}
        //else
        //{
        //    P1 = pi1 / sgm_q1;
        //    P2 = pi2 / sgm_q1;
        //}

        float cost = min(output_s[d], output_min[0] + P2);
        if (d - 1 >= 0)
        {
            cost = min(cost, output_s[d - 1] + P1);
        }
        if (d + 1 < DEP_CNT)
        {
            cost = min(cost, output_s[d + 1] + P1);
        }

        float val = input_s[d] + cost - output_min[0];
        if (input_min[0] < 0.0f)
        {
            output[INDEX(y, x, d)] = 0.0;
        }
        else
        {
            output[INDEX(y, x, d)] += val;
        }

        __syncthreads();
        output_min[d] = output_s[d] = val;
    }
}

void sgm2(
    unsigned char *x0, size_t pitch_x0,
    unsigned char *x1, size_t pitch_x1,
    unsigned char *input,
    unsigned char *output)
{
    sgm2<0, 0, 1, 0, WIDTH> << <HEIGHT, DEP_CNT>>> (x0, pitch_x0,
                                                    x1, pitch_x1,
                                                    reinterpret_cast<float *>(input),
                                                    reinterpret_cast<float *>(output),
                                                    pi1, pi2, tau_so, sgm_q1, sgm_q2);

    sgm2<0, WIDTH - 1, -1, 0, WIDTH> << <HEIGHT, DEP_CNT>>> (x0, pitch_x0,
                                                             x1, pitch_x1,
                                                             reinterpret_cast<float *>(input),
                                                             reinterpret_cast<float *>(output),
                                                             pi1, pi2, tau_so, sgm_q1, sgm_q2);

    sgm2<1, 0, 0, 1, HEIGHT> << <WIDTH, DEP_CNT>>> (x0, pitch_x0,
                                                    x1, pitch_x1,
                                                    reinterpret_cast<float *>(input),
                                                    reinterpret_cast<float *>(output),
                                                    pi1, pi2, tau_so, sgm_q1, sgm_q2);

    sgm2<1, HEIGHT - 1, 0, -1, HEIGHT> << <WIDTH, DEP_CNT>>> (x0, pitch_x0,
                                                              x1, pitch_x1,
                                                              reinterpret_cast<float *>(input),
                                                              reinterpret_cast<float *>(output),
                                                              pi1, pi2, tau_so, sgm_q1, sgm_q2);
    cudaDeviceSynchronize();
}
