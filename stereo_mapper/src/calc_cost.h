#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <cuda_runtime.h>

#include "parameters.h"

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), "error", func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void ad_calc_cost(
    int measurement_cnt,
    float r11, float r12, float r13,
    float r21, float r22, float r23,
    float r31, float r32, float r33,
    float t1, float t2, float t3,
    float fx, float fy, float cx, float cy,
    unsigned char *img_l, size_t pitch_img_l,
    unsigned char *img_r, size_t pitch_img_r,
    unsigned char *cost);

void object_mask(unsigned char *mask, size_t pitch_mask,
                 unsigned char *cnt, size_t pitch_cnt,
                 unsigned char *cost);
//void census_calc_cost(int k,
//                      unsigned char *img_l, size_t pitch_img_l,
//                      unsigned char *img_warp, size_t pitch_img_warp,
//                      unsigned char *cost);

void filter_cost(unsigned char *cost,
                 unsigned char *dep, size_t pitch_dep);

void sgm2(unsigned char *x0, size_t pitch_x0,
          unsigned char *x1, size_t pitch_x1,
          unsigned char *input,
          unsigned char *output);
