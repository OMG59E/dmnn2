/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:43:14
 * @FilePath: /dmnn2/src/plugin/common/kernels/reducedMath.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef TRT_REDUCED_MATH_H
#define TRT_REDUCED_MATH_H
/*
 * Dynamically strength-reduced div and mod
 * Ideas taken from Sean Baxter's MGPU library.
 * These classes provide for reduced complexity division and modulus
 * on integers, for the case where the same divisor or modulus will
 * be used repeatedly.
 */
namespace nvinfer1 {
    namespace rt {
        namespace detail {
            void find_divisor(int denom, unsigned int &mul_coeff, unsigned int &shift_coeff);
            __host__ __device__
            __forceinline__ unsigned int umulhi(unsigned int x, unsigned int y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
                return __umulhi(x, y);
#else
                unsigned long long z = (unsigned long long) x * (unsigned long long) y;
                return (unsigned int) (z >> 32);
#endif
            }

/*
 * This is a weird implementation that returns div_up(0,1)=0 but
 * div_up(0,2)=1 (wrong) -- just do not use it with a=0.
 */
            __host__ __device__

            inline int div_up(int a, int b) {
                return (a - 1) / b + 1;
            }

        } // end namespace detail

        class reduced_divisor {
        public:
            reduced_divisor() = default;

            __host__ __forceinline__

            reduced_divisor(int _y)
                    : y(_y) {
                detail::find_divisor(y, mul_coeff, shift_coeff);
            }

            __host__ __device__

            __forceinline__ reduced_divisor(unsigned _mul_coeff, unsigned _shift_coeff, int _y)
                    : mul_coeff(_mul_coeff), shift_coeff(_shift_coeff), y(_y) {
            }

            __host__ __device__

            __forceinline__ int div(int x) const {
                /*
                 * if dividing by 1, then find_divisor wouldn't have worked because
                 * mul_coeff would have had to be 2^32, which can't be represented,
                 * so we have to special case that one.
                 */
                return (y != 1) ? detail::umulhi((unsigned int) x, mul_coeff) >> shift_coeff : x;
            }

            __host__ __device__

            __forceinline__ int mod(int x) const {
                return x - (div(x) * y);
            }

            __host__ __device__

            __forceinline__ void divmod(int x, int &q, int &mod) const {
                q = div(x);
                mod = x - (q * y);
            }

            __host__ __device__

            __forceinline__ int get() const {
                return y;
            }

            inline __host__ void get_mul_shift(unsigned &mul, unsigned &shift) {
                mul = mul_coeff;
                shift = shift_coeff;
            }

        protected:
            unsigned int mul_coeff;
            unsigned int shift_coeff;
            int y;
        };

    } // namespace rt
} // namespace nvinfer1

#endif // TRT_REDUCED_MATH_H
