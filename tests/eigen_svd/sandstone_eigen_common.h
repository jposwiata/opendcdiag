/**
 * @file
 *
 * @copyright
 * Copyright 2022 Intel Corporation.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SANDSTONE_EIGEN_COMMON_H
#define SANDSTONE_EIGEN_COMMON_H

#include <sandstone.h>

#include <boost/type_traits/is_complex.hpp>
#include <Eigen/Eigenvalues>

#include <sandstone_p.h>

namespace {
template <typename SVD, int Dim> struct EigenSVDTest
{
    using Mat = typename SVD::MatrixType;
    struct eigen_test_data {
        Mat orig_matrix;
        Mat u_matrix;
        Mat v_matrix;

        int written;
        std::string initial_seed;
        std::string run_seed;
    };

    [[gnu::noinline]] static void calculate_once(const Mat &orig_matrix, Mat &u, Mat &v)
    {
        SVD fullSvd(orig_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        u = fullSvd.matrixU();
        v = fullSvd.matrixV();
    }

    template <typename FP> static inline std::enable_if_t<boost::is_complex<FP>::value>
    compare_or_fail(const FP *actual, const FP *expected, const char *name, struct test* test)
    {
        eigen_test_data* d = static_cast<eigen_test_data*>(test->data);
        size_t count = Dim * Dim;
        size_t elemSize = sizeof(FP);

        if (memcmp(actual, expected, count * elemSize) != 0) {
            fprintf(stderr, "Verification %s failed for initial_seed=%s, run_seed=%s\n", name, d->initial_seed.c_str(), d->run_seed.c_str());
            char filename[256];
            sprintf(filename, "%s-%s", test->id, d->initial_seed.c_str());

            // "semaphore" for poor people :P
            if (d->written++ == 0) {
                FILE* stored = fopen(filename, "wb");
                if (stored) {
                    fprintf(stderr, "Store run_seed='%s'. length %d..\n", d->run_seed.c_str(), (int) d->run_seed.length());
                    fputs(d->run_seed.c_str(), stored);
                    fputc(0, stored);

                    size_t to_go = elemSize * count;
                    const uint8_t* data = (const uint8_t*) d->orig_matrix.data();

                    // "golden" values are NOT intended to be stored, these depend on binary of the code!!!!
                    fprintf(stderr, "store %d bytes from %p..\n", (int) to_go, data);
                    while (to_go > 0)
                    {
                        const size_t written = fwrite(data, 1, to_go, stored);
                        if (written <= 0) {
                            break;
                        }
                        to_go -= written;
                        data += written;
                    }

                    fclose(stored);
                } else {
                    fprintf(stderr, "Cannot open file %s to write..\n", filename);
                }
            } else {
                fprintf(stderr, "File '%s' already written (%d has tried)\n", filename, d->written);
            }

            report_fail_msg("initial_seed=%s, run_seed=%s", d->initial_seed.c_str(), d->run_seed.c_str());
        }
    }

    template <typename FP> static inline std::enable_if_t<!boost::is_complex<FP>::value>
    compare_or_fail(const FP *actual, const FP *expected, const char *name, struct test* test)
    {
        memcmp_or_fail(actual, expected, Dim * Dim, name);
    }

    static int init(struct test *test)
    {
        size_t to_go = -1;

        auto d = new eigen_test_data;
        d->initial_seed = random_format_seed();
        d->written = 0;

        char filename[256];
        sprintf(filename, "%s-%s", test->id, d->initial_seed.c_str());

        FILE* restored = fopen(filename, "rb");
        if (restored) {
            fprintf(stderr, "Reading file %s for initial_seed=%s\n", filename, d->initial_seed.c_str());

            // read "run seed"
            char run_seed[128];
            int len = 0;
            do {
                int c = fgetc(restored);
                if (c <= 0) {
                    break;
                }
                run_seed[len++] = c;
                if (len >= sizeof(run_seed)) {
                    len = 0;
                    break;
                }
            } while (1);
            run_seed[len] = '\0';
            fprintf(stderr, "Read run_seed=%s (len %d)\n", run_seed, len);
            if (len != 0) {
                fprintf(stderr, "Setting run seed='%s'\n", run_seed);
                random_global_init(run_seed);
            }
            d->run_seed = run_seed;

            // read input matrix
            d->orig_matrix = Mat::Zero(Dim, Dim);
            to_go = Dim * Dim * sizeof(*d->orig_matrix.data());
            uint8_t* data = (uint8_t*) d->orig_matrix.data();
            fprintf(stderr, "Reading %d bytes to matrix %p\n", (int) to_go, data);
            while (to_go > 0)
            {
                const size_t read = fread(data, 1, to_go, restored);
                if(read <= 0) {
                    break;
                }
                to_go -= read;
                data += read;
            }
            if (fgetc(restored) != EOF) {
                to_go = -2;
            }
            fclose(restored);
        }

        if (to_go == 0) {
            d->written = 666;
        } else {
            random_global_init(d->initial_seed.c_str());
            d->orig_matrix = Mat::Random(Dim, Dim);
            d->run_seed = random_format_seed();
            fprintf(stderr, "%s data file (to go %d), initialize with random data, initial_seed=%s, execution seed=%s\n",
                    to_go == -1 ? "No" : to_go == -2 ? "Too long" : "Incorrect", (int) to_go, d->initial_seed.c_str(), d->run_seed.c_str());
        }

        calculate_once(d->orig_matrix, d->u_matrix, d->v_matrix);
        test->data = d;
        return EXIT_SUCCESS;
    }

    static int cleanup(struct test *test)
    {
        delete static_cast<eigen_test_data *>(test->data);
        return EXIT_SUCCESS;
    }

    static int run(struct test *test, int cpu)
    {
        auto d = static_cast<eigen_test_data *>(test->data);
        do {
            Mat u, v;
            calculate_once(d->orig_matrix, u, v);

            compare_or_fail(u.data(), d->u_matrix.data(), "Matrix U", test);
            compare_or_fail(v.data(), d->v_matrix.data(), "Matrix V", test);
        } while (test_time_condition(test));

        return EXIT_SUCCESS;
    }
};

} // unnamed namespace

#endif // SANDSTONE_EIGEN_COMMON_H
