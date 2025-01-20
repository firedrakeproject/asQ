#include <complex.h>
#include <math.h>
#include <petsc.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

static void form1_cell_integral(double *__restrict__ A, double const *__restrict__ coords, double const *__restrict__ w_0, double const *__restrict__ w_1, double const *__restrict__ c_0, double const *__restrict__ c_1, double const *__restrict__ c_2, double const *__restrict__ c_3, double const *__restrict__ c_4, double const *__restrict__ c_5);
static void form1_cell_integral(double *__restrict__ A, double const *__restrict__ coords, double const *__restrict__ w_0, double const *__restrict__ w_1, double const *__restrict__ c_0, double const *__restrict__ c_1, double const *__restrict__ c_2, double const *__restrict__ c_3, double const *__restrict__ c_4, double const *__restrict__ c_5)
{
  double t0;
  double t1;
  double t10[6] = { 0.0, 0.0, 0.0, 4.8989794855663575, 0.0, 0.0 };
  double t11[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, -4.8989794855663575 };
  double t12[6] = { 1.4142135623730951, 2.4494897427831788, -1.4142135623730951, -2.4494897427831788, 1.4142135623730951, -2.4494897427831788 };
  double t13;
  double t14;
  double t15;
  double t16;
  double t17;
  double t18;
  double t19;
  double t2;
  double t20;
  double t21;
  double t22;
  double t23;
  double t24;
  double t25;
  double t26;
  double t27;
  double t28;
  double t29;
  double t3;
  double t30;
  double t31;
  double t32;
  double t33[1] = { 1.0 };
  double t4;
  double t5;
  double t6;
  double t7;
  double t8[15] = { 0.026590416648380227, 0.026590416648380227, 0.026590416648380227, 0.020459085197028423, 0.020459085197028423, 0.020459085197028423, 0.027877270270345533, 0.027877270270345533, 0.027877270270345533, 0.027877270270345533, 0.027877270270345533, 0.027877270270345533, 0.06386262428056678, 0.06386262428056678, 0.06386262428056678 };
  double t9[6] = { 1.4142135623730954, -2.4494897427831788, -1.4142135623730954, 2.4494897427831788, 1.4142135623730951, 2.4494897427831788 };

  t21 = (double) (0.0);
  t0 = -1.0 * coords[0];
  t1 = t0 + coords[2];
  t2 = -1.0 * coords[1];
  t3 = t2 + coords[5];
  t4 = t0 + coords[4];
  t5 = t2 + coords[3];
  t6 = t1 * t3 + -1.0 * t4 * t5;
  t7 = fabs(t6);
  t20 = (double) (0.0);
  t19 = (double) (0.0);
  t18 = (double) (0.0);
  t17 = (double) (0.0);
  t16 = (double) (0.0);
  t15 = (double) (0.0);
  t14 = (double) (0.0);
  t13 = (double) (0.0);
  for (int32_t i = 0; i <= 5; ++i)
  {
    t13 = t13 + t12[i] * w_0[1 + 2 * i];
    t14 = t14 + t11[i] * w_0[1 + 2 * i];
    t15 = t15 + t10[i] * w_0[1 + 2 * i];
    t16 = t16 + t9[i] * w_0[1 + 2 * i];
    t17 = t17 + t12[i] * w_0[2 * i];
    t18 = t18 + t11[i] * w_0[2 * i];
    t19 = t19 + t10[i] * w_0[2 * i];
    t20 = t20 + t9[i] * w_0[2 * i];
  }
  for (int32_t ip = 0; ip <= 14; ++ip)
    t21 = t21 + t8[ip] * t7;
  t22 = 1.0 / t6;
  t23 = t3 * t22;
  t24 = -1.0 * t5 * t22;
  t25 = t1 * t22;
  t26 = t4 * t22;
  t27 = -1.0 * t4 * t22;
  t28 = t5 * t22;
  t29 = -1.0 * ((t20 * t23 + t19 * t24) * t25 + (t18 * t23 + t17 * t24) * t26 + (t20 * t27 + t19 * t25) * t28 + (t18 * t27 + t17 * t25) * t23);
  t30 = -1.0 * ((t16 * t23 + t15 * t24) * t25 + (t14 * t23 + t13 * t24) * t26 + (t16 * t27 + t15 * t25) * t28 + (t14 * t27 + t13 * t25) * t23);
  t31 = (w_1[0] * c_3[0] + t29 * c_5[0] + t30 * c_4[0] + w_1[1] * c_2[0]) * t21;
  t32 = (w_1[1] * -1.0 * c_3[0] + t30 * -1.0 * c_5[0] + t29 * c_4[0] + w_1[0] * c_2[0]) * t21;
  {
    int32_t const j0 = 0;

    A[0] = A[0] + t33[0] * t32;
    A[1] = A[1] + t33[0] * t31;
  }

}

void wrap_form1_cell_integral(int32_t const start, int32_t const end, double *__restrict__ dat0, double const *__restrict__ dat1, double const *__restrict__ dat2, double const *__restrict__ dat3, double const *__restrict__ glob0, double const *__restrict__ glob1, double const *__restrict__ glob2, double const *__restrict__ glob3, double const *__restrict__ glob4, double const *__restrict__ glob5, int32_t const *__restrict__ map0, int32_t const *__restrict__ map1, int32_t const *__restrict__ map2)
{
  double t0[2];
  double t1[3 * 2];
  double t2[6 * 2];
  double t3[2];

  for (int32_t n = start; n <= -1 + end; ++n)
  {
    {
      int32_t const i21 = 0;

      {
        int32_t const i22 = 0;

        for (int32_t i23 = 0; i23 <= 1; ++i23)
          t0[i23] = (double) (0.0);
      }
    }
    {
      int32_t const i24 = 0;

      for (int32_t i25 = 0; i25 <= 2; ++i25)
        for (int32_t i26 = 0; i26 <= 1; ++i26)
          t1[2 * i25 + i26] = dat1[2 * map1[3 * n + i25] + i26];
    }
    {
      int32_t const i27 = 0;

      for (int32_t i28 = 0; i28 <= 5; ++i28)
        for (int32_t i29 = 0; i29 <= 1; ++i29)
          t2[2 * i28 + i29] = dat2[2 * map2[6 * n + i28] + i29];
    }
    {
      int32_t const i30 = 0;

      {
        int32_t const i31 = 0;

        for (int32_t i32 = 0; i32 <= 1; ++i32)
          t3[i32] = dat3[2 * map0[n] + i32];
      }
    }
    form1_cell_integral(&(t0[0]), &(t1[0]), &(t2[0]), &(t3[0]), &(glob0[0]), &(glob1[0]), &(glob2[0]), &(glob3[0]), &(glob4[0]), &(glob5[0]));
    {
      int32_t const i18 = 0;

      for (int32_t i19 = 0; i19 <= 1; ++i19)
      {
        int32_t const i20 = 0;

        dat0[2 * map0[n] + i19] = dat0[2 * map0[n] + i19] + t0[i19];
      }
    }
  }
}