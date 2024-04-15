#include <stdio.h>
#include <arm_neon.h>

void Gaussian_Elimination(float A[][3], float b[], float x[], int n) {
	for (int k = 0; k < n; ++k) {
		float factor = 1.0f / A[k][k];
		float32x2_t factor_vec = vdup_n_f32(factor);

		for (int j = k + 1; j < n; ++j) {
			float32x2_t Akj_vec = vdup_n_f32(A[k][j]);
			float32x2_t Aki_vec = vld1_dup_f32(&A[k][k]);
			float32x2_t result = vmul_f32(Akj_vec, vrecpe_f32(Aki_vec));
			vst1_f32(&A[k][j], result);
		}

		for (int i = k + 1; i < n; ++i) {
			float32x2_t Aik_vec = vld1_f32(&A[i][k]);

			for (int j = k + 1; j < n; j += 2) {
				float32x2_t Akj_vec = vld1_f32(&A[k][j]);
				float32x2_t Aij_vec = vld1_f32(&A[i][j]);
				float32x2_t result = vmls_f32(Aij_vec, Aik_vec, Akj_vec);
				vst1_f32(&A[i][j], result);
			}

			float32x2_t factor_b = vdup_n_f32(factor * b[k]);
			float32x2_t bi_vec = vld1_f32(&b[i]);
			float32x2_t result = vmls_f32(bi_vec, factor_b, vld1_f32(&b[k]));
			vst1_f32(&b[i], result);
		}
	}

	for (int i = n - 1; i >= 0; --i) {
		float32x2_t sum_vec = vld1_f32(&b[i]);

		for (int j = i + 1; j < n; ++j) {
			float32x2_t Aij_vec = vld1_f32(&A[i][j]);
			float32x2_t xj_vec = vld1_f32(&x[j]);
			sum_vec = vmls_f32(sum_vec, Aij_vec, xj_vec);
		}

		float32x2_t Aii_vec = vld1_f32(&A[i][i]);
		float32x2_t xi_vec = vdup_n_f32(0.0f);
		vst1_f32(&x[i], vdiv_f32(sum_vec, Aii_vec));
	}
}

int main() {
	float A[3][3] = { {2.0, 1.0, 1.0},
					  {4.0, -6.0, 0.0},
					  {-2.0, 7.0, 2.0} };
	float b[3] = { 5.0, -2.0, 9.0 };
	float x[3] = { 0.0 };

	Gaussian_Elimination(A, b, x, 3);

	printf("Solution vector x:\n");
	for (int i = 0; i < 3; ++i) {
		printf("%.2f ", x[i]);
	}
	printf("\n");

	return 0;
}
