#include <stdio.h>
#include <arm_neon.h>
void gaussian_elimination_neon(float* matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		float scale = matrix[i * cols + i];
		float32x4_t scale_vec = vdupq_n_f32(scale);
		for (int j = 0; j < cols; j += 4) {
			float32x4_t row_vec = vld1q_f32(&matrix[i * cols + j]);
			row_vec = vmulq_f32(row_vec, vrecpeq_f32(scale_vec));
			vst1q_f32(&matrix[i * cols + j], row_vec);
		}
		for (int k = i + 1; k < rows; ++k) {
			float multiplier = matrix[k * cols + i];
			float32x4_t multiplier_vec = vdupq_n_f32(multiplier);
			for (int j = 0; j < cols; j += 4) {
				float32x4_t row_i_vec = vld1q_f32(&matrix[i * cols + j]);
				float32x4_t row_k_vec = vld1q_f32(&matrix[k * cols + j]);
				row_k_vec = vmlaq_f32(row_k_vec, row_i_vec, multiplier_vec);
				vst1q_f32(&matrix[k * cols + j], row_k_vec);
			}
		}
	}
}

void print_matrix(float* matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			printf("%f ", matrix[i * cols + j]);
		}
		printf("\n");
	}
}
