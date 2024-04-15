//#include <stdio.h>
//#include <stdlib.h>
//#include <arm_neon.h>
//void gaussian_elimination_simd(float* matrix, int rows, int cols) {
//	for (int i = 0; i < rows; ++i) {
//		float scale = matrix[i * cols + i];
//		float32x4_t scale_vec = vdupq_n_f32(scale);
//		for (int j = 0; j < cols; j += 4) {
//			float32x4_t row_vec = vld1q_f32(&matrix[i * cols + j]);
//			row_vec = vmulq_f32(row_vec, vrecpeq_f32(scale_vec));
//			vst1q_f32(&matrix[i * cols + j], row_vec);
//		}
//		for (int k = i + 1; k < rows; ++k) {
//			float multiplier = matrix[k * cols + i];
//			float32x4_t multiplier_vec = vdupq_n_f32(multiplier);
//			for (int j = 0; j < cols; j += 4) {
//				float32x4_t row_i_vec = vld1q_f32(&matrix[i * cols + j]);
//				float32x4_t row_k_vec = vld1q_f32(&matrix[k * cols + j]);
//				row_k_vec = vmlaq_f32(row_k_vec, row_i_vec, multiplier_vec);
//				vst1q_f32(&matrix[k * cols + j], row_k_vec);
//			}
//		}
//	}
//}
//void print_matrix(float* matrix, int rows, int cols) {
//	for (int i = 0; i < rows; ++i) {
//		for (int j = 0; j < cols; ++j) {
//			printf("%f ", matrix[i * cols + j]);
//		}
//		printf("\n");
//	}
//}
//
//int main() {
//	float matrix[3][3] = { {2.0, 1.0, 1.0},
//						  {4.0, -6.0, 0.0},
//						  {-2.0, 7.0, 2.0} };
//	float* matrix_ptr = (float*)matrix;
//	printf("Original Matrix:\n");
//	print_matrix(matrix_ptr, 3, 3);
//	gaussian_elimination_simd(matrix_ptr, 3, 3);
//	printf("\nMatrix after Gaussian elimination:\n");
//	print_matrix(matrix_ptr, 3, 3);
//
//	return 0;
//}
#include <stdio.h>

void gaussian_elimination(float* matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		float scale = matrix[i * cols + i];
		for (int j = i + 1; j < cols; ++j) {
			matrix[i * cols + j] /= scale;
		}
		matrix[i * cols + i] = 1.0;
		for (int k = i + 1; k < rows; ++k) {
			float multiplier = matrix[k * cols + i];
			for (int j = i; j < cols; ++j) {
				matrix[k * cols + j] -= multiplier * matrix[i * cols + j];
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

int main() {
	float matrix[3][3] = { {2.0, 1.0, 1.0},
						  {4.0, -6.0, 0.0},
						  {-2.0, 7.0, 2.0} };
	float* matrix_ptr = (float*)matrix;
	printf("Original Matrix:\n");
	print_matrix(matrix_ptr, 3, 3);
	gaussian_elimination(matrix_ptr, 3, 3);
	printf("\nMatrix after Gaussian elimination:\n");
	print_matrix(matrix_ptr, 3, 3);

	return 0;
}
