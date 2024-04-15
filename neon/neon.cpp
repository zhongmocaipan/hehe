#include <stdio.h>
#include <arm_neon.h>

// Neon向量化的高斯消去法示例

void gaussian_elimination_neon(float* matrix, int rows, int cols) {
	// 对矩阵的每一行执行消元操作
	for (int i = 0; i < rows; ++i) {
		// 将对角线元素归一化
		float scale = matrix[i * cols + i];
		float32x4_t scale_vec = vdupq_n_f32(scale);
		for (int j = 0; j < cols; j += 4) {
			float32x4_t row_vec = vld1q_f32(&matrix[i * cols + j]);
			row_vec = vmulq_f32(row_vec, vrecpeq_f32(scale_vec));
			vst1q_f32(&matrix[i * cols + j], row_vec);
		}

		// 使用消元操作更新矩阵的其他行
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

// 打印矩阵
void print_matrix(float* matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			printf("%f ", matrix[i * cols + j]);
		}
		printf("\n");
	}
}

int main() {
	// 定义测试矩阵
	float matrix[3][3] = { {2.0, 1.0, 1.0},
						  {4.0, -6.0, 0.0},
						  {-2.0, 7.0, 2.0} };

	// 转换为一维数组
	float* matrix_ptr = (float*)matrix;

	// 打印原始矩阵
	printf("Original Matrix:\n");
	print_matrix(matrix_ptr, 3, 3);

	// 对矩阵进行Neon向量化的高斯消去法
	gaussian_elimination_neon(matrix_ptr, 3, 3);

	// 打印消去后的矩阵
	printf("\nMatrix after Gaussian elimination:\n");
	print_matrix(matrix_ptr, 3, 3);

	return 0;
}
