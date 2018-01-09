package com.qq.mail271127035.util;

import org.ujmp.core.Matrix;

public class MathUtil {
	// 激活函数elu y = max(e^x - 1,x);
	// -1 < y < x ，适用于输入x主要分布在大于0位置, 有稀疏激活的作用。
	// y' = max(y - 1, 1);
	public static Double elu(final Double x) {
		Double y = 0.0;
		if (x > 0) {
			y = x;
		} else {
			y = Math.pow(Math.E, x) - 1.0;
		}
		return y;
	}

	// 激活函数MPELU，y = max(α*e^βx - 1)
	public static double mpelu(final double alpha, final double beta, final double x) {
		double y = 0.0;
		if (x > 0) {
			y = x;
		} else {
			y = alpha * (Math.pow(Math.E, beta * x) - 1.0);
		}
		return y;
	}

	public static Matrix mpelu(final double alpha, final double beta, final Matrix m) {
		Double[] doubles = { alpha, beta };
		Matrix[] matrixs = { m };
		return calculateMatrix(matrixs, "mpelu", doubles);
	}

	public static double deriveMpelu(final double alpha, final double beta, double out) {
		double y = 0.0;
		if (out > 0) {
			y = 1;
		} else {
			y = (out + alpha) * beta;
		}
		return y;
	}

	public static Matrix deriveMpelu(final double alpha, final double beta, final Matrix m) {
		Double[] doubles = { alpha, beta };
		Matrix[] matrixs = { m };
		return calculateMatrix(matrixs, "deriveMpelu", doubles);
	}

	public static double deriveAlpha(final double alpha, double out) {
		double y = 0.0;
		if (out > 0) {

		} else {
			y = out / alpha;
		}
		return y;
	}

	public static Matrix deriveAlpha(final double alpha, final double beta, final Matrix m) {
		Double[] doubles = { alpha, beta };
		Matrix[] matrixs = { m };
		
		return calculateMatrix(matrixs, "deriveAlpha", doubles);
	}

	public static double deriveBeta(final double alpha, double in, double out) {
		double y = 0.0;
		if (out > 0) {

		} else {
			y = (out + alpha) * in;
		}
		return y;
	}

	public static Matrix deriveBeta(final double alpha, final Matrix in, final Matrix out) {
		Double[] doubles = { alpha };
		Matrix[] matrixs = { in ,out };
		return calculateMatrix(matrixs, "deriveBeta", doubles);
	}

	// 激活函数sigmod y = 1 / (1 + e^-x);
	// 0 < y < 1 ，适用于输入x分布在0附近, 输出大于0 ，如输出概率。
	// y' = y *( 1 - y);
	public static Double sigmoid(final Double x) {
		Double y = 1.0 / (1.0 + Math.pow(Math.E, -x));
		return y;
	}

	public static Matrix sigmoid(final Matrix x) {
		Double[] doubles = null;
		Matrix[] matrixs = { x };
		return calculateMatrix(matrixs, "sigmoid", doubles);
	}

	// 激活函数tanh y = (1 - e^-2x)/(1 + e^-2x);
	// -1 < y < 1 , 适用于输入x分布在0附近, 目标输出对称的
	// y' = 1 - y^2;
	public static Double tanh(final Double x) {
		double temp = Math.pow(Math.E, -2.0 * x);
		Double y = (1.0 - temp) / (1.0 + temp);
		return y;
	}

	public static Matrix tanh(final Matrix x) {
		Double[] doubles = null;
		Matrix[] matrixs = { x };
		return calculateMatrix(matrixs, "tanh", doubles);
	}

	public static Matrix deriveElu(final Matrix out) {
		Matrix result = MyMatrixUtil.copyZerosMatrix(out);
		for (int n = 0; n < out.getRowCount(); n++) {
			for (int m = 0; m < out.getColumnCount(); m++) {
				if (out.getAsDouble(n, m) > 0) {
					result.setAsDouble(1.0, n, m);
				} else {
					Double d = out.getAsDouble(n, m) + 1.0;
					result.setAsDouble(d, n, m);
				}
			}
		}
		return result;
	}

	public static Matrix gradCheck(final Matrix grad) {
		Matrix result = MyMatrixUtil.copyZerosMatrix(grad);
		for (int i = 0; i < grad.getRowCount(); i++) {
			for (int j = 0; j < grad.getColumnCount(); j++) {
				if (grad.getAsDouble(i, j) > 15) {
					result.setAsDouble(15.0, i, j);
				} else {
					result.setAsDouble(grad.getAsDouble(i, j), i, j);
				}
			}
		}
		return result;
	}

	public static void matrixCheck(final Matrix matrix) {
		for (int i = 0; i < matrix.getRowCount(); i++) {
			for (int j = 0; j < matrix.getColumnCount(); j++) {
				if (matrix.getAsDouble(i, j) > 30) {
					System.out.println("有参数项超过30");
					System.out.println(matrix);
				}
			}
		}
	}

	private static Matrix calculateMatrix(final Matrix[] matrixs, final String deriveMethod, final Double... doubles) {

		Matrix result = MyMatrixUtil.copyZerosMatrix(matrixs[0]);

		for (int i = 0; i < matrixs[0].getRowCount(); i++) {
			for (int j = 0; j < matrixs[0].getColumnCount(); j++) {
				switch (deriveMethod) {
				case "sigmoid":
					result.setAsDouble(sigmoid(matrixs[0].getAsDouble(i, j)), i, j);
					break;

				case "tanh":
					result.setAsDouble(tanh(matrixs[0].getAsDouble(i, j)), i, j);
					break;

				case "mpelu":
					result.setAsDouble(mpelu(doubles[0], doubles[1], matrixs[0].getAsDouble(i, j)), i, j);
					break;

				case "deriveMpelu":
					result.setAsDouble(deriveMpelu(doubles[0], doubles[1], matrixs[0].getAsDouble(i, j)), i, j);
					break;

				case "deriveAlpha":
					result.setAsDouble(deriveAlpha(doubles[0], matrixs[0].getAsDouble(i, j)), i, j);
					break;

				case "deriveBeta":
					result.setAsDouble(
							deriveBeta(doubles[0], matrixs[0].getAsDouble(i, j), matrixs[1].getAsDouble(i, j)), i, j);
					break;
				default :
					System.out.println("未知方法");
					break;
				}
			}
		}
		
		return result;
	}
	public static void main(String[] args) {
		Matrix matrix2 = Matrix.Factory.randn(1,7);
		double alpha = 1;
		double beta = 1;
		Matrix matrix1 = mpelu(alpha, beta, matrix2);
		System.out.println(matrix1);
		System.out.println(sigmoid(matrix1));
		System.out.println(tanh(matrix1));		
		System.out.println(deriveElu(matrix1));
		System.out.println(deriveMpelu(alpha, beta, matrix1));
		System.out.println(deriveAlpha(alpha, beta ,matrix1));
		System.out.println(deriveBeta(alpha, matrix2, matrix1));
	}
}
