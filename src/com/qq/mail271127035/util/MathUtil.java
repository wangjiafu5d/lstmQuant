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

	// 激活函数sigmod y = 1 / (1 + e^-x);
	// 0 < y < 1 ，适用于输入x分布在0附近, 输出大于0 ，如输出概率。
	// y' = y *( 1 - y);
	public static Double sigmoid(final Double x) {
		Double y = 1.0 / (1.0 + Math.pow(Math.E, -x));
		return y;
	}

	public static Matrix sigmoid(final Matrix x) {
		Matrix y = Matrix.Factory.zeros(x.getRowCount(), x.getColumnCount());
		for (int i = 0; i < x.getRowCount(); i++) {
			for (int j = 0; j < x.getColumnCount(); j++) {
				y.setAsDouble(sigmoid(x.getAsDouble(i, j)), i, j);
			}
		}
		return y;
	}

	// 激活函数tanh y = (1 - e^-2x)/(1 + e^-2x);
	// -1 < y < 1 , 适用于输入x分布在0附近, 目标输出对称的
	// y' = 1 - y^2;
	public static Double tanh(final Double x) {
		double temp = Math.pow(Math.E, -2.0*x);
		Double y =( 1.0 - temp) / (1.0 + temp);
		return y;
	}

	public static Matrix tanh(final Matrix x) {
		Matrix result = Matrix.Factory.zeros(x.getRowCount(), x.getColumnCount());
		for (int i = 0; i < x.getRowCount(); i++) {
			for (int j = 0; j < x.getColumnCount(); j++) {
				result.setAsDouble(tanh(x.getAsDouble(i, j)), i, j);
			}
		}
		return result;
	}

	public static Matrix derivativeElu(final Matrix out) {
		Matrix result = Matrix.Factory.zeros(out.getRowCount(), out.getColumnCount());
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
		Matrix result = Matrix.Factory.zeros(grad.getRowCount(), grad.getColumnCount());
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

}
