package com.qq.mail271127035.util;

import org.ujmp.core.Matrix;

public class MathUtil {
	public static Double elu(final Double x) {
		Double y = 0.0;
		if (x > 0) {
			y = x;
		} else {
			y = Math.pow(Math.E, x) - 1.0;
		}
		return y;
	}

	public static Double sigmoid(final Double x) {
		Double y = 1.0 / (1.0 + Math.pow(Math.E, -x));
		return y;
	}

	public static Double tanh(final Double x) {
		Double y = (Math.pow(Math.E, x) - Math.pow(Math.E, -x)) / (Math.pow(Math.E, x) - Math.pow(Math.E, -x));
		return y;
	}

	public static Matrix tanh(final Matrix x) {
		Matrix result = Matrix.Factory.zeros(x.getRowCount(), x.getColumnCount());
		for (int i = 0; i < x.getRowCount(); i++) {
			for (int j = 0; j < x.getColumnCount(); j++) {
				result.setAsDouble(x.getAsDouble(i, j), i, j);
			}
		}
		return result;
	}

	public static Matrix hadamard(final Matrix x1, final Matrix x2) {
		Matrix result = Matrix.Factory.zeros(x1.getRowCount(), x1.getColumnCount());
		if (x1.getRowCount() == x2.getRowCount() && x1.getColumnCount() == x2.getColumnCount()) {
			for (int i = 0; i < x1.getRowCount(); i++) {
				for (int j = 0; j < x1.getColumnCount(); j++) {
					result.setAsDouble(x1.getAsDouble(i, j) * x2.getAsDouble(i, j), i, j);
				}
			}
		} else {
			System.out.println("矩阵x1与x2的行列数不相等，无法hadamard相乘");
			System.out.println("*******x1:" + x1.getRowCount() + "×" + x1.getColumnCount() + "\r\n" + x1);
			System.out.println("*******x2:" + x2.getRowCount() + "×" + x2.getColumnCount() + "\r\n" + x2);
			throw new RuntimeException();
		}
		return result;
	}

	public static Matrix seriesHadamard(final Matrix... matrixs) {
		Matrix result = Matrix.Factory.ones(matrixs[0].getRowCount(), matrixs[0].getColumnCount());
		for (Matrix matrix : matrixs) {
			result = MathUtil.hadamard(result, matrix);
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

	public static Matrix gradTrain(final Matrix w, final Matrix grad, final Double eta) {
		Matrix result = Matrix.Factory.zeros(w.getRowCount(), w.getColumnCount());
		if (w.getRowCount() == grad.getRowCount() && w.getColumnCount() == grad.getColumnCount()) {
			result = w.minus(MathUtil.gradCheck(grad).times(eta));			
		} else {
			System.out.println("矩阵w和矩阵grad的行列数不相等，不满足运算规则");
			System.out.println("*******x1:" + w.getRowCount() + "×" + w.getColumnCount() + "\r\n" + w);
			System.out.println("*******x2:" + grad.getRowCount() + "×" + grad.getColumnCount() + "\r\n" + grad);
			throw new RuntimeException();
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
				if (matrix.getAsDouble(i,j) > 30) {
					System.out.println("有参数项超过30");
					System.out.println(matrix);
				}
			}
		}
	}
	
}
