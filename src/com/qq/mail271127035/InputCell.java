package com.qq.mail271127035;

import org.ujmp.core.Matrix;

import com.qq.mail271127035.util.MathUtil;

/**
 * 
 * 输入向量，经y=δ(f(Wx+b)得到输出
 * 
 * @author chuan
 * @Date 2017-10-29
 * @version 1.0
 */
public class InputCell {

	Matrix w;
	Matrix b;
	Matrix out;
//	private static InputCell inputCell = new InputCell();

	/**
	 * 输出y=δ(f(Wx+b)，激活函数δ为elu，y = e^x - 1(x<=0),y = x(x>0);
	 * 
	 * @param x
	 *            Matrix类型输入向量x
	 * @return 返回经过神经元处理后得到的Matrix类型矩阵,并且取转置
	 */
	public Matrix out(final Matrix x) {
		// 矩阵转换
		Matrix fx = Matrix.Factory.ones(w.getRowCount(), x.getColumnCount());
		out = Matrix.Factory.copyFromMatrix(fx);
		if (w.getColumnCount() == x.getRowCount() && fx.getRowCount() == b.getRowCount()
				&& fx.getColumnCount() == b.getColumnCount()) {
			fx = w.mtimes(x).plus(b);
		} else {
			System.out.println("w向量,x向量,b向量行列数不匹配，无法计算");
		}		
		// 激活函数eLU y = e^x - 1(x<=0),y = x(x>0);
		for (long i = 0; i < fx.getRowCount(); i++) {
			for (long j = 0; j < fx.getColumnCount(); j++) {
				Double scalar = fx.getAsDouble(i, j);
				Double y = MathUtil.elu(scalar);
				out.setAsDouble(y, i, j);
			}

		}

		return out;
	}

	/**
	 * 设置神经元系数矩阵，并得到构造好的神经元
	 * 
	 * @param w
	 *            矩阵乘法系数矩阵
	 * @param b
	 *            矩阵加法系数矩阵
	 * @return 设置好系数矩阵的神经元inputcell
	 */
	public static InputCell build(final Matrix w, final Matrix b) {
		InputCell inputCell = new InputCell();
		inputCell.setW(w);
		inputCell.setB(b);
		return inputCell;
	}

	private InputCell() {
	}

	public Matrix getW() {
		return w;
	}

	public void setW(Matrix w) {
		this.w = w;
	}

	public Matrix getB() {
		return b;
	}

	public void setB(Matrix b) {
		this.b = b;
	}

	public Matrix getOut() {
		return out;
	}

	public void setOut(Matrix out) {
		this.out = out;
	}
}
