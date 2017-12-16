package com.qq.mail271127035;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import com.qq.mail271127035.util.MathUtil;

public class LstmCell {
	List<Matrix> wList = new ArrayList<Matrix>();
	List<Matrix> bList = new ArrayList<Matrix>();
	List<Matrix> lstmCell_result = new ArrayList<Matrix>();
//	private static LstmCell lstmCell;

	public static LstmCell build(final List<Matrix> wList, final List<Matrix> bList) {
		LstmCell lstmCell = new LstmCell();
		lstmCell.setwList(wList);
		lstmCell.setbList(bList);
		return lstmCell;
	}

	public List<Matrix> lstmCell_Out(final Matrix x1, final Matrix h_in, final Matrix ct_in) {
		lstmCell_result.clear();
		Matrix ft = calculate_Ft(wList.get(0), h_in, x1, bList.get(0));
		lstmCell_result.add(ft);
		Matrix it = calculate_It(wList.get(1), h_in, x1, bList.get(1));
		lstmCell_result.add(it);
		Matrix ct_cell = calculate_Ct_Cell(wList.get(2), h_in, x1, bList.get(2));
		lstmCell_result.add(ct_cell);
		Matrix ct_out = calculate_Ct_Out(ft, ct_in, it, ct_cell);
		lstmCell_result.add(ct_out);
		Matrix ot = calculate_Ot(wList.get(3), h_in, x1, bList.get(3));
		lstmCell_result.add(ot);
		Matrix ht = calculate_Ht(ot, ct_out);
		lstmCell_result.add(ht);
		List<Matrix> lstmCell_Out = new ArrayList<Matrix>();
		lstmCell_Out.add(ct_out);
		lstmCell_Out.add(ht);
		return lstmCell_Out;
	}

	public Matrix calculate_Ft(final Matrix w_f, final Matrix h_in, final Matrix x, final Matrix b_f) {
		Matrix result = calculateVector(w_f, h_in, x, b_f);		
		// 将矩阵运算结果中每个Double元素都通过sigmoid转化为0~1的值
		result = MathUtil.sigmoid(result);
		return result;
	}

	public Matrix calculate_It(final Matrix w_i, final Matrix h_in, final Matrix x1, final Matrix b_i) {
		return calculate_Ft(w_i, h_in, x1, b_i);
	}

	public Matrix calculate_Ct_Cell(final Matrix w_c, final Matrix h_in, final Matrix x1, final Matrix b_c) {
		Matrix d = calculateVector(w_c, h_in, x1, b_c);		
		// 将矩阵运算结果中每个Double元素都通过tanh转化为-1~1的值		
		return MathUtil.tanh(d);
	}

	public Matrix calculate_Ct_Out(final Matrix ft, final Matrix ct_in, final Matrix it, final Matrix ct_cell) {
		// ft与ct_in的Hadamard乘积，it与ct_cell的Hadamard乘积，其元素定义为两个矩阵对应元素的乘积的m×n矩阵
		Matrix result = Matrix.Factory.zeros(ft.getRowCount(), ft.getColumnCount());
		result = ft.times(ct_in).plus(it.times(ct_cell));
		return result;
	}

	public Matrix calculate_Ot(final Matrix w_o, final Matrix h_in, final Matrix x1, final Matrix b_o) {
		return calculate_Ft(w_o, h_in, x1, b_o);
	}

	public Matrix calculate_Ht(final Matrix ot, final Matrix ct_out) {
		// ot与ct_out的Hadamard乘积，其元素定义为两个矩阵对应元素的乘积的m×n矩阵		
		return ot.times(MathUtil.tanh(ct_out));
	}

	/**
	 * 
	 * @param w
	 *            系数矩阵w,1×2
	 * @param h_in
	 *            上一个lstm单元的输出矩阵h_in,1×6
	 * @param x
	 *            本个单元的x输入,1×6
	 * @param b
	 *            系数矩阵b,1×2
	 * @return Matrix类型的矩阵运算结果，y = w[h_in,x1] + b
	 */
	public Matrix calculateVector(final Matrix w, final Matrix h_in, final Matrix x, final Matrix b) {
		Matrix xAppendHin = Matrix.Factory.zeros(2*x.getRowCount(), x.getColumnCount());
		Matrix y = Matrix.Factory.zeros(w.getRowCount(), x.getColumnCount());
		if (w.getColumnCount() == xAppendHin.getRowCount()) {
			// 将矩阵h_in,x1合并成矩阵h_in_And_x1再与矩阵w相乘
			xAppendHin = x.appendVertically(Ret.LINK, h_in);
			y = w.mtimes(xAppendHin).plus(b);
		} else {
			System.out.println("矩阵w与矩阵[h0,x1]行列不匹配，无法相乘");
		}
		// System.out.println("calculateVector y = " + y);
		return y;
	}

	public List<Matrix> getwList() {
		return wList;
	}

	public void setwList(List<Matrix> wList) {
		this.wList = wList;
	}

	public List<Matrix> getbList() {
		return bList;
	}

	public void setbList(List<Matrix> bList2) {
		this.bList = bList2;
	}

	private LstmCell() {

	}
}
