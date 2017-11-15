package com.qq.mail271127035;

import java.util.ArrayList;
import java.util.List;
import org.ujmp.core.Matrix;

public class LstmLayer {
//	private static LstmLayer lstmLayer = new LstmLayer();
	public List<Matrix> ct_out_list = new ArrayList<Matrix>();
	public List<Matrix> ht_out_list = new ArrayList<Matrix>();
	public List<List<Matrix>> cells_result = new ArrayList<List<Matrix>>();

	/**
	 * 
	 * @param wList ,1×2
	 *            一系列乘法系数矩阵w的集合
	 * @param bList ,1×2
	 *            一系列加法系数矩阵b的集合
	 * @param xList ,1×6
	 *            一系列输入矩阵x的集合
	 * @return 经过lstm单元循环构成的计算网络，他保存有运算过程中得到的中间值和最终值，1×6
	 */
	public static LstmLayer build(final List<Matrix> wList, final List<Matrix> bList,
			final List<Matrix> xList) {
		LstmLayer lstmLayer = new LstmLayer();
//		lstmLayer.clearList();
		if (wList.size() > 0 && bList.size() > 0 && xList.size() > 0) {
			// 初始化h_in,ct_in；初始值是全零矩阵
			Matrix h_in = Matrix.Factory.zeros(xList.get(0).getRowCount(), xList.get(0).getColumnCount());
			Matrix ct_in = Matrix.Factory.zeros(xList.get(0).getRowCount(), xList.get(0).getColumnCount());
			// lstm循环层，每次输入一个xt矩阵，得到ht和ct矩阵，结果作为下一个xt+1矩阵的输入参数
			for (int i = 0; i < xList.size(); i++) {
				LstmCell lstmCell = LstmCell.build(wList, bList);
				List<Matrix> cell_out_list = lstmCell.lstmCell_Out(xList.get(i), h_in, ct_in);
				// 保存每个LstmCell单元内部的运算结果
				List<Matrix> cell_results = new ArrayList<Matrix>();
				for (int j = 0; j < lstmCell.lstmCell_result.size(); j++) {
					cell_results.add(lstmCell.lstmCell_result.get(j));
				}
				lstmLayer.cells_result.add(cell_results);
				ct_in = cell_out_list.get(0);
				h_in = cell_out_list.get(1);
				// 保存每个LstmCell的输出ht和Ct
				lstmLayer.ct_out_list.add(ct_in);
//				System.out.println("ct_in"+ct_in);
				lstmLayer.ht_out_list.add(h_in);
			}
		} else {
			System.out.println("输入参数矩阵有错误");
		}
		return lstmLayer;
	}

//	private void clearList() {
//		ct_out_list.clear();
//		ht_out_list.clear();
//		cells_result.clear();
//	}
}
