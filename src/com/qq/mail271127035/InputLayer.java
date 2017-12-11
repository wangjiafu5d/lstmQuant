package com.qq.mail271127035;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;

/**
 * 由多个神经元inputCell并联构成的输入层InputLayer
 * 
 * @author chuan
 * @Date 2010-10-29
 * @version 1.0
 *
 */
public class InputLayer {

//	private static InputLayer inputLayer = new InputLayer();
	public List<Matrix> out_list = new ArrayList<Matrix>();

	/**
	 * 
	 * 输入设置好的系数w,b，和一段连续输入list_x，得到输出out_list和设置好的输入神经层；
	 * 
	 * @param w
	 *            矩阵乘法系数w矩阵
	 * @param b
	 *            矩阵加法系数b矩阵
	 * @param xList
	 *            总共有n个X_t作为输入，构成连续时间段内的输入，X_t为某个时刻的输入
	 * @return n个输入对应n个InputCell并联构成的InputLayer，返回构造好的Inputlayer
	 */
	public static InputLayer build(final Matrix w, final Matrix b, final List<Matrix> xList) {
		InputLayer inputLayer = new InputLayer();
//		inputLayer.out_list.clear();
		for (int i = 0; i < xList.size(); i++) {
			Matrix out = InputCell.build(w, b).inputCell_Out(xList.get(i));
			inputLayer.out_list.add(out);
		}

		return inputLayer;
	}

	private InputLayer() {

	}
}
