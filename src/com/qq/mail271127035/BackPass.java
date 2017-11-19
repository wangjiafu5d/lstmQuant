package com.qq.mail271127035;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.ujmp.core.Matrix;

import com.qq.mail271127035.util.MathUtil;

/**
 * 实现反向传播算法的类，输入正向传播时的参数和计算结果，反向求梯度，然后得到修正后的矩阵参数。
 * 
 * @author chuan
 * @Date 2017-11-2
 * @version 1.0
 */
public class BackPass {

	public List<Matrix> xList = new ArrayList<Matrix>();
	public List<Matrix> bList = new ArrayList<Matrix>();
	public List<Matrix> momentum = new ArrayList<Matrix>();
	public List<Matrix> vt = new ArrayList<Matrix>();
	public Matrix ht;
	public Matrix out;
	public Matrix ct_out;
	public Matrix ct_prev;
	public Matrix ht_prev;
	public List<Matrix> last_cell_result;
	public Matrix target;
	public Double eta;
	public Matrix delta_ht;
	public Matrix delta_xt;

	public List<Matrix> backTrain(final Matrix w_output, final List<Matrix> w_hidden_list, final Matrix w_input,
			final List<Matrix> xList) {
		List<Matrix> wList = new ArrayList<Matrix>();
		// xList中为6×1矩阵，转化为1×6矩阵
		List<Matrix> x_trans_list = new ArrayList<Matrix>();
		for (Iterator<Matrix> iterator = xList.iterator(); iterator.hasNext();) {
			Matrix matrix = (Matrix) iterator.next();
			x_trans_list.add(matrix.transpose());
		}
		this.xList = x_trans_list;
		wList.add(backTrainOutputLayer(w_output));
		List<Matrix> new_hidden_w_list = backTrainHiddenLayer(w_hidden_list);
		for (int i = 0; i < new_hidden_w_list.size(); i++) {
			wList.add(new_hidden_w_list.get(i));
		}
		wList.add(backTrainInputLayer(w_input));
		return wList;
	}

	public BackPass build(final Matrix ht, final Matrix out, final Matrix ct_out, final Matrix ct_prev,
			final Matrix ht_prev, final List<Matrix> last_cell_result,final List<Matrix> momentum, final Matrix target, final Double eta) {
		BackPass backPass = new BackPass();
		backPass.ht = ht;
		backPass.out = out;
		backPass.ct_out = ct_out;
		backPass.ct_prev = ct_prev;
		backPass.ht_prev = ht_prev;
		backPass.last_cell_result = last_cell_result;
		backPass.momentum = momentum;
		backPass.target = target;
		backPass.eta = eta;
		return backPass;
	}

	private Matrix backTrainOutputLayer(final Matrix w_output) {
		
		delta_ht = Matrix.Factory.zeros(ht.getRowCount(), ht.getColumnCount());		
		Matrix delta_out = out.minus(target);
		Matrix delta_elu = Matrix.Factory.zeros(delta_out.getRowCount(), delta_out.getColumnCount());
		Matrix grad_node = Matrix.Factory.zeros(w_output.getRowCount(), w_output.getColumnCount());
		delta_elu = MathUtil.derivativeElu(out);
		grad_node = MathUtil.hadamard(delta_out, delta_elu).mtimes(ht);

		delta_ht = MathUtil.hadamard(delta_out, delta_elu).transpose().mtimes(w_output);
		delta_ht = MathUtil.gradCheck(delta_ht);
		Matrix v = momentum.get(0).times(0.9).plus(grad_node.times(eta));
		vt.add(v);
		return MathUtil.gradTrain(w_output, v, 1.0);
	}

	private List<Matrix> backTrainHiddenLayer(final List<Matrix> w_hidden_list) {
		List<Matrix> new_w_list = new ArrayList<Matrix>();
		
		// 最后一个lstm单元的输入Xt
		Matrix xt = xList.get(xList.size() - 1);		
		// 最后一次lstmCell计算的ft,it,ct_cell,ot
		Matrix ft = last_cell_result.get(0);
		Matrix it = last_cell_result.get(1);
		Matrix ct_cell = last_cell_result.get(2);
		// Matrix ct_out = last_cell_result.get(3);
		Matrix ot = last_cell_result.get(4);
		// [ht-1,xt]的转置,由于前面ht-1与xt合并时，xt作第一行，ht-1作第二行，构成2×6矩阵，所以转置为6×2矩阵
		Matrix in_trans = Matrix.Factory.zeros(xt.getColumnCount(), 2);
		for (int i = 0; i < xt.getColumnCount(); i++) {
			in_trans.setAsDouble(xt.getAsDouble(0, i), i, 0);
			in_trans.setAsDouble(ht_prev.getAsDouble(0, i), i, 1);
		}

		// tanh导数等于1-（tanh）^2
		// δCt = ot*(1-(tanhCt)^2)*δht
		Matrix grad_tanh_ct_out = Matrix.Factory.ones(ct_out.getRowCount(), ct_out.getColumnCount())
				.minus(MathUtil.hadamard(MathUtil.tanh(ct_out), MathUtil.tanh(ct_out)));
		Matrix delta_ct_out = MathUtil.seriesHadamard(ot, grad_tanh_ct_out, delta_ht);

		// 求训练得到的new_wf
		// grad_wf = delta_ct_out*Ct-1*ft*(1-ft)×([ht-1,xt]转置)
		Matrix ele1 = Matrix.Factory.ones(ft.getRowCount(), ft.getColumnCount()).minus(ft);
		// System.out.println("in_trans "+in_trans);
		Matrix grad_wf = MathUtil.seriesHadamard(delta_ct_out, ct_prev, ft, ele1).mtimes(in_trans);
		Matrix vf = momentum.get(1).times(0.9).plus(grad_wf.times(eta));
		vt.add(vf);
		Matrix new_wf = MathUtil.gradTrain(w_hidden_list.get(0), vf, 1.0);

		new_w_list.add(new_wf);
		// 求训练得到的new_wi
		// grad_wi = delta_ct_out*Ct_cell*it*(1-it)×([ht-1,xt]转置)
		Matrix ele2 = Matrix.Factory.ones(it.getRowCount(), it.getColumnCount()).minus(it);
		Matrix grad_wi = MathUtil.seriesHadamard(delta_ct_out, ct_cell, it, ele2).mtimes(in_trans);
		Matrix vi = momentum.get(2).times(0.9).plus(grad_wi.times(eta));
		vt.add(vi);
		Matrix new_wi = MathUtil.gradTrain(w_hidden_list.get(1), vi, 1.0);
		new_w_list.add(new_wi);
		// 求训练得到的new_wc
		// grad_wc = delta_ct_out*it*(1-ct_cell^2)×([ht-1,xt]转置)
		Matrix ele3 = Matrix.Factory.ones(ct_cell.getRowCount(), ct_cell.getColumnCount())
				.minus(MathUtil.hadamard(ct_cell, ct_cell));
		Matrix grad_wc = MathUtil.seriesHadamard(delta_ct_out, it, ele3).mtimes(in_trans);
		Matrix vc = momentum.get(3).times(0.9).plus(grad_wc.times(eta));
		vt.add(vc);
		Matrix new_wc = MathUtil.gradTrain(w_hidden_list.get(2), vc, 1.0);
		new_w_list.add(new_wc);
		// 求训练得到的new_wo
		// grad_wo = delta_ht*tanh(Ct)*ot(1-ot)*×([ht-1,xt]转置)
		Matrix ele4 = Matrix.Factory.ones(ot.getRowCount(), ot.getColumnCount()).minus(ot);
		Matrix grad_wo = MathUtil.seriesHadamard(delta_ht, MathUtil.tanh(ct_out), ot, ele4).mtimes(in_trans);
		Matrix vo = momentum.get(4).times(0.9).plus(grad_wo.times(eta));
		vt.add(vo);
		Matrix new_wo = MathUtil.gradTrain(w_hidden_list.get(3), vo, 1.0);
		new_w_list.add(new_wo);
		// 求训练得到的xt的梯度
		// grad_xt_o = delta_ht*tanh(Ct)*ot*(1-ot)*wo(0,0)
		// grad_xt_c = delta_ct_out*it*(1-ct_cell^2)*wc(0,0)
		// grad_xt_i = delta_ct_out*Ct_cell*it*(1-it)*wi(0,0)
		// grad_xt_f = delta_ct_out*Ct-1*ft*(1-ft)*wf(0,0)
		Matrix grad_xt_o = MathUtil.seriesHadamard(delta_ht, MathUtil.tanh(ct_out), ot, ele4)
				.times(w_hidden_list.get(3).getAsDouble(0, 0));
		Matrix grad_xt_c = MathUtil.seriesHadamard(delta_ct_out, it, ele3)
				.times(w_hidden_list.get(2).getAsDouble(0, 0));
		Matrix grad_xt_i = MathUtil.seriesHadamard(delta_ct_out, ct_cell, it, ele2)
				.times(w_hidden_list.get(1).getAsDouble(0, 0));
		Matrix grad_xt_f = MathUtil.seriesHadamard(delta_ct_out, ct_prev, ft, ele1)
				.times(w_hidden_list.get(0).getAsDouble(0, 0));
		delta_xt = grad_xt_o.plus(grad_xt_c).plus(grad_xt_i).plus(grad_xt_f);

		return new_w_list;
	}

	private Matrix backTrainInputLayer(final Matrix w_input) {
		Matrix result = Matrix.Factory.zeros(w_input.getRowCount(), w_input.getColumnCount());
		if (null != delta_xt) {
			Matrix grad_w_input = Matrix.Factory.zeros(w_input.getRowCount(), w_input.getColumnCount());
			// System.out.println("delta "+delta_xt.transpose());
			// System.err.println(xLis);
			grad_w_input = delta_xt.transpose().mtimes(xList.get(xList.size() - 1));
			Matrix v = momentum.get(5).times(0.9).plus(grad_w_input.times(eta));
			vt.add(v);
			result = MathUtil.gradTrain(w_input, v, 1.0);
			// System.out.println("eta "+eta);
		} else {
			System.out.println("程序执行顺序问题，grad_xt为空指针");
		}
		return result;
	}

	public List<Matrix> getVt() {
		return vt;
	}

}
