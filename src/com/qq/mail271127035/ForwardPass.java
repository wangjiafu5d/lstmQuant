package com.qq.mail271127035;

import java.util.List;

import org.ujmp.core.Matrix;

public class ForwardPass {
	
	private InputLayer inputLayer;
	private LstmLayer lstmLayer;
	private OutputLayer outputLayer;
	private Matrix w_output;
	private Matrix b_output;
	private List<Matrix> w_hidden_list;
	private List<Matrix> b_hidden_list;
	private Matrix w_input;
	private Matrix b_input;
	private List<Matrix> xList;
	

	public ForwardPass run() {
		if (null != w_output && null != b_output && null != w_hidden_list && null != b_hidden_list && null != w_input
				&& null != b_input && null != xList) {
			inputLayer = InputLayer.build(w_input, b_input, xList);
			lstmLayer = LstmLayer.build(w_hidden_list, b_hidden_list, inputLayer.out_list);
			outputLayer = OutputLayer.build(w_output, b_output,
					lstmLayer.ht_out_list.get(lstmLayer.ht_out_list.size() - 1).transpose());
		} else {
			System.out.println("参数未初始化完成，请检查是否已经添加完所有参数");
		}
		return this;
	}

	public ForwardPass add_w_output(final Matrix w_output) {
		this.w_output = w_output;
		return this;
	}

	public ForwardPass add_b_output(final Matrix b_output) {
		this.b_output = b_output;
		return this;
	}

	public ForwardPass add_w_hidden_list(final List<Matrix> w_hidden_list) {
		this.w_hidden_list = w_hidden_list;
		return this;
	}

	public ForwardPass add_b_hidden_list(final List<Matrix> b_hidden_list) {
		this.b_hidden_list = b_hidden_list;
		return this;
	}

	public ForwardPass add_w_input(final Matrix w_input) {
		this.w_input = w_input;
		return this;
	}

	public ForwardPass add_b_input(final Matrix b_input) {
		this.b_input = b_input;
		return this;
	}

	public ForwardPass add_xList(List<Matrix> xList) {
		this.xList = xList;
		return this;
	}

	

	public InputLayer getInputLayer() {
		return inputLayer;
	}

	public void setInputLayer(InputLayer inputLayer) {
		this.inputLayer = inputLayer;
	}

	public LstmLayer getLstmLayer() {
		return lstmLayer;
	}

	public void setLstmLayer(LstmLayer lstmLayer) {
		this.lstmLayer = lstmLayer;
	}

	public OutputLayer getOutputLayer() {
		return outputLayer;
	}

	public void setOutputLayer(OutputLayer outputLayer) {
		this.outputLayer = outputLayer;
	}

	
}
