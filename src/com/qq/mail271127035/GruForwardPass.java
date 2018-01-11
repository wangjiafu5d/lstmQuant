package com.qq.mail271127035;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;

public class GruForwardPass {
	private InputLayer inputLayer;
	private GruLayer gruLayer;
	private Matrix w_output;
	private Matrix b_output;
	private List<Matrix> w_hidden_list;
	private List<Matrix> b_hidden_list;
	private Matrix w_input;
	private Matrix b_input;
	private List<Matrix> xList;
	private List<Matrix> resultList  = new ArrayList<Matrix>() ;
	private List<GruCell> gruLsit = new ArrayList<GruCell>();	
	private List<Matrix> htList = new ArrayList<Matrix>();
	public static GruForwardPass build(final Matrix w_input, final Matrix b_input, final List<Matrix> w_hidden_list,
			final List<Matrix> b_hidden_list,final Matrix w_output, final Matrix b_output,final List<Matrix> xList) {
		GruForwardPass gruForwardPass = new GruForwardPass();
		gruForwardPass.w_input = w_input;
		gruForwardPass.b_input = b_input;
		gruForwardPass.w_hidden_list = w_hidden_list;
		gruForwardPass.b_hidden_list = b_hidden_list;
		gruForwardPass.w_output = w_output;
		gruForwardPass.b_output = b_output;
		gruForwardPass.xList = xList;
		return gruForwardPass;
	}

	public void run(final int outputSize) {
		inputLayer = InputLayer.build(w_input, b_input, xList);
		List<Matrix> inputList = new ArrayList<>();
		for (Matrix matrix : inputLayer.out_list) {
			inputList.add(matrix.transpose());
		}
		gruLayer = GruLayer.build(w_hidden_list.get(0), w_hidden_list.get(1), w_hidden_list.get(2),b_hidden_list.get(0), b_hidden_list.get(1), b_hidden_list.get(2),
				inputList);
		gruLsit = gruLayer.getGruCellList();
		htList = gruLayer.out(outputSize);		
		for (int i = 0; i <htList.size(); i++) {
			resultList.add(InputCell.build(w_output, b_output).out(htList.get(i).transpose()));
		}
			
		

	}

	public List<Matrix> getResultList() {
		return resultList;
	}	

	public List<Matrix> getHtList() {
		return htList;
	}

	public List<GruCell> getGruLsit() {
		return gruLsit;
	}
	
}
