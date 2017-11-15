package com.qq.mail271127035;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;

import test.Test;

public class TrainThread extends Thread{
	int i = 0;
	double learning_rate = 0.0;
	List<Matrix> xList = new ArrayList<Matrix>();
	Matrix target = Matrix.Factory.rand(2, 1);
	List<Matrix> trained_list = new ArrayList<Matrix>();
	@Override
	public void run() {
//		System.out.println("learning_rate: "+ learning_rate);
//		System.out.println("i: "+ (i+8) + " target: "+target.transpose());
		ForwardPass forwardPass = new ForwardPass();
		forwardPass.add_w_input(Test.w_input).add_b_input(Test.b_input).add_w_hidden_list(Test.w_hidden_list)
				.add_b_hidden_list(Test.b_hidden_list).add_w_output(Test.w_output).add_b_output(Test.b_output)
				.add_xList(xList);
		forwardPass.run();
		Matrix out = forwardPass.getOutputLayer().getOut();
//		 System.out.println("i: "+ i + " " + out.transpose());

		Matrix delta_m = out.minus(target);
		 if (delta_m.getAsDouble(0, 0) > 0) {
		 delta_m.setAsDouble(delta_m.getAsDouble(0, 0) * 1.5, 0, 0);
		 }
		 if (delta_m.getAsDouble(1, 0) < 0) {
		 delta_m.setAsDouble(delta_m.getAsDouble(1, 0) * 1.5, 1, 0);
		 }
		Double delta = 0.5 * (Math.pow(delta_m.getAsDouble(0, 0), 2) + Math.pow(delta_m.getAsDouble(1, 0), 2));
//		System.out.println("loss"+i+" = "+delta);
//		System.out.println(" " + delta);
		Test.loss.add(delta);
		BackPass backPass = new BackPass().build(forwardPass.getInputLayer(), forwardPass.getLstmLayer(),
				forwardPass.getOutputLayer(), target, learning_rate);
		trained_list = backPass.backTrain(Test.w_output, Test.w_hidden_list, Test.w_input, xList);
		Test.trainedVectorLists.add(trained_list);
	}
	public int getI() {
		return i;
	}
	public void setI(int i) {
		this.i = i;
	}
	public Double getLearning_rate() {
		return learning_rate;
	}
	public void setLearning_rate(Double learning_rate) {
		this.learning_rate = learning_rate;
	}
	public List<Matrix> getxList() {
		return xList;
	}
	public void setxList(List<Matrix> xList) {
		this.xList = xList;
	}
	public Matrix getTarget() {
		return target;
	}
	public void setTarget(Matrix target) {
		this.target = target;
	}	
	
}
