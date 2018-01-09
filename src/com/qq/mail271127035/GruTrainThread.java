package com.qq.mail271127035;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;

import test.TestGru;

public class GruTrainThread extends Thread {
	private Matrix wInput;
	private Matrix bInput;
	private List<Matrix> wHiddenList;
	private List<Matrix> bHiddenList;
	private Matrix wOutput;
	private Matrix bOutput;
	private List<Matrix> xList;
	private int outputSize;
	private List<Matrix> targetList;

	public void build(Matrix wInput, Matrix bInput, List<Matrix> wHiddenList,List<Matrix> bHiddenList, Matrix wOutput, Matrix bOutput,
			List<Matrix> xList, int outputSize, List<Matrix> targetList) {
		this.wInput = wInput;
		this.bInput = bInput;
		this.wHiddenList = wHiddenList;
		this.bHiddenList = bHiddenList;
		this.wOutput = wOutput;
		this.bOutput = bOutput;
		this.xList = xList;
		this.outputSize = outputSize;
		this.targetList = targetList;
	}

	@Override
	public void run() {
		GruForwardPass gruForwardPass = GruForwardPass.build(wInput, bInput, wHiddenList,bHiddenList, wOutput, bOutput, xList);
		gruForwardPass.run(outputSize);
		List<Matrix> gradSumList = new ArrayList<Matrix>();
		List<Matrix> resultList = gruForwardPass.getResultList();
		List<GruCell> gruList = gruForwardPass.getGruLsit();
		Matrix deltaHtNext = gruForwardPass.getHtList().get(0).times(0);
		Double loss1 = 0.0;
		Double loss2 = 0.0;
		for (int i = 0; i < outputSize; i++) {			
			Matrix out = resultList.get(resultList.size() - 1 - i);
			Matrix target = targetList.get(targetList.size() - 1 - i);
			GruCell gruCell = gruList.get(gruList.size() -1 -i);
			Matrix xInput = xList.get(xList.size() - 1 - i);
			GruBackwardPass gruBackwardPass = GruBackwardPass.build(out, target, wOutput, gruCell, xInput,deltaHtNext);
			gruBackwardPass.run();
			deltaHtNext = gruBackwardPass.getDeltaHtPre();
			List<Matrix> gradList = gruBackwardPass.getGradList();
			
			if (gradSumList.size() == 0) {
				
				for (Matrix matrix : gradList) {
					gradSumList.add(matrix);
					
				}
			} else {
				for (int j = 0; j < gradList.size(); j++) {					
					gradSumList.add(gradSumList.get(j).plus(gradList.get(j)));
				}
			}
			if(i==0) {
			Matrix l = out.minus(target);
			for (int j = 0; j < l.getRowCount(); j++) {
				for (int j2 = 0; j2 < l.getColumnCount(); j2++) {
					loss1 += 0.5*l.getAsDouble(j,j2)*l.getAsDouble(j,j2);
				}
			}
			loss2 += Math.abs(l.getAsDouble(0,0));
			}
		}
	
		TestGru.gradList.add(gradSumList);
		TestGru.loss1.add(loss1);
		TestGru.loss2.add(loss2);
	}
	
}
