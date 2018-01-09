package com.qq.mail271127035;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import com.qq.mail271127035.util.MathUtil;

public class GruBackwardPass {
	private Matrix out;
	private Matrix target;
	private Matrix wOutput;
	private Matrix xInput;
	private GruCell gruCell;
	private Matrix deltaHtNext;

	private Matrix deltaHt;
	private Matrix deltaXEncoded;
	private Matrix deltaHtPre;
	private List<Matrix> gradList = new ArrayList<Matrix>();

	public static GruBackwardPass build(final Matrix out, final Matrix target, final Matrix wOutput,
			final GruCell gruCell, final Matrix xInput, Matrix deltaHtNext) {
		GruBackwardPass gruBackwardPass = new GruBackwardPass();
		gruBackwardPass.out = out;
		gruBackwardPass.target = target;
		gruBackwardPass.gruCell = gruCell;
		gruBackwardPass.wOutput = wOutput;
		gruBackwardPass.xInput = xInput;
		gruBackwardPass.deltaHtNext = deltaHtNext;
		return gruBackwardPass;
	}

	public void run() {
		Matrix ht = gruCell.getHt();
		backTrainOutputLayer(wOutput, ht);	
		backTrainHiddenLayer();		
		backTrainInputLayer();

	}

	private Matrix backTrainOutputLayer(final Matrix wOutput, final Matrix ht) {
		deltaHt = Matrix.Factory.zeros(ht.getRowCount(), ht.getColumnCount());
		Matrix deltaOut = out.minus(target);
		Matrix deltaElu = Matrix.Factory.zeros(deltaOut.getRowCount(), deltaOut.getColumnCount());
		Matrix gradWOutput = Matrix.Factory.zeros(wOutput.getRowCount(), wOutput.getColumnCount());
		deltaElu = MathUtil.deriveElu(out);
		Matrix gradBOutput = deltaOut.times(deltaElu).transpose();
		gradWOutput = gradBOutput.mtimes(ht);
		// MathUtil.gradCheck(gradNode);
		deltaHt = deltaHtNext.plus(deltaOut.times(deltaElu).mtimes(wOutput));
		// delta_ht = MathUtil.gradCheck(delta_ht);
		gradList.add(gradWOutput);
		gradList.add(gradBOutput);
		return gradWOutput;

	}

	private List<Matrix> backTrainHiddenLayer() {

		Matrix zt = gruCell.getZt();
		Matrix rt = gruCell.getRt();
		Matrix htCell = gruCell.getHtCell();
		Matrix wz = gruCell.getWz();
		Matrix wr = gruCell.getWr();
		Matrix wo = gruCell.getWo();
		Matrix htPrev = gruCell.getHtPre();
		Matrix xEncoded = gruCell.getXt();

		// deltaHtPre0 = delta_ht * (1-zt)
		Matrix deltaHtPre0 = deltaHt.times(Matrix.Factory.ones(zt.getRowCount(), zt.getColumnCount()).minus(zt));
		// deltaWo = (delta_ht * zt * (1 - htCell^2)) × [rt * ht-1,xt]转置
		Matrix ele0 = Matrix.Factory.ones(htCell.getRowCount(), htCell.getColumnCount()).minus(htCell).times(htCell);
		Matrix ele1 = deltaHt.times(zt).times(ele0);
		Matrix ele2 = rt.times(htPrev).appendVertically(Ret.LINK, xEncoded);
		Matrix gradWo = ele1.mtimes(ele2.transpose());
		// deltaX0 = wo转置 × (delta_ht * zt * (1 - htCell^2));
		// deltaHtPre1 = rt * deltaX3.selectRows(0);
		// deltaXt0 = deltaX3.selectRows(1);
		Matrix deltaX0 = wo.transpose().mtimes(ele1);
		Matrix deltaHtPre1 = rt.times(deltaX0.selectRows(Ret.LINK, 0));
		Matrix deltaXt0 = deltaX0.selectRows(Ret.LINK, 1);
		// deltaRt = htPre * deltaX3.selectRows(0);
		Matrix deltaRt = htPrev.times(deltaX0.selectRows(Ret.LINK, 0));

		// deltaWr = (deltaRt * rt * (1 - rt)) × [ht-1,xt]转置
		// deltaX1 = wr转置 × (deltaRt * rt * (1 - rt));
		// deltaHtPre2 = deltaX1.selectRows(0);
		// deltaXt1 = deltaX1.selectRows(1);
		Matrix ele3 = htPrev.appendVertically(Ret.LINK, xEncoded).transpose();
		Matrix ele4 = deltaRt.times(rt).times(Matrix.Factory.ones(rt.getRowCount(), rt.getColumnCount()).minus(rt));
		Matrix gradWr = ele4.mtimes(ele3);
		Matrix deltaX1 = wr.transpose().mtimes(ele4);
		Matrix deltaHtPre2 = deltaX1.selectRows(Ret.LINK, 0);
		Matrix deltaXt1 = deltaX1.selectRows(Ret.LINK, 1);

		// deltaZt = delta_ht * (htCell - htPrev)
		// deltaWz = (deltaZt * zt *(1 - zt)) × [ht-1,xt]转置
		// deltaX2 = wz转置 × (deltaZt * zt * (1 - zt));
		// deltaHtPre3 = deltaX1.selectRows(0);
		// deltaXt2 = deltaX1.selectRows(1);
		Matrix deltaZt = deltaHt.times(htCell.minus(htPrev));
		Matrix ele5 = deltaZt.times(zt).times(Matrix.Factory.ones(zt.getRowCount(), zt.getColumnCount()).minus(zt));
		Matrix gradWz = ele5.mtimes(ele3);
		Matrix deltaX2 = wz.transpose().mtimes(ele5);
		Matrix deltaHtPre3 = deltaX2.selectRows(Ret.LINK, 0);
		Matrix deltaXt2 = deltaX2.selectRows(Ret.LINK, 1);

		deltaHtPre = deltaHtPre0.plus(deltaHtPre1).plus(deltaHtPre2).plus(deltaHtPre3);
		deltaXEncoded = deltaXt0.plus(deltaXt1).plus(deltaXt2);		
		gradList.add(gradWz);
		gradList.add(gradWr);
		gradList.add(gradWo);

		return gradList;
	}

	private Matrix backTrainInputLayer() {
		Matrix xEncoded = gruCell.getXt();
		Matrix gradWInput = Matrix.Factory.zeros(0, 0);
		Matrix gradBInput = Matrix.Factory.zeros(0, 0);
		if (null != deltaXEncoded) {
			Matrix deltaElu = MathUtil.deriveElu(xEncoded);
//			Matrix deltaElu = Matrix.Factory.ones(xEncoded.getRowCount(),xEncoded.getColumnCount()).minus(xEncoded.times(xEncoded));
			gradBInput = deltaXEncoded.times(deltaElu).transpose();
			gradWInput = gradBInput.mtimes(xInput.transpose());
		} else {
			System.out.println("程序执行顺序问题，grad_xt为空指针");
		}
		gradList.add(gradWInput);
		gradList.add(gradBInput);
		return gradWInput;
	}

	public Matrix getDeltaHtPre() {
		return deltaHtPre;
	}

	public List<Matrix> getGradList() {
		return gradList;
	}

}
