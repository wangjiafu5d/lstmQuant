package com.qq.mail271127035;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import com.qq.mail271127035.util.MathUtil;
import com.qq.mail271127035.util.MyMatrixUtil;

public class GruCell {
	private Matrix wz;
	private Matrix wr;
	private Matrix wo;
	private Matrix htPre;
	private Matrix xt;
	private Matrix zt;
	private Matrix rt;
	private Matrix htCell;
	private Matrix ht;
	private Matrix bz;
	private Matrix br;
	private Matrix bo;

	public static GruCell build(final Matrix wz, final Matrix wr, final Matrix wo, final Matrix bz, final Matrix br, final Matrix bo,final Matrix htPre,
			final Matrix xt) {
		GruCell gruCell = new GruCell();
		gruCell.wz = wz;
		gruCell.wr = wr;
		gruCell.wo = wo;
		gruCell.htPre = htPre;
		gruCell.xt = xt;
		gruCell.bz = bz;
		gruCell.br = br;
		gruCell.bo = bo;

		return gruCell;
	}

	public Matrix gruOut() {
		Matrix htPreAppendXt = htPre.appendVertically(Ret.LINK, xt);
		zt = MathUtil.sigmoid(wz.mtimes(htPreAppendXt).plus(bz));
		rt = MathUtil.sigmoid(wr.mtimes(htPreAppendXt).plus(br));
		htCell = wo.mtimes(rt.times(htPre).appendVertically(Ret.LINK, xt)).plus(bo).tanh();
		ht = MyMatrixUtil.copyOnesMatrix(zt).minus(zt).times(htPre).plus(zt.times(htCell));

		return ht;
	}

	public Matrix getWz() {
		return wz;
	}

	public Matrix getWr() {
		return wr;
	}

	public Matrix getWo() {
		return wo;
	}

	public Matrix getHtPre() {
		return htPre;
	}

	public Matrix getXt() {
		return xt;
	}

	public Matrix getZt() {
		return zt;
	}

	public Matrix getRt() {
		return rt;
	}

	public Matrix getHtCell() {
		return htCell;
	}

	public Matrix getHt() {
		return ht;
	}

}
