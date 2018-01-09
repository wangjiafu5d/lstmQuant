package com.qq.mail271127035;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;

public class GruLayer {
	private Matrix wz;
	private Matrix wr;
	private Matrix wo;
	private Matrix bz;
	private Matrix br;
	private Matrix bo;
	private List<Matrix> xList;
	private List<Matrix> htList = new ArrayList<Matrix>();
	private List<GruCell> gruCellList = new ArrayList<GruCell>();

	public static GruLayer build(final Matrix wz, final Matrix wr, final Matrix wo, final Matrix bz, final Matrix br, final Matrix bo, final List<Matrix> xList) {
		GruLayer gruLayer = new GruLayer();
		gruLayer.xList = xList;
		gruLayer.wz = wz;
		gruLayer.wr = wr;
		gruLayer.wo = wo;
		gruLayer.bz = bz;
		gruLayer.br = br;
		gruLayer.bo = bo;
		return gruLayer;
	}

	public List<Matrix> out(int outputSize) {
		Matrix htPre = Matrix.Factory.zeros(xList.get(0).getRowCount(), xList.get(0).getColumnCount());

		for (int i = 0; i < xList.size(); i++) {
			GruCell gruCell = GruCell.build(wz, wr, wo, bz,br,bo, htPre, xList.get(i));
			Matrix ht = gruCell.gruOut();
			htPre = ht;
			if (i > xList.size() - outputSize - 1) {			
				gruCellList.add(gruCell);
				htList.add(ht);
			}
		}

		return htList;
	}

	public List<GruCell> getGruCellList() {
		return gruCellList;
	}
	
}
