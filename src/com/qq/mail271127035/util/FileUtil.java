package com.qq.mail271127035.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.ujmp.core.Matrix;

public class FileUtil {
	public static Matrix readMatrix(String address) {
		File file = new File(address);
		List<List<Double>> list = new ArrayList<List<Double>>();
		if (file.exists() && file.isFile()) {
			BufferedReader br;
			try {
				br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
				String line = null;
				while ((line = br.readLine()) != null) {
					List<Double> l = new ArrayList<Double>();
					Pattern p = Pattern.compile("[-]{0,1}\\d+\\.{1}\\d+");
					Matcher m = p.matcher(line);
					while (m.find()) {
						// System.out.println(m.group());
						l.add(Double.valueOf(m.group()));
					}
					if (l.size() > 0) {
						list.add(l);
					}

				}
				br.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			try {
				throw new Exception();
			} catch (Exception e) {
				e.printStackTrace();
			}
			System.out.println("文件地址有误");
		}
		Matrix matrix = Matrix.Factory.zeros(list.size(), list.get(0).size());
		for (int row = 0; row < list.size(); row++) {
			for (int column = 0; column < list.get(0).size(); column++) {
				// System.out.println(row+" "+" "+column);
				matrix.setAsDouble(list.get(row).get(column), row, column);
			}
		}
		return matrix;
	}

	public static void writeMatix(String address, Matrix matrix) {
		File file = new File(address);
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {				
				e.printStackTrace();
			}
		}
		try {
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8"));
			String[][] outString = matrix.toStringArray();
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < outString.length; i++) {
				for (int j = 0; j < outString[0].length; j++) {
					sb.append(outString[i][j]).append("	");
				}
				sb.append("\r\n");
			}
			bw.write(sb.toString());
//			System.out.println(matrix);
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}
