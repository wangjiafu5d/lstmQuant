package test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.ujmp.core.Matrix;

public class TrainTest {
	public static void main(String[] args) throws UnsupportedEncodingException, FileNotFoundException {
//		String fileName = "C:/Users/chuan/Desktop/data.txt";
		String fileName ="C:/Users/chuan/Desktop/data.txt";
		List<Double[]> list = new ArrayList<Double[]>();
		BufferedReader br;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(fileName)), "UTF-8"));
			String line = null;
			while ((line = br.readLine()) != null) {
			
//				String[] s = line.split(" 	");
				Double[] d = new Double[6];
				Pattern p = Pattern.compile("[-]{0,1}\\d+\\.{1}\\d+");
				Matcher m = p.matcher(line);
				int count = 0;
				while (m.find()) {
//					System.out.println(m.group());
					d[count] = Double.valueOf(m.group());
					count++;
				}
				
				list.add(d);

			}
			Matrix m = Matrix.Factory.zeros(list.size(),list.get(0).length);
			Double[][] matrix =new Double[list.size()][list.get(0).length];
			for (int i = 0; i < list.size(); i++) {
				matrix[i] = list.get(i);
				
			}
			m = Matrix.Factory.importFromArray(matrix);			
			m = m.times(10.0);
			for (int i = 0; i <m.getRowCount(); i++) {
				for (int j = 0; j < m.getColumnCount(); j++) {
					m.setAsDouble(Double.valueOf(list.get(i)[j]), i,j);
				}
			}
			System.out.println(m.times(10).toString());
		} catch (UnsupportedEncodingException e) {
			// TODO 自动生成的 catch 块
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO 自动生成的 catch 块
			e.printStackTrace();
		} catch (IOException e) {
			// TODO 自动生成的 catch 块
			e.printStackTrace();
		}
		Matrix w_input = Matrix.Factory.randn(6, 6);
		Matrix b_input = Matrix.Factory.randn(6, 1);
		List<Matrix> w_hidden_list = new ArrayList<Matrix>();
		List<Matrix> b_hidden_list = new ArrayList<Matrix>();
		Matrix w_output = Matrix.Factory.randn(2, 6);
		Matrix b_output = Matrix.Factory.randn(2, 1);
		for (int i = 0; i < 4; i++) {
			w_hidden_list.add(Matrix.Factory.randn(1, 2));
			b_hidden_list.add(Matrix.Factory.randn(1, 6));
		}
		File file = new File("C:/Users/chuan/Desktop/parameters.txt");
		try {
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8"));
			String[][] outString = new String[(int) w_input.getRowCount()][(int) w_input.getColumnCount()];
			outString = w_input.toStringArray();
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < outString.length; i++) {
				for (int j = 0; j < outString[0].length; j++) {
					sb.append(outString[i][j]).append("	");
				}
				sb.append("\r\n");
			}
			bw.write(sb.toString());
			System.out.println(sb.toString());
//			bw.newLine();
//			bw.write(b_input.toString());
//			bw.flush();
			bw.close();			
		} catch (IOException  e) {
			e.printStackTrace();
		}
	}
}
