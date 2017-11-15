package test;

public class GetUserName {
	public static void main(String[] args) {
		String name = System.getenv("USERNAME");
		System.out.println(name);
	}
}
