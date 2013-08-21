public class HelloWorld {
	public static void project(String imgPath) {
		System.out.println();
		System.out.println("Welcome to the OpenBR JNI transform! The image path string " + imgPath + " has been passed here from C++.");
		System.out.println("To incorporate your Java code into OpenBR assemble a jar file will of the relevant classes and save it in the openbr/share/openbr/Java/jniLibraries folder.");
		System.out.println("For documentation and additional resources on the JNI visit 'https://docs.oracle.com/javase/1.5.0/docs/guide/jni/specs/functions.html'");
		System.out.println();
	}
}