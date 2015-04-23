//Need to include location of jvm.dll (jdk version) and its parent directory in the environment variables

#include <limits>
#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/resource.h>
#include <jni.h>

namespace br
{

/*!
 * \ingroup initializers
 * \brief Initialize JNI
 * \author Jordan Cheney \cite jcheney
 */
class JNIInitializer : public Initializer
{   
    Q_OBJECT
    public:
        static JavaVM* jvm;
        static JavaVMInitArgs vm_args;

    void initialize() const
    {
        JNIEnv *env;
        JavaVMOption options[1];

        //Location of Java files
        QByteArray classpath = QString("-Djava.class.path=").append(Globals->sdkPath).append(QString("/share/openbr/Java/jniLibraries/")).toLocal8Bit();
        char *charClasspath = classpath.data();

        options[0].optionString = charClasspath;
        vm_args.version = JNI_VERSION_1_6;
        vm_args.nOptions = 1;
        vm_args.options = options;
        vm_args.ignoreUnrecognized = JNI_FALSE;

        JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args);

        Globals->abbreviations.insert("JNIHelloWorld","Open+JNI(HelloWorld)");
    }

    void finalize() const
    {
        jvm->DestroyJavaVM();
    }
};

JavaVM *JNIInitializer::jvm;
JavaVMInitArgs JNIInitializer::vm_args;

BR_REGISTER(Initializer, JNIInitializer)

/*!
 * \ingroup transforms
 * \brief Execute Java code from OpenBR using the JNI
 * \author Jordan Cheney \cite jcheney
 */
class JNITransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString className READ get_className WRITE set_className RESET reset_className STORED false)
    BR_PROPERTY(QString, className, "")

    void project(const Template &src, Template &dst) const
    {
        (void)dst; //Eliminates a compiler warning.

		JNIEnv *env;

        //Attach current thread to the thread of the JavaVM and access env
        JNIInitializer::jvm->AttachCurrentThreadAsDaemon((void**)&env, NULL);
        if (JNIInitializer::jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
            qFatal("Failed to initialize JNI environment");
        }

        //Convert QString to const char*
        QByteArray tmpClass = className.toLocal8Bit();
        const char* charClassName = tmpClass.constData();

        //Call java method
        jclass cls = env->FindClass(charClassName);
        if (cls == NULL) { qFatal("Class not found"); }
        jmethodID mid = env->GetStaticMethodID(cls, "project", "(Ljava/lang/String;)V");
        if (mid == NULL) { qFatal("MethodID not found"); }

        QByteArray tmp = src.file.name.toLocal8Bit();
        const char* fileName = tmp.constData();

        //Convert char* to java compatible string
        jstring jfileName = env->NewStringUTF(fileName);

        env->CallStaticObjectMethod(cls, mid, jfileName);

        JNIInitializer::jvm->DetachCurrentThread();
	}
};

BR_REGISTER(Transform, JNITransform)

} // namespace br

#include "jni.moc"
