## void initialize(int &argc, char \*argv[], [QString][QString] sdkPath = "", bool useGui = true) {: #initialize }

Call *once* at the start of the application to allocate global variables. If the project is a [Qt][Qt] project this call should occur after initializing <tt>QApplication</tt>.

* **function definition:**

        static void initialize(int &argc, char *argv[], QString sdkPath = "", bool useGui = true);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    argc | int & | Number of command line arguments as provided by <tt>main()</tt>
    argv | char * [] | Command line arguments as provided by <tt>main()</tt>
    sdkPath | [QString][QString] | (Optional) The path to the folder containing **share/openbr/openbr.bib**. If no path is provided (default) OpenBR automatically searches: <ul> <li>The working directory</li> <li>The executable's location</li> </ul>
    useGui | bool | (Optional) Make OpenBR as a [QApplication][QApplication] instead of a [QCoreApplication][QCoreApplication]. Default is true.

* **output:** (void)
* **see:** [finalize](#finalize)
* **example:**

        int main(int argc, char \*argv[])
        {
            QApplication(argc, argv); // ONLY FOR QT PROJECTS
            br::Context::initialize(argc, argv);

            // ...

            br::Context::finalize();
            return 0;
        }

## void finalize() {: #finalize }

Call *once* at the end of the application to deallocate global variables.

* **function definition:**

        static void finalize();

* **parameters:** NONE
* **output:** (void)
* **see:** [initialize](#initialize)


## bool checkSDKPath(const [QString][QString] &sdkPath) {: #checksdkpath }

Check if a given SDK path is valid. A valid SDK satisfies

    exists(sdkPath + "share/openbr/openbr.bib")

* **function definition:**

        static bool checkSDKPath(const QString &sdkPath);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    sdkPath | const [QString][QString] & | Possible sdk path to examine

* **output:** (bool) Returns true if the sdkPath + "share/openbr/openbr.bib" exists, otherwise returns false.
* **example:**

        // OpenBR is at /libs/openbr

        checkSDKPath("/libs/openbr/"); // returns true
        checkSDKPath("/libs/"); // returns false

## [QString][QString] about() {: #about }

Get a string with the name, version, and copyright of the project. This string is suitable for printing or terminal.

* **function definition:**

        static QString about();

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a string containing the name, version and copyright of the project
* **example:**

        // Using OpenBR version 0.6.0
        Context::about(); // returns "OpenBR 0.6.0 Copyright (c) 2013 OpenBiometrics. All rights reserved."

## [QString][QString] version() {: #version }

Get the version of the SDK.

* **function definition:**

        static QString version();

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a string containing the version of the OpenBR SDK. The string has the format *<MajorVersion\>*\.*<MinorVersion\>*\.*<PatchVersion\>*
* **example:**

        // Using OpenBR version 0.6.0
        Context::version(); // returns "0.6.0"

## [QString][QString] scratchPath() {: #scratchpath }

Get the scratch directory used by OpenBR. This directory should be used as the root directory for managing temporary files and providing process persistence.

* **function definition:**

        static QString scratchPath();

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a string pointing to the OpenBR scratch directory. The string has the format *<path/to/user/home\><OpenBR-\><MajorVersion\>*\.*<MinorVersion\>*.
* **see:** [version](#version)
* **example:**

        // Using OpenBR version 0.6.0
        Context::scratchPath(); // returns "/path/to/user/home/OpenBR-0.6"

## [QStringList][QStringList] objects(const char \*abstractions = ".\*", const char \*implementations = ".\*", bool parameters = true) {: #objects }

Get a collection of objects in OpenBR that match provided regular expressions. This function uses [QRegExp][QRegExp] syntax.

* **function definition:**

        static QStringList objects(const char *abstractions = ".*", const char *implementations = ".*", bool parameters = true)

* **parameters:**

        Parameter | Type | Description
        --- | --- | ---
        abstractions | const char \* | (Optional) Regular expression of the abstractions to search. Default is ".\*"
        implementations | const char \* | (Optional) Regular expression of the implementations to search. Default is ".\*".
        parameters | bool | (Optional) If true include parameters after object name. Default is true.

* **output:** ([QStringList][QStringList]) Return names and parameters for the requested objects. Each object is newline separated. Arguments are separated from the object name with tabs.
* **example:**

        // Find all 'Rnd' Transforms
        Context::objects("Transform", "Rnd.*", false); // returns ["RndPoint", "RndRegion", "RndRotate", "RndSample", "RndSubspace"]

<!-- Links -->
[Qt]: http://qt-project.org/ "Qt"
[QApplication]: http://doc.qt.io/qt-5/qapplication.html "QApplication"
[QCoreApplication]: http://doc.qt.io/qt-5/qcoreapplication.html "QCoreApplication"

[QRegExp]: http://doc.qt.io/qt-5/QRegExp.html "QRegExp"

[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QStringList]: http://doc.qt.io/qt-5/qstringlist.html "QStringList"
