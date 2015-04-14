<!-- CONTEXT -->

The singleton class of global settings. Before including and using OpenBR in a project the user must call [initialize](#context-static-initialize). Before the program terminates the user must call [finalize](#context-static-finalize). The settings are accessible as Context \*Globals.

## Members {: #context-members }

Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=context-members-sdkpath></a>sdkPath | [QString][QString] | Path to the sdk. Path + **share/openbr/openbr.bib** must exist.
<a class="table-anchor" id=context-members-algorithm></a>algorithm | [QString][QString] | The default algorithm to use when enrolling and comparing templates.
<a class="table-anchor" id=context-members-log></a>log | [QString][QString] | Optional log file to copy **stderr** to.
<a class="table-anchor" id=context-members-path></a>path | [QString][QString] | Path to use when resolving images specified with relative paths. Multiple paths can be specified using a semicolon separator.
<a class="table-anchor" id=context-members-parallelism></a>parallelism | int | The number of threads to use. The default is the maximum of 1 and the value returned by ([QThread][QThread]::idealThreadCount() + 1).
<a class="table-anchor" id=context-members-usegui></a>useGui | bool | Whether or not to use GUI functions. The default is true.
<a class="table-anchor" id=context-members-blocksize></a>blockSize | int | The maximum number of templates to process in parallel. The default is: ```parallelism * ((sizeof(void*) == 4) ? 128 : 1024)```
<a class="table-anchor" id=context-members-quiet></a>quiet | bool | If true, no messages will be sent to the terminal. The default is false.
<a class="table-anchor" id=context-members-verbose></a>verbose | bool | If true, extra messages will be sent to the terminal. The default is false.
<a class="table-anchor" id=context-members-mostrecentmessage></a>mostRecentMessage | [QString][QString] | The most recent message sent to the terminal.
<a class="table-anchor" id=context-members-currentstep></a>currentStep | double | Used internally to compute [progress](#context-function-progress) and [timeRemaining](#context-function-timeremaining).
<a class="table-anchor" id=context-members-totalsteps></a>totalSteps | double | Used internally to compute [progress](#context-function-progress) and [timeRemaining](#context-function-timeremaining).
<a class="table-anchor" id=context-members-enrollall></a>enrollAll | bool | If true, enroll 0 or more templates per image. Otherwise, enroll exactly one. The default is false.
<a class="table-anchor" id=context-members-filters></a>filters | Filters | Filters is a ```typedef QHash<QString,QStringList> Filters```. Filters that automatically determine imposter matches based on target ([gallery](#gallery)) template metadata. See [FilterDistance](plugins/distance.md#filterdistance).
<a class="table-anchor" id=context-members-buffer></a>buffer | [QByteArray][QByteArray] | File output is redirected here if the file's basename is "buffer". This clears previous contents.
<a class="table-anchor" id=context-members-scorenormalization></a>scoreNormalization | bool | If true, enable score normalization. Otherwise disable it. The default is true.
<a class="table-anchor" id=context-members-crossValidate></a>crossValidate | int | Perform k-fold cross validation where k is the value of **crossValidate**. The default value is 0.
<a class="table-anchor" id=context-members-modelsearch></a>modelSearch | [QList][QList]&lt;[QString][QString]&gt; | List of paths to search for sub-models on.
<a class="table-anchor" id=context-members-abbreviations></a>abbreviations | [QHash][QHash]&lt;[QString][QString], [QString][QString]&gt; | Used by [Transform](#transform)::[make](#transform-function-make) to expand abbreviated algorithms into their complete definitions.
<a class="table-anchor" id=context-members-starttime></a>startTime | [QTime][QTime] | Used to estimate [timeRemaining](#context-function-timeremaining).
<a class="table-anchor" id=context-members-logfile></a>logFile | [QFile][QFile] | Log file to write to.

---

## Constructors {: #context-constructors }

NONE

---

## Static Functions {: #context-static-functions }

### void initialize(int &argc, char \*argv[], [QString][QString] sdkPath = "", bool useGui = true) {: #context-static-initialize }

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
* **see:** [finalize](#context-static-finalize)
* **example:**

        int main(int argc, char \*argv[])
        {
            QApplication(argc, argv); // ONLY FOR QT PROJECTS
            br::Context::initialize(argc, argv);

            // ...

            br::Context::finalize();
            return 0;
        }

### void finalize() {: #context-static-finalize }

Call *once* at the end of the application to deallocate global variables.

* **function definition:**

        static void finalize();

* **parameters:** NONE
* **output:** (void)
* **see:** [initialize](#context-static-initialize)


### bool checkSDKPath(const [QString][QString] &sdkPath) {: #context-static-checksdkpath }

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

### [QString][QString] about() {: #context-static-about }

Get a string with the name, version, and copyright of the project. This string is suitable for printing or terminal.

* **function definition:**

        static QString about();

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a string containing the name, version and copyright of the project
* **example:**

        // Using OpenBR version 0.6.0
        Context::about(); // returns "OpenBR 0.6.0 Copyright (c) 2013 OpenBiometrics. All rights reserved."

### [QString][QString] version() {: #context-static-version }

Get the version of the SDK.

* **function definition:**

        static QString version();

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a string containing the version of the OpenBR SDK. The string has the format *<MajorVersion\>*\.*<MinorVersion\>*\.*<PatchVersion\>*
* **example:**

        // Using OpenBR version 0.6.0
        Context::version(); // returns "0.6.0"

### [QString][QString] scratchPath() {: #context-static-scratchpath }

Get the scratch directory used by OpenBR. This directory should be used as the root directory for managing temporary files and providing process persistence.

* **function definition:**

        static QString scratchPath();

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a string pointing to the OpenBR scratch directory. The string has the format *<path/to/user/home\><OpenBR-\><MajorVersion\>*\.*<MinorVersion\>*.
* **see:** [version](#context-static-version)
* **example:**

        // Using OpenBR version 0.6.0
        Context::scratchPath(); // returns "/path/to/user/home/OpenBR-0.6"

### [QStringList][QStringList] objects(const char \*abstractions = ".\*", const char \*implementations = ".\*", bool parameters = true) {: #context-static-objects }

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

<!-- no italics* -->

---

## Functions {: #context-functions }


### bool contains(const [QString][QString] &name) {: #context-function-contains }

Check if a property exists in the [global metadata](#context).

* **function definition:**

        bool contains(const QString &name);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    name | const [QString][QString] & | [Metadata](#context) key. It must be queryable using [QObject::property][QObject::property].

* **output:** (bool) Returns true if the provided key is a global property.
* **see:** [setProperty](#context-function-setproperty)
* **example:**

        Globals->contains("path"); // returns true
        Globals->contains("key"); // returns false


### void setProperty(const [QString][QString] &key, const [QString][QString] &value) {: #context-function-setproperty }

Set a global property.

* **function definition:**

        void setProperty(const QString &key, const QString &value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | [Metadata](#context) key
    value | const [QString][QString] & | Value to be added to the [Metadata](#context)

* **output:** (void)
* **see:** [contains](#context-function-contains)
* **example:**

        Globals->contains("key"); // returns false
        Globals->setProperty("key", "value");
        Globals->contains("key"); // returns true


### void printStatus() {: #context-function-printstatus }

Prints the current progress statistics to **stdout**.

* **function definition:**

void printStatus();

* **parameters:** NONE
* **output:** (void)
* **see:** [progress](#context-function-progress)
* **example:**

        Globals->printStatus(); // returns 00.00%  ELAPSED=00:00:00  REMAINING=99:99:99  COUNT=0


### int timeRemaining() const {: #context-function-timeremaining }

Get the time remaining in seconds of a call to [Train](#function-train), [Enroll](#function-enroll-1) or [Compare](#function-compare).

* **function defintion:**

        int timeRemaining() const;

* **parameters:** NONE
* **output:** (int) Returns the estimated time remaining in the currently running process. If not process is running returns -1.

### float progress() {: #context-function-progress }

Get the completion percentage of a call to [Train](#function-train), [Enroll](#function-enroll-1), or [Compare](#function-compare).

* **function definition:**

        float progress() const;

* **parameters:** NONE
* **output:** (float) Returns the fraction of the currently running job that has been completed.
