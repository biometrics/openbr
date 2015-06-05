Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=sdkpath></a>sdkPath | [QString][QString] | Path to the sdk. Path + **share/openbr/openbr.bib** must exist.
<a class="table-anchor" id=algorithm></a>algorithm | [QString][QString] | The default algorithm to use when enrolling and comparing templates.
<a class="table-anchor" id=log></a>log | [QString][QString] | Optional log file to copy **stderr** to.
<a class="table-anchor" id=path></a>path | [QString][QString] | Path to use when resolving images specified with relative paths. Multiple paths can be specified using a semicolon separator.
<a class="table-anchor" id=parallelism></a>parallelism | int | The number of threads to use. The default is the maximum of 1 and the value returned by ([QThread][QThread]::idealThreadCount() + 1).
<a class="table-anchor" id=usegui></a>useGui | bool | Whether or not to use GUI functions. The default is true.
<a class="table-anchor" id=blocksize></a>blockSize | int | The maximum number of templates to process in parallel. The default is: ```parallelism * ((sizeof(void*) == 4) ? 128 : 1024)```
<a class="table-anchor" id=quiet></a>quiet | bool | If true, no messages will be sent to the terminal. The default is false.
<a class="table-anchor" id=verbose></a>verbose | bool | If true, extra messages will be sent to the terminal. The default is false.
<a class="table-anchor" id=mostrecentmessage></a>mostRecentMessage | [QString][QString] | The most recent message sent to the terminal.
<a class="table-anchor" id=currentstep></a>currentStep | double | Used internally to compute [progress](functions.md#progress) and [timeRemaining](functions.md#timeremaining).
<a class="table-anchor" id=totalsteps></a>totalSteps | double | Used internally to compute [progress](functions.md#progress) and [timeRemaining](functions.md#timeremaining).
<a class="table-anchor" id=enrollall></a>enrollAll | bool | If true, enroll 0 or more templates per image. Otherwise, enroll exactly one. The default is false.
<a class="table-anchor" id=filters></a>filters | Filters | Filters is a ```typedef QHash<QString,QStringList> Filters```. Filters that automatically determine imposter matches based on target ([gallery](../gallery/gallery.md)) template metadata. See [FilterDistance](../../../plugin_docs/distance.md#filterdistance).
<a class="table-anchor" id=buffer></a>buffer | [QByteArray][QByteArray] | File output is redirected here if the file's basename is "buffer". This clears previous contents.
<a class="table-anchor" id=scorenormalization></a>scoreNormalization | bool | If true, enable score normalization. Otherwise disable it. The default is true.
<a class="table-anchor" id=crossvalidate></a>crossValidate | int | Perform k-fold cross validation where k is the value of **crossValidate**. The default value is 0.
<a class="table-anchor" id=modelsearch></a>modelSearch | [QList][QList]&lt;[QString][QString]&gt; | List of paths to search for sub-models on.
<a class="table-anchor" id=abbreviations></a>abbreviations | [QHash][QHash]&lt;[QString][QString], [QString][QString]&gt; | Used by [Transform](../transform/transform.md)::[make](../transform/statics.md#make) to expand abbreviated algorithms into their complete definitions.
<a class="table-anchor" id=starttime></a>startTime | [QTime][QTime] | Used to estimate [timeRemaining](functions.md#timeremaining).
<a class="table-anchor" id=logfile></a>logFile | [QFile][QFile] | Log file to write to.


<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[QHash]: http://doc.qt.io/qt-5/qhash.html "QHash"
[QThread]: http://doc.qt.io/qt-5/qthread.html "QThread"
[QByteArray]: http://doc.qt.io/qt-5/qbytearray.html "QByteArray"
[QTime]: http://doc.qt.io/qt-5/QTime.html "QTime"
[QFile]: http://doc.qt.io/qt-5/qfile.html "QFile"
