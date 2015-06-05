## bool contains(const [QString][QString] &name) {: #contains }

Check if a property exists in the [global metadata](context.md).

* **function definition:**

        bool contains(const QString &name);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    name | const [QString][QString] & | [Metadata](context.md) key. It must be queryable using [QObject::property][QObject::property].

* **output:** (bool) Returns true if the provided key is a global property.
* **see:** [setProperty](#setproperty)
* **example:**

        Globals->contains("path"); // returns true
        Globals->contains("key"); // returns false


## void setProperty(const [QString][QString] &key, const [QString][QString] &value) {: #setproperty }

Set a global property.

* **function definition:**

        void setProperty(const QString &key, const QString &value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | [Metadata](context.md) key
    value | const [QString][QString] & | Value to be added to the [Metadata](context.md)

* **output:** (void)
* **see:** [contains](#contains)
* **example:**

        Globals->contains("key"); // returns false
        Globals->setProperty("key", "value");
        Globals->contains("key"); // returns true


## void printStatus() {: #printstatus }

Prints the current progress statistics to **stdout**.

* **function definition:**

void printStatus();

* **parameters:** NONE
* **output:** (void)
* **see:** [progress](#progress)
* **example:**

        Globals->printStatus(); // returns 00.00%  ELAPSED=00:00:00  REMAINING=99:99:99  COUNT=0


## int timeRemaining() const {: #timeremaining }

Get the time remaining in seconds of a call to [Train](../apifunctions.md#train), [Enroll](../apifunctions.md#enroll-1) or [Compare](../apifunctions.md#compare).

* **function defintion:**

        int timeRemaining() const;

* **parameters:** NONE
* **output:** (int) Returns the estimated time remaining in the currently running process. If not process is running returns -1.

## float progress() {: #progress }

Get the completion percentage of a call to [Train](../apifunctions.md#train), [Enroll](../apifunctions.md#enroll-1) or [Compare](../apifunctions.md#compare).

* **function definition:**

        float progress() const;

* **parameters:** NONE
* **output:** (float) Returns the fraction of the currently running job that has been completed.

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QObject::property]: http://doc.qt.io/qt-5/qobject.html#property "QObject::property"
