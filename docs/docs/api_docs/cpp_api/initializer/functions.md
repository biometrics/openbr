## void initialize() {: #initialize }

This is a pure virtual function. It is called once at the end of [initialize](../context/statics.md#initialize). Any global initialization that needs to occur should occur within this function.

* **function definition:**

        virtual void initialize() const = 0

* **parameters:** NONE
* **output:** (void)

## void finalize() {: #finalize }

This is a virtual function. It is called once at the beginning of [finalize](../context/statics.md#finalize). Any global finalization should occur within this function. This includes deallocating anything that was allocated in [initialize](#initialize)

* **function definition**:

        virtual void finalize() const

* **parameters:** NONE
* **output:** (void)
