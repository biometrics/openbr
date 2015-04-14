<!-- INITIALIZER -->

Inherits [Object](#object)

Plugin base class for initializing resources. On startup (the call to [Context](#context)::(initialize)(#context-static-initialize)), OpenBR will call [initialize](#initializer-function-initialize) on every Initializer that has been registered with the [Factory](#factory). On shutdown (the call to [Context](#context)::(finalize)(#context-static-finalize)), OpenBR will call [finalize](#initializer-function-finalize) on every registered initializer.

The general use case for initializers is to launch shared contexts for third party integrations into OpenBR. These cannot be launched during [Transform](#transform)::(init)(#transform-function-init) for example, because multiple instances of the [Transform](#transform) object could exist across multiple threads.

## Members {: #initializer-members }

NONE

## Constructors {: #initializer-constructors }

Destructor | Description
--- | ---
virtual ~Initializer() | Virtual function. Default destructor.

## Static Functions {: #initializer-static-functions }

NONE

## Functions {: #initializer-functions }

### virtual void initialize() const {: #initializer-function-initialize }

Initialize

* **function definition:**

        virtual void initialize() const = 0
