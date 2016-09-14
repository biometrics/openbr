**www.openbiometrics.org**

1) Identify the latest stable [release tag](https://github.com/biometrics/openbr/releases) such as "v1.1.0"

2) Download all OpenBR source code and switch to that release tag:

    $ git clone https://github.com/biometrics/openbr.git
    $ cd openbr
    $ git checkout <tag>   (eg: git checkout v1.1.0)
    $ git submodule init
    $ git submodule update
    
3) Build OpenBR by following the **[Build Instructions](http://openbiometrics.org/docs/install/)** for your OS.
