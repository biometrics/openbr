Contributing to OpenBR should be straightforward and enjoyable. This guide elucidates some guidelines such that your contribution fits nicely into our framework.

## Contributing a Plugin

You should consider contributing a plugin (or plugins!) if you have an algorithm that you would like to express via OpenBR's algorithm grammar or you have an extension to an existing algorithm. Below are the steps for creating and contributing a plugin:

1. Check out the [C++ Plugin API](api_docs/cpp_api.md) and decide which abstraction best suits the needs of your contribution.
2. Select a module in the `openbr/plugins/` directory that describes your plugin, then create a new source (`.cpp`) file in that directory. Your file should have the same name as your plugin.
3. Implement your plugin! Make sure to adhere to the [Style Guide](#style-guide) to keep the code consistent within OpenBR. This increases overall readability and makes it easier for newcomers to learn!

### Common Mistakes

Some common mistakes that will stop your plugin from working:

* `#include` the  `<openbr/plugins/openbr_internal.h>` header.
* The entire plugin should be inside `namespace br`.
* Make sure your plugin declares `Q_OBJECT` right after it's definition.
* Remember to call [BR_REGISTER](api_docs/cpp_api/factory/macros.md#br_register) at the end of your plugin.
* Remeber to add `#include "module/filename.moc"` at the very bottom of your file.

When in doubt, check out existing [Transforms](api_docs/cpp_api/transform/transform.md). [MAddTransform](plugin_docs/imgproc.md#maddtransform) is a simple and clear example of how a [Transform](api_docs/cpp_api/transform/transform.md) should look.

### Documenting

Documenting your plugin is very important. OpenBR supports custom, doxygen-style, in-code comments to make documentation simple, clear, and easy. Comments should be written as:

        /*!
         * ...
         */

Comments are organized using tags, which are of the form `\tag`. There are a few *required* tags that all OpenBR transforms must have:

Tag | Description
--- | ---
\ingroup | The abstraction to which your plugin belongs
\brief | A description of your plugin
\author | Your name
\cite | The citation link for the author. There must be citation tag for every author who appears. If you haven't already please add your information to <tt>openbr/docs/docs/contributors.md</tt>
\br_property | Describes a <tt>BR_PROPERTY</tt> of your plugin. This should take the format ```\br_property type name description```. In certain cases, for enumerations for example, it is beneficial to add a bulleted list to the description. This is done using a comma seperated **[]** list. ```[item1,item2,item3]``` will appear like <ul><li>item1</li><li>item2</li><li>item3</li></ul> Each property of the plugin must have a corresponding \br_property tag.

At a minimum, a comment should look like this:

        /*!
         * \ingroup abstraction group
         * \brief A description of the plugin
         * \author Your Name \cite Your Citation
         * \br_property percentage float The percentage of something
         * \br_property enum choice A choice with possible values: [choice1, choice2, choice3]
         */

There are also a few *optional* tags to provide more information:

Tag | Description
--- | ---
\br_link | A link to a webpage. It can take an optional preceding argument of a title for the link.
\br_paper | An academic paper your plugin needs to cite. This is a multi-line tag- The first line should contain the paper authors, the second line should contain the paper title, and the third should contain other information about the paper (for example conference and year). See below for an example.
\br_related_plugin | Link to a related plugin within OpenBR. The full name of the plugin should be provided. Multiple plugins can be given and they should be seperated by a space.
\br_format | A specifically formatted section that should be rendered as is. Everything following this tag (up to the next tag) is wrapped in an html <code> block and is displayed exactly as it appears.

Optional tags could look like this

        /*!
         * \br_link http://www.openbiometrics.org
         * \br_link OpenBR http://www.openbiometrics.org
         * \br_paper Author1, Author2, Author3
         *           Paper Title
         *           Conference. Year
         * \br_related_plugin ExampleTransform ExampleDistance ExampleGallery
         * \br_format
         * I will          show        up exactly
         * like
         * this
         */

Tables are also supported within any of the tags defined above. Tables are created using the standard markdown syntax. For example, to add a table to a \brief use code like the following-

        /*!
         * \brief A short brief describing the plugin.
         *
         *     Table Header | Table Header
         *     --- | ---
         *     table content | table content
         *     table content | table content
         */

Finally, OpenBR supports automatic linking for abstractions found in comments. For example, Transform will automatically become [Transform](api_docs/cpp_api/transform/transform.md).

---

## Contributing to the API

You should contribute to the API if you want to add a new abstraction or extend an existing abstraction with new functionality. Please note, this occurs *very* rarely. Our goal is to leave the core API as stable and consistent as possible and change only the surrounding plugins. If you believe your idea offers exciting new functionality or greatly increases efficiency please [open an issue](https://github.com/biometrics/openbr/issues) so that it can be discussed as a community.

---

## Style Guide

The most important rule is that **new code should be consistent with the existing code around it**. The rules below illustrate the preferred style when cleaning up existing inconsistently-styled code.

These rules are a work in progress and are subject to additions. Changes to the style can be made with a pull request implementing the change across the entire repository.

### Structs & Classes
    struct FooBar
    {

    };

### Functions
    int *fooBar(const int &x, int *y, int z)
    {
        *y = x + z;
        return y;
    }

### Variables
    int x = 2;
    int *y = &x;
    int &z = x;

### Loops and Conditionals
#### Single-statement
    for (int i=start; i<end; i++)
        foo();

#### Multiple-statement
    for (int i=start; i<end; i++) {
        foo();
        bar();
    }

### Const
Use `const` whenever possible.

### Static
Use `static` function declarations whenever possible but `static` variables sparingly.

### Unused variables
    int foo(int used, int)
    {
        // Unused variables are nameless in the function definition
        return used;
    }

### Indentation
4 spaces, no tabs.
