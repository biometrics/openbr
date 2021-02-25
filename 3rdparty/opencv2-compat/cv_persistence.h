#ifndef _CV_PERSISTENCE_H_
#define _CV_PERSISTENCE_H_

#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/persistence.hpp>

#include <queue>

//=====================================================================================
// Data structures for persistence
//=====================================================================================

/** "black box" file storage */
typedef struct CvFileStorage CvFileStorage;

/** Storage flags: */
#define CV_STORAGE_READ          0
#define CV_STORAGE_WRITE         1
#define CV_STORAGE_WRITE_TEXT    CV_STORAGE_WRITE
#define CV_STORAGE_WRITE_BINARY  CV_STORAGE_WRITE
#define CV_STORAGE_APPEND        2
#define CV_STORAGE_MEMORY        4
#define CV_STORAGE_FORMAT_MASK   (7<<3)
#define CV_STORAGE_FORMAT_AUTO   0
#define CV_STORAGE_FORMAT_XML    8
#define CV_STORAGE_FORMAT_YAML  16
#define CV_STORAGE_FORMAT_JSON  24
#define CV_STORAGE_BASE64       64
#define CV_STORAGE_WRITE_BASE64  (CV_STORAGE_BASE64 | CV_STORAGE_WRITE)

/** @brief List of attributes. :

In the current implementation, attributes are used to pass extra parameters when writing user
objects (see cvWrite). XML attributes inside tags are not supported, aside from the object type
specification (type_id attribute).
@see cvAttrList, cvAttrValue
 */
typedef struct CvAttrList
{
    const char** attr;         /**< NULL-terminated array of (attribute_name,attribute_value) pairs. */
    struct CvAttrList* next;   /**< Pointer to next chunk of the attributes list.                    */
}
CvAttrList;

/** initializes CvAttrList structure */
CV_INLINE CvAttrList cvAttrList( const char** attr CV_DEFAULT(NULL),
                                 CvAttrList* next CV_DEFAULT(NULL) )
{
    CvAttrList l;
    l.attr = attr;
    l.next = next;

    return l;
}

struct CvTypeInfo;

#define CV_NODE_NONE        0
#define CV_NODE_INT         1
#define CV_NODE_INTEGER     CV_NODE_INT
#define CV_NODE_REAL        2
#define CV_NODE_FLOAT       CV_NODE_REAL
#define CV_NODE_STR         3
#define CV_NODE_STRING      CV_NODE_STR
#define CV_NODE_REF         4 /**< not used */
#define CV_NODE_SEQ         5
#define CV_NODE_MAP         6
#define CV_NODE_TYPE_MASK   7

#define CV_NODE_TYPE(flags)  ((flags) & CV_NODE_TYPE_MASK)

/** file node flags */
#define CV_NODE_FLOW        8 /**<Used only for writing structures in YAML format. */
#define CV_NODE_USER        16
#define CV_NODE_EMPTY       32
#define CV_NODE_NAMED       64

#define CV_NODE_IS_INT(flags)        (CV_NODE_TYPE(flags) == CV_NODE_INT)
#define CV_NODE_IS_REAL(flags)       (CV_NODE_TYPE(flags) == CV_NODE_REAL)
#define CV_NODE_IS_STRING(flags)     (CV_NODE_TYPE(flags) == CV_NODE_STRING)
#define CV_NODE_IS_SEQ(flags)        (CV_NODE_TYPE(flags) == CV_NODE_SEQ)
#define CV_NODE_IS_MAP(flags)        (CV_NODE_TYPE(flags) == CV_NODE_MAP)
#define CV_NODE_IS_COLLECTION(flags) (CV_NODE_TYPE(flags) >= CV_NODE_SEQ)
#define CV_NODE_IS_FLOW(flags)       (((flags) & CV_NODE_FLOW) != 0)
#define CV_NODE_IS_EMPTY(flags)      (((flags) & CV_NODE_EMPTY) != 0)
#define CV_NODE_IS_USER(flags)       (((flags) & CV_NODE_USER) != 0)
#define CV_NODE_HAS_NAME(flags)      (((flags) & CV_NODE_NAMED) != 0)

#define CV_NODE_SEQ_SIMPLE 256
#define CV_NODE_SEQ_IS_SIMPLE(seq) (((seq)->flags & CV_NODE_SEQ_SIMPLE) != 0)

typedef struct CvString
{
    int len;
    char* ptr;
}
CvString;

/** All the keys (names) of elements in the read file storage
   are stored in the hash to speed up the lookup operations: */
typedef struct CvStringHashNode
{
    unsigned hashval;
    CvString str;
    struct CvStringHashNode* next;
}
CvStringHashNode;

typedef struct CvGenericHash CvFileNodeHash;

/** Basic element of the file storage - scalar or collection: */
typedef struct CvFileNode
{
    int tag;
    struct CvTypeInfo* info; /**< type information
            (only for user-defined object, for others it is 0) */
    union
    {
        double f; /**< scalar floating-point number */
        int i;    /**< scalar integer number */
        CvString str; /**< text string */
        CvSeq* seq; /**< sequence (ordered collection of file nodes) */
        CvFileNodeHash* map; /**< map (collection of named file nodes) */
    } data;
}
CvFileNode;

#ifdef __cplusplus
extern "C" {
#endif
typedef int (CV_CDECL *CvIsInstanceFunc)( const void* struct_ptr );
typedef void (CV_CDECL *CvReleaseFunc)( void** struct_dblptr );
typedef void* (CV_CDECL *CvReadFunc)( CvFileStorage* storage, CvFileNode* node );
typedef void (CV_CDECL *CvWriteFunc)( CvFileStorage* storage, const char* name,
                                      const void* struct_ptr, CvAttrList attributes );
typedef void* (CV_CDECL *CvCloneFunc)( const void* struct_ptr );
#ifdef __cplusplus
}
#endif

/** @brief Type information

The structure contains information about one of the standard or user-defined types. Instances of the
type may or may not contain a pointer to the corresponding CvTypeInfo structure. In any case, there
is a way to find the type info structure for a given object using the cvTypeOf function.
Alternatively, type info can be found by type name using cvFindType, which is used when an object
is read from file storage. The user can register a new type with cvRegisterType that adds the type
information structure into the beginning of the type list. Thus, it is possible to create
specialized types from generic standard types and override the basic methods.
 */
typedef struct CvTypeInfo
{
    int flags; /**< not used */
    int header_size; /**< sizeof(CvTypeInfo) */
    struct CvTypeInfo* prev; /**< previous registered type in the list */
    struct CvTypeInfo* next; /**< next registered type in the list */
    const char* type_name; /**< type name, written to file storage */
    CvIsInstanceFunc is_instance; /**< checks if the passed object belongs to the type */
    CvReleaseFunc release; /**< releases object (memory etc.) */
    CvReadFunc read; /**< reads object from file storage */
    CvWriteFunc write; /**< writes object to file storage */
    CvCloneFunc clone; /**< creates a copy of the object */
}
CvTypeInfo;

//=====================================================================================
// Core file operations
//=====================================================================================

struct CV_EXPORTS CvType
{
    CvType( const char* type_name,
            CvIsInstanceFunc is_instance, CvReleaseFunc release=0,
            CvReadFunc read=0, CvWriteFunc write=0, CvCloneFunc clone=0 );
    ~CvType();
    CvTypeInfo* info;

    static CvTypeInfo* first;
    static CvTypeInfo* last;
};

static inline void* cvAlignPtr( const void* ptr, int align = 32 )
{
    CV_DbgAssert ( (align & (align-1)) == 0 );
    return (void*)( ((size_t)ptr + align - 1) & ~(size_t)(align-1) );
}

static inline int cvAlign( int size, int align )
{
    CV_DbgAssert( (align & (align-1)) == 0 && size < INT_MAX );
    return (size + align - 1) & -align;
}

CVAPI(CvFileStorage*)  cvOpenFileStorage( const char* filename, CvMemStorage* memstorage,
                                          int flags, const char* encoding CV_DEFAULT(NULL) );

CVAPI(void) cvReleaseFileStorage( CvFileStorage** fs );

CVAPI(const char*) cvAttrValue( const CvAttrList* attr, const char* attr_name );

CVAPI(void) cvStartWriteStruct( CvFileStorage* fs, const char* name,
                                int struct_flags, const char* type_name CV_DEFAULT(NULL),
                                CvAttrList attributes CV_DEFAULT(cvAttrList()));

/** @brief Finishes writing to a file node collection.
@param fs File storage
@sa cvStartWriteStruct.
 */
CVAPI(void) cvEndWriteStruct( CvFileStorage* fs );

/** @brief Writes an integer value.

The function writes a single integer value (with or without a name) to the file storage.
@param fs File storage
@param name Name of the written value. Should be NULL if and only if the parent structure is a
sequence.
@param value The written value
 */
CVAPI(void) cvWriteInt( CvFileStorage* fs, const char* name, int value );

/** @brief Writes a floating-point value.

The function writes a single floating-point value (with or without a name) to file storage. Special
values are encoded as follows: NaN (Not A Number) as .NaN, infinity as +.Inf or -.Inf.

The following example shows how to use the low-level writing functions to store custom structures,
such as termination criteria, without registering a new type. :
@code
    void write_termcriteria( CvFileStorage* fs, const char* struct_name,
                             CvTermCriteria* termcrit )
    {
        cvStartWriteStruct( fs, struct_name, CV_NODE_MAP, NULL, cvAttrList(0,0));
        cvWriteComment( fs, "termination criteria", 1 ); // just a description
        if( termcrit->type & CV_TERMCRIT_ITER )
            cvWriteInteger( fs, "max_iterations", termcrit->max_iter );
        if( termcrit->type & CV_TERMCRIT_EPS )
            cvWriteReal( fs, "accuracy", termcrit->epsilon );
        cvEndWriteStruct( fs );
    }
@endcode
@param fs File storage
@param name Name of the written value. Should be NULL if and only if the parent structure is a
sequence.
@param value The written value
*/
CVAPI(void) cvWriteReal( CvFileStorage* fs, const char* name, double value );

/** @brief Writes a text string.

The function writes a text string to file storage.
@param fs File storage
@param name Name of the written string . Should be NULL if and only if the parent structure is a
sequence.
@param str The written text string
@param quote If non-zero, the written string is put in quotes, regardless of whether they are
required. Otherwise, if the flag is zero, quotes are used only when they are required (e.g. when
the string starts with a digit or contains spaces).
 */
CVAPI(void) cvWriteString( CvFileStorage* fs, const char* name,
                           const char* str, int quote CV_DEFAULT(0) );

/** @brief Writes a comment.

The function writes a comment into file storage. The comments are skipped when the storage is read.
@param fs File storage
@param comment The written comment, single-line or multi-line
@param eol_comment If non-zero, the function tries to put the comment at the end of current line.
If the flag is zero, if the comment is multi-line, or if it does not fit at the end of the current
line, the comment starts a new line.
 */
CVAPI(void) cvWriteComment( CvFileStorage* fs, const char* comment,
                            int eol_comment );

/** @brief Writes an object to file storage.

The function writes an object to file storage. First, the appropriate type info is found using
cvTypeOf. Then, the write method associated with the type info is called.

Attributes are used to customize the writing procedure. The standard types support the following
attributes (all the dt attributes have the same format as in cvWriteRawData):

-# CvSeq
    -   **header_dt** description of user fields of the sequence header that follow CvSeq, or
        CvChain (if the sequence is a Freeman chain) or CvContour (if the sequence is a contour or
        point sequence)
    -   **dt** description of the sequence elements.
    -   **recursive** if the attribute is present and is not equal to "0" or "false", the whole
        tree of sequences (contours) is stored.
-# CvGraph
    -   **header_dt** description of user fields of the graph header that follows CvGraph;
    -   **vertex_dt** description of user fields of graph vertices
    -   **edge_dt** description of user fields of graph edges (note that the edge weight is
        always written, so there is no need to specify it explicitly)

Below is the code that creates the YAML file shown in the CvFileStorage description:
@code
    #include "cxcore.h"

    int main( int argc, char** argv )
    {
        CvMat* mat = cvCreateMat( 3, 3, CV_32F );
        CvFileStorage* fs = cvOpenFileStorage( "example.yml", 0, CV_STORAGE_WRITE );

        cvSetIdentity( mat );
        cvWrite( fs, "A", mat, cvAttrList(0,0) );

        cvReleaseFileStorage( &fs );
        cvReleaseMat( &mat );
        return 0;
    }
@endcode
@param fs File storage
@param name Name of the written object. Should be NULL if and only if the parent structure is a
sequence.
@param ptr Pointer to the object
@param attributes The attributes of the object. They are specific for each particular type (see
the discussion below).
 */
CVAPI(void) cvWrite( CvFileStorage* fs, const char* name, const void* ptr,
                         CvAttrList attributes CV_DEFAULT(cvAttrList()));

/** @brief Starts the next stream.

The function finishes the currently written stream and starts the next stream. In the case of XML
the file with multiple streams looks like this:
@code{.xml}
    <opencv_storage>
    <!-- stream #1 data -->
    </opencv_storage>
    <opencv_storage>
    <!-- stream #2 data -->
    </opencv_storage>
    ...
@endcode
The YAML file will look like this:
@code{.yaml}
    %YAML 1.0
    # stream #1 data
    ...
    ---
    # stream #2 data
@endcode
This is useful for concatenating files or for resuming the writing process.
@param fs File storage
 */
CVAPI(void) cvStartNextStream( CvFileStorage* fs );

/** @brief Writes multiple numbers.

The function writes an array, whose elements consist of single or multiple numbers. The function
call can be replaced with a loop containing a few cvWriteInt and cvWriteReal calls, but a single
call is more efficient. Note that because none of the elements have a name, they should be written
to a sequence rather than a map.
@param fs File storage
@param src Pointer to the written array
@param len Number of the array elements to write
@param dt Specification of each array element, see @ref format_spec "format specification"
 */
CVAPI(void) cvWriteRawData( CvFileStorage* fs, const void* src,
                                int len, const char* dt );

/** @brief Writes multiple numbers in Base64.

If either CV_STORAGE_WRITE_BASE64 or cv::FileStorage::WRITE_BASE64 is used,
this function will be the same as cvWriteRawData. If neither, the main
difference is that it outputs a sequence in Base64 encoding rather than
in plain text.

This function can only be used to write a sequence with a type "binary".

@param fs File storage
@param src Pointer to the written array
@param len Number of the array elements to write
@param dt Specification of each array element, see @ref format_spec "format specification"
*/
CVAPI(void) cvWriteRawDataBase64( CvFileStorage* fs, const void* src,
                                 int len, const char* dt );

/** @brief Returns a unique pointer for a given name.

The function returns a unique pointer for each particular file node name. This pointer can be then
passed to the cvGetFileNode function that is faster than cvGetFileNodeByName because it compares
text strings by comparing pointers rather than the strings' content.

Consider the following example where an array of points is encoded as a sequence of 2-entry maps:
@code
    points:
      - { x: 10, y: 10 }
      - { x: 20, y: 20 }
      - { x: 30, y: 30 }
      # ...
@endcode
Then, it is possible to get hashed "x" and "y" pointers to speed up decoding of the points. :
@code
    #include "cxcore.h"

    int main( int argc, char** argv )
    {
        CvFileStorage* fs = cvOpenFileStorage( "points.yml", 0, CV_STORAGE_READ );
        CvStringHashNode* x_key = cvGetHashedNode( fs, "x", -1, 1 );
        CvStringHashNode* y_key = cvGetHashedNode( fs, "y", -1, 1 );
        CvFileNode* points = cvGetFileNodeByName( fs, 0, "points" );

        if( CV_NODE_IS_SEQ(points->tag) )
        {
            CvSeq* seq = points->data.seq;
            int i, total = seq->total;
            CvSeqReader reader;
            cvStartReadSeq( seq, &reader, 0 );
            for( i = 0; i < total; i++ )
            {
                CvFileNode* pt = (CvFileNode*)reader.ptr;
    #if 1 // faster variant
                CvFileNode* xnode = cvGetFileNode( fs, pt, x_key, 0 );
                CvFileNode* ynode = cvGetFileNode( fs, pt, y_key, 0 );
                assert( xnode && CV_NODE_IS_INT(xnode->tag) &&
                        ynode && CV_NODE_IS_INT(ynode->tag));
                int x = xnode->data.i; // or x = cvReadInt( xnode, 0 );
                int y = ynode->data.i; // or y = cvReadInt( ynode, 0 );
    #elif 1 // slower variant; does not use x_key & y_key
                CvFileNode* xnode = cvGetFileNodeByName( fs, pt, "x" );
                CvFileNode* ynode = cvGetFileNodeByName( fs, pt, "y" );
                assert( xnode && CV_NODE_IS_INT(xnode->tag) &&
                        ynode && CV_NODE_IS_INT(ynode->tag));
                int x = xnode->data.i; // or x = cvReadInt( xnode, 0 );
                int y = ynode->data.i; // or y = cvReadInt( ynode, 0 );
    #else // the slowest yet the easiest to use variant
                int x = cvReadIntByName( fs, pt, "x", 0 );
                int y = cvReadIntByName( fs, pt, "y", 0 );
    #endif
                CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
                printf("
            }
        }
        cvReleaseFileStorage( &fs );
        return 0;
    }
@endcode
Please note that whatever method of accessing a map you are using, it is still much slower than
using plain sequences; for example, in the above example, it is more efficient to encode the points
as pairs of integers in a single numeric sequence.
@param fs File storage
@param name Literal node name
@param len Length of the name (if it is known apriori), or -1 if it needs to be calculated
@param create_missing Flag that specifies, whether an absent key should be added into the hash table
*/
CVAPI(CvStringHashNode*) cvGetHashedKey( CvFileStorage* fs, const char* name,
                                        int len CV_DEFAULT(-1),
                                        int create_missing CV_DEFAULT(0));

/** @brief Retrieves one of the top-level nodes of the file storage.

The function returns one of the top-level file nodes. The top-level nodes do not have a name, they
correspond to the streams that are stored one after another in the file storage. If the index is out
of range, the function returns a NULL pointer, so all the top-level nodes can be iterated by
subsequent calls to the function with stream_index=0,1,..., until the NULL pointer is returned.
This function can be used as a base for recursive traversal of the file storage.
@param fs File storage
@param stream_index Zero-based index of the stream. See cvStartNextStream . In most cases,
there is only one stream in the file; however, there can be several.
 */
CVAPI(CvFileNode*) cvGetRootFileNode( const CvFileStorage* fs,
                                     int stream_index CV_DEFAULT(0) );

/** @brief Finds a node in a map or file storage.

The function finds a file node. It is a faster version of cvGetFileNodeByName (see
cvGetHashedKey discussion). Also, the function can insert a new node, if it is not in the map yet.
@param fs File storage
@param map The parent map. If it is NULL, the function searches a top-level node. If both map and
key are NULLs, the function returns the root file node - a map that contains top-level nodes.
@param key Unique pointer to the node name, retrieved with cvGetHashedKey
@param create_missing Flag that specifies whether an absent node should be added to the map
 */
CVAPI(CvFileNode*) cvGetFileNode( CvFileStorage* fs, CvFileNode* map,
                                 const CvStringHashNode* key,
                                 int create_missing CV_DEFAULT(0) );

/** @brief Finds a node in a map or file storage.

The function finds a file node by name. The node is searched either in map or, if the pointer is
NULL, among the top-level file storage nodes. Using this function for maps and cvGetSeqElem (or
sequence reader) for sequences, it is possible to navigate through the file storage. To speed up
multiple queries for a certain key (e.g., in the case of an array of structures) one may use a
combination of cvGetHashedKey and cvGetFileNode.
@param fs File storage
@param map The parent map. If it is NULL, the function searches in all the top-level nodes
(streams), starting with the first one.
@param name The file node name
 */
CVAPI(CvFileNode*) cvGetFileNodeByName( const CvFileStorage* fs,
                                       const CvFileNode* map,
                                       const char* name );

/** @brief Retrieves an integer value from a file node.

The function returns an integer that is represented by the file node. If the file node is NULL, the
default_value is returned (thus, it is convenient to call the function right after cvGetFileNode
without checking for a NULL pointer). If the file node has type CV_NODE_INT, then node-\>data.i is
returned. If the file node has type CV_NODE_REAL, then node-\>data.f is converted to an integer
and returned. Otherwise the error is reported.
@param node File node
@param default_value The value that is returned if node is NULL
 */
CV_INLINE int cvReadInt( const CvFileNode* node, int default_value CV_DEFAULT(0) )
{
    return !node ? default_value :
        CV_NODE_IS_INT(node->tag) ? node->data.i :
        CV_NODE_IS_REAL(node->tag) ? cvRound(node->data.f) : 0x7fffffff;
}

/** @brief Finds a file node and returns its value.

The function is a simple superposition of cvGetFileNodeByName and cvReadInt.
@param fs File storage
@param map The parent map. If it is NULL, the function searches a top-level node.
@param name The node name
@param default_value The value that is returned if the file node is not found
 */
CV_INLINE int cvReadIntByName( const CvFileStorage* fs, const CvFileNode* map,
                         const char* name, int default_value CV_DEFAULT(0) )
{
    return cvReadInt( cvGetFileNodeByName( fs, map, name ), default_value );
}

/** @brief Retrieves a floating-point value from a file node.

The function returns a floating-point value that is represented by the file node. If the file node
is NULL, the default_value is returned (thus, it is convenient to call the function right after
cvGetFileNode without checking for a NULL pointer). If the file node has type CV_NODE_REAL ,
then node-\>data.f is returned. If the file node has type CV_NODE_INT , then node-:math:\>data.f
is converted to floating-point and returned. Otherwise the result is not determined.
@param node File node
@param default_value The value that is returned if node is NULL
 */
CV_INLINE double cvReadReal( const CvFileNode* node, double default_value CV_DEFAULT(0.) )
{
    return !node ? default_value :
        CV_NODE_IS_INT(node->tag) ? (double)node->data.i :
        CV_NODE_IS_REAL(node->tag) ? node->data.f : 1e300;
}

/** @brief Finds a file node and returns its value.

The function is a simple superposition of cvGetFileNodeByName and cvReadReal .
@param fs File storage
@param map The parent map. If it is NULL, the function searches a top-level node.
@param name The node name
@param default_value The value that is returned if the file node is not found
 */
CV_INLINE double cvReadRealByName( const CvFileStorage* fs, const CvFileNode* map,
                        const char* name, double default_value CV_DEFAULT(0.) )
{
    return cvReadReal( cvGetFileNodeByName( fs, map, name ), default_value );
}

/** @brief Retrieves a text string from a file node.

The function returns a text string that is represented by the file node. If the file node is NULL,
the default_value is returned (thus, it is convenient to call the function right after
cvGetFileNode without checking for a NULL pointer). If the file node has type CV_NODE_STR , then
node-:math:\>data.str.ptr is returned. Otherwise the result is not determined.
@param node File node
@param default_value The value that is returned if node is NULL
 */
CV_INLINE const char* cvReadString( const CvFileNode* node,
                        const char* default_value CV_DEFAULT(NULL) )
{
    return !node ? default_value : CV_NODE_IS_STRING(node->tag) ? node->data.str.ptr : 0;
}

/** @brief Finds a file node by its name and returns its value.

The function is a simple superposition of cvGetFileNodeByName and cvReadString .
@param fs File storage
@param map The parent map. If it is NULL, the function searches a top-level node.
@param name The node name
@param default_value The value that is returned if the file node is not found
 */
CV_INLINE const char* cvReadStringByName( const CvFileStorage* fs, const CvFileNode* map,
                        const char* name, const char* default_value CV_DEFAULT(NULL) )
{
    return cvReadString( cvGetFileNodeByName( fs, map, name ), default_value );
}


/** @brief Decodes an object and returns a pointer to it.

The function decodes a user object (creates an object in a native representation from the file
storage subtree) and returns it. The object to be decoded must be an instance of a registered type
that supports the read method (see CvTypeInfo). The type of the object is determined by the type
name that is encoded in the file. If the object is a dynamic structure, it is created either in
memory storage and passed to cvOpenFileStorage or, if a NULL pointer was passed, in temporary
memory storage, which is released when cvReleaseFileStorage is called. Otherwise, if the object is
not a dynamic structure, it is created in a heap and should be released with a specialized function
or by using the generic cvRelease.
@param fs File storage
@param node The root object node
@param attributes Unused parameter
 */
CVAPI(void*) cvRead( CvFileStorage* fs, CvFileNode* node,
                        CvAttrList* attributes CV_DEFAULT(NULL));

/** @brief Finds an object by name and decodes it.

The function is a simple superposition of cvGetFileNodeByName and cvRead.
@param fs File storage
@param map The parent map. If it is NULL, the function searches a top-level node.
@param name The node name
@param attributes Unused parameter
 */
CV_INLINE void* cvReadByName( CvFileStorage* fs, const CvFileNode* map,
                              const char* name, CvAttrList* attributes CV_DEFAULT(NULL) )
{
    return cvRead( fs, cvGetFileNodeByName( fs, map, name ), attributes );
}


/** @brief Initializes the file node sequence reader.

The function initializes the sequence reader to read data from a file node. The initialized reader
can be then passed to cvReadRawDataSlice.
@param fs File storage
@param src The file node (a sequence) to read numbers from
@param reader Pointer to the sequence reader
 */
CVAPI(void) cvStartReadRawData( const CvFileStorage* fs, const CvFileNode* src,
                               CvSeqReader* reader );

/** @brief Initializes file node sequence reader.

The function reads one or more elements from the file node, representing a sequence, to a
user-specified array. The total number of read sequence elements is a product of total and the
number of components in each array element. For example, if dt=2if, the function will read total\*3
sequence elements. As with any sequence, some parts of the file node sequence can be skipped or read
repeatedly by repositioning the reader using cvSetSeqReaderPos.
@param fs File storage
@param reader The sequence reader. Initialize it with cvStartReadRawData .
@param count The number of elements to read
@param dst Pointer to the destination array
@param dt Specification of each array element. It has the same format as in cvWriteRawData .
 */
CVAPI(void) cvReadRawDataSlice( const CvFileStorage* fs, CvSeqReader* reader,
                               int count, void* dst, const char* dt );

/** @brief Reads multiple numbers.

The function reads elements from a file node that represents a sequence of scalars.
@param fs File storage
@param src The file node (a sequence) to read numbers from
@param dst Pointer to the destination array
@param dt Specification of each array element. It has the same format as in cvWriteRawData .
 */
CVAPI(void) cvReadRawData( const CvFileStorage* fs, const CvFileNode* src,
                          void* dst, const char* dt );

/** @brief Writes a file node to another file storage.

The function writes a copy of a file node to file storage. Possible applications of the function are
merging several file storages into one and conversion between XML, YAML and JSON formats.
@param fs Destination file storage
@param new_node_name New name of the file node in the destination file storage. To keep the
existing name, use cvcvGetFileNodeName
@param node The written node
@param embed If the written node is a collection and this parameter is not zero, no extra level of
hierarchy is created. Instead, all the elements of node are written into the currently written
structure. Of course, map elements can only be embedded into another map, and sequence elements
can only be embedded into another sequence.
 */
CVAPI(void) cvWriteFileNode( CvFileStorage* fs, const char* new_node_name,
                            const CvFileNode* node, int embed );

/** @brief Returns the name of a file node.

The function returns the name of a file node or NULL, if the file node does not have a name or if
node is NULL.
@param node File node
 */
CVAPI(const char*) cvGetFileNodeName( const CvFileNode* node );

/*********************************** Adding own types ***********************************/

/** @brief Registers a new type.

The function registers a new type, which is described by info . The function creates a copy of the
structure, so the user should delete it after calling the function.
@param info Type info structure
 */
CVAPI(void) cvRegisterType( const CvTypeInfo* info );

/** @brief Unregisters the type.

The function unregisters a type with a specified name. If the name is unknown, it is possible to
locate the type info by an instance of the type using cvTypeOf or by iterating the type list,
starting from cvFirstType, and then calling cvUnregisterType(info-\>typeName).
@param type_name Name of an unregistered type
 */
CVAPI(void) cvUnregisterType( const char* type_name );

/** @brief Returns the beginning of a type list.

The function returns the first type in the list of registered types. Navigation through the list can
be done via the prev and next fields of the CvTypeInfo structure.
 */
CVAPI(CvTypeInfo*) cvFirstType(void);

/** @brief Finds a type by its name.

The function finds a registered type by its name. It returns NULL if there is no type with the
specified name.
@param type_name Type name
 */
CVAPI(CvTypeInfo*) cvFindType( const char* type_name );

/** @brief Returns the type of an object.

The function finds the type of a given object. It iterates through the list of registered types and
calls the is_instance function/method for every type info structure with that object until one of
them returns non-zero or until the whole list has been traversed. In the latter case, the function
returns NULL.
@param struct_ptr The object pointer
 */
CVAPI(CvTypeInfo*) cvTypeOf( const void* struct_ptr );

//=====================================================================================
// Base64
//=====================================================================================

static const size_t PARSER_BASE64_BUFFER_SIZE = 1024U * 1024U / 8U;

namespace base64 {

namespace fs {
enum State
{
    Uncertain,
    NotUse,
    InUse,
};
} // fs::

static const size_t HEADER_SIZE         = 24U;
static const size_t ENCODED_HEADER_SIZE = 32U;

size_t base64_encode(uint8_t const * src, uint8_t * dst, size_t off,      size_t cnt);
size_t base64_encode(   char const * src,    char * dst, size_t off = 0U, size_t cnt = 0U);
size_t base64_decode(uint8_t const * src, uint8_t * dst, size_t off,      size_t cnt);
size_t base64_decode(   char const * src,    char * dst, size_t off = 0U, size_t cnt = 0U);
bool   base64_valid (uint8_t const * src, size_t off,      size_t cnt);
bool   base64_valid (   char const * src, size_t off = 0U, size_t cnt = 0U);
size_t base64_encode_buffer_size(size_t cnt, bool is_end_with_zero = true);
size_t base64_decode_buffer_size(size_t cnt, bool is_end_with_zero = true);
size_t base64_decode_buffer_size(size_t cnt, char  const * src, bool is_end_with_zero = true);
size_t base64_decode_buffer_size(size_t cnt, uchar const * src, bool is_end_with_zero = true);
std::string make_base64_header(const char * dt);
bool read_base64_header(std::vector<char> const & header, std::string & dt);
void make_seq(::CvFileStorage* fs, const uchar* binary_data, size_t elem_cnt, const char * dt, CvSeq & seq);
void cvWriteRawDataBase64(::CvFileStorage* fs, const void* _data, int len, const char* dt);

class Base64ContextEmitter;

class Base64Writer
{
public:
    Base64Writer(::CvFileStorage * fs);
    ~Base64Writer();
    void write(const void* _data, size_t len, const char* dt);
    template<typename _to_binary_convertor_t> void write(_to_binary_convertor_t & convertor, const char* dt);

private:
    void check_dt(const char* dt);

private:
    // disable copy and assignment
    Base64Writer(const Base64Writer &);
    Base64Writer & operator=(const Base64Writer &);

private:

    Base64ContextEmitter * emitter;
    std::string data_type_string;
};

class Base64ContextParser
{
public:
    explicit Base64ContextParser(uchar * buffer, size_t size);
    ~Base64ContextParser();
    Base64ContextParser & read(const uchar * beg, const uchar * end);
    bool flush();
private:
    static const size_t BUFFER_LEN = 120U;
    uchar * dst_cur;
    uchar * dst_end;
    std::vector<uchar> base64_buffer;
    uchar * src_beg;
    uchar * src_cur;
    uchar * src_end;
    std::vector<uchar> binary_buffer;
};

} // base64::

//=====================================================================================

#define CV_FS_MAX_LEN 4096
#define CV_FS_MAX_FMT_PAIRS  128

#define CV_FILE_STORAGE ('Y' + ('A' << 8) + ('M' << 16) + ('L' << 24))

#define CV_IS_FILE_STORAGE(fs) ((fs) != 0 && (fs)->flags == CV_FILE_STORAGE)

#define CV_CHECK_FILE_STORAGE(fs)                       \
{                                                       \
    if( !CV_IS_FILE_STORAGE(fs) )                       \
        CV_Error( (fs) ? CV_StsBadArg : CV_StsNullPtr,  \
                  "Invalid pointer to file storage" );  \
}

#define CV_CHECK_OUTPUT_FILE_STORAGE(fs)                \
{                                                       \
    CV_CHECK_FILE_STORAGE(fs);                          \
    if( !fs->write_mode )                               \
        CV_Error( CV_StsError, "The file storage is opened for reading" ); \
}

#define CV_PARSE_ERROR( errmsg )                                    \
    icvParseError( fs, CV_Func, (errmsg), __FILE__, __LINE__ )

typedef struct CvGenericHash
{
    CV_SET_FIELDS()
    int tab_size;
    void** table;
}
CvGenericHash;
typedef CvGenericHash CvStringHash;

//typedef void (*CvParse)( struct CvFileStorage* fs );
typedef void (*CvStartWriteStruct)( struct CvFileStorage* fs, const char* key,
                                    int struct_flags, const char* type_name );
typedef void (*CvEndWriteStruct)( struct CvFileStorage* fs );
typedef void (*CvWriteInt)( struct CvFileStorage* fs, const char* key, int value );
typedef void (*CvWriteReal)( struct CvFileStorage* fs, const char* key, double value );
typedef void (*CvWriteString)( struct CvFileStorage* fs, const char* key,
                               const char* value, int quote );
typedef void (*CvWriteComment)( struct CvFileStorage* fs, const char* comment, int eol_comment );
typedef void (*CvStartNextStream)( struct CvFileStorage* fs );

typedef void* gzFile;

typedef struct CvFileStorage
{
    int flags;
    int fmt;
    int write_mode;
    int is_first;
    CvMemStorage* memstorage;
    CvMemStorage* dststorage;
    CvMemStorage* strstorage;
    CvStringHash* str_hash;
    CvSeq* roots;
    CvSeq* write_stack;
    int struct_indent;
    int struct_flags;
    CvString struct_tag;
    int space;
    char* filename;
    FILE* file;
    gzFile gzfile;
    char* buffer;
    char* buffer_start;
    char* buffer_end;
    int wrap_margin;
    int lineno;
    int dummy_eof;
    const char* errmsg;
    char errmsgbuf[128];

    CvStartWriteStruct start_write_struct;
    CvEndWriteStruct end_write_struct;
    CvWriteInt write_int;
    CvWriteReal write_real;
    CvWriteString write_string;
    CvWriteComment write_comment;
    CvStartNextStream start_next_stream;

    const char* strbuf;
    size_t strbufsize, strbufpos;
    std::deque<char>* outbuf;

    base64::Base64Writer * base64_writer;
    bool is_default_using_base64;
    base64::fs::State state_of_writing_base64;  /**< used in WriteRawData only */

    bool is_write_struct_delayed;
    char* delayed_struct_key;
    int   delayed_struct_flags;
    char* delayed_type_name;

    bool is_opened;
}
CvFileStorage;

typedef struct CvFileMapNode
{
    CvFileNode value;
    const CvStringHashNode* key;
    struct CvFileMapNode* next;
}
CvFileMapNode;

/****************************************************************************************\
*                            Common macros and type definitions                          *
\****************************************************************************************/

#define cv_isprint(c)     ((uchar)(c) >= (uchar)' ')
#define cv_isprint_or_tab(c)  ((uchar)(c) >= (uchar)' ' || (c) == '\t')

inline bool cv_isalnum(char c)
{
    return ('0' <= c && c <= '9') || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

inline bool cv_isalpha(char c)
{
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

inline bool cv_isdigit(char c)
{
    return '0' <= c && c <= '9';
}

inline bool cv_isspace(char c)
{
    return (9 <= c && c <= 13) || c == ' ';
}

inline char* cv_skip_BOM(char* ptr)
{
    if((uchar)ptr[0] == 0xef && (uchar)ptr[1] == 0xbb && (uchar)ptr[2] == 0xbf) //UTF-8 BOM
    {
      return ptr + 3;
    }
    return ptr;
}

/****************************************************************************************\
*                                       XML                                              *
\****************************************************************************************/

char* icv_itoa( int _val, char* buffer, int /*radix*/ );
double icv_strtod( CvFileStorage* fs, char* ptr, char** endptr );
char* icvFloatToString( char* buf, float value );
char* icvDoubleToString( char* buf, double value );

char icvTypeSymbol(int depth);
void icvClose( CvFileStorage* fs, cv::String* out );
void icvCloseFile( CvFileStorage* fs );
void icvPuts( CvFileStorage* fs, const char* str );
char* icvGets( CvFileStorage* fs, char* str, int maxCount );
int icvEof( CvFileStorage* fs );
void icvRewind( CvFileStorage* fs );
char* icvFSFlush( CvFileStorage* fs );
void icvFSCreateCollection( CvFileStorage* fs, int tag, CvFileNode* collection );
char* icvFSResizeWriteBuffer( CvFileStorage* fs, char* ptr, int len );
int icvCalcStructSize( const char* dt, int initial_size );
int icvCalcElemSize( const char* dt, int initial_size );
void CV_NORETURN icvParseError(const CvFileStorage* fs, const char* func_name, const char* err_msg, const char* source_file, int source_line);
char* icvEncodeFormat( int elem_type, char* dt );
int icvDecodeFormat( const char* dt, int* fmt_pairs, int max_len );
int icvDecodeSimpleFormat( const char* dt );
void icvWriteFileNode( CvFileStorage* fs, const char* name, const CvFileNode* node );
void icvWriteCollection( CvFileStorage* fs, const CvFileNode* node );
void switch_to_Base64_state( CvFileStorage* fs, base64::fs::State state );
void make_write_struct_delayed( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name );
void check_if_write_struct_is_delayed( CvFileStorage* fs, bool change_type_to_base64 = false );
CvGenericHash* cvCreateMap( int flags, int header_size, int elem_size, CvMemStorage* storage, int start_tab_size );

//
// XML
//
void icvXMLParse( CvFileStorage* fs );
void icvXMLStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name CV_DEFAULT(0));
void icvXMLEndWriteStruct( CvFileStorage* fs );
void icvXMLStartNextStream( CvFileStorage* fs );
void icvXMLWriteScalar( CvFileStorage* fs, const char* key, const char* data, int len );
void icvXMLWriteInt( CvFileStorage* fs, const char* key, int value );
void icvXMLWriteReal( CvFileStorage* fs, const char* key, double value );
void icvXMLWriteString( CvFileStorage* fs, const char* key, const char* str, int quote );
void icvXMLWriteComment( CvFileStorage* fs, const char* comment, int eol_comment );

typedef struct CvXMLStackRecord
{
    CvMemStoragePos pos;
    CvString struct_tag;
    int struct_indent;
    int struct_flags;
}
CvXMLStackRecord;

//
// YML
//
void icvYMLParse( CvFileStorage* fs );
void icvYMLWrite( CvFileStorage* fs, const char* key, const char* data );
void icvYMLStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name CV_DEFAULT(0));
void icvYMLEndWriteStruct( CvFileStorage* fs );
void icvYMLStartNextStream( CvFileStorage* fs );
void icvYMLWriteInt( CvFileStorage* fs, const char* key, int value );
void icvYMLWriteReal( CvFileStorage* fs, const char* key, double value );
void icvYMLWriteString( CvFileStorage* fs, const char* key, const char* str, int quote CV_DEFAULT(0));
void icvYMLWriteComment( CvFileStorage* fs, const char* comment, int eol_comment );

//
// JSON
//
void icvJSONParse( CvFileStorage* fs );
void icvJSONWrite( CvFileStorage* fs, const char* key, const char* data );
void icvJSONStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name CV_DEFAULT(0));
void icvJSONEndWriteStruct( CvFileStorage* fs );
void icvJSONStartNextStream( CvFileStorage* fs );
void icvJSONWriteInt( CvFileStorage* fs, const char* key, int value );
void icvJSONWriteReal( CvFileStorage* fs, const char* key, double value );
void icvJSONWriteString( CvFileStorage* fs, const char* key, const char* str, int quote CV_DEFAULT(0));
void icvJSONWriteComment( CvFileStorage* fs, const char* comment, int eol_comment );

// Adding icvGets is not enough - we need to merge buffer contents (see #11061)
#define CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG() \
    CV_Assert((ptr[0] != 0 || ptr != fs->buffer_end - 1) && "OpenCV persistence doesn't support very long lines")

CVAPI(CvString) cvMemStorageAllocString( CvMemStorage* storage, const char* ptr, int len );

#endif // _CV_PERSISTENCE_H_
