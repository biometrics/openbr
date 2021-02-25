// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "cv_persistence.h"
#include <opencv2/core/core_c.h>

char* icv_itoa( int _val, char* buffer, int /*radix*/ )
{
    const int radix = 10;
    char* ptr=buffer + 23 /* enough even for 64-bit integers */;
    unsigned val = abs(_val);

    *ptr = '\0';
    do
    {
        unsigned r = val / radix;
        *--ptr = (char)(val - (r*radix) + '0');
        val = r;
    }
    while( val != 0 );

    if( _val < 0 )
        *--ptr = '-';

    return ptr;
}

void icvPuts( CvFileStorage* fs, const char* str )
{
    if( fs->outbuf )
        std::copy(str, str + strlen(str), std::back_inserter(*fs->outbuf));
    else if( fs->file )
        fputs( str, fs->file );
#if USE_ZLIB
    else if( fs->gzfile )
        gzputs( fs->gzfile, str );
#endif
    else
        CV_Error( CV_StsError, "The storage is not opened" );
}

char* icvGets( CvFileStorage* fs, char* str, int maxCount )
{
    if( fs->strbuf )
    {
        size_t i = fs->strbufpos, len = fs->strbufsize;
        int j = 0;
        const char* instr = fs->strbuf;
        while( i < len && j < maxCount-1 )
        {
            char c = instr[i++];
            if( c == '\0' )
                break;
            str[j++] = c;
            if( c == '\n' )
                break;
        }
        str[j++] = '\0';
        fs->strbufpos = i;
        if (maxCount > 256 && !(fs->flags & cv::FileStorage::BASE64))
            CV_Assert(j < maxCount - 1 && "OpenCV persistence doesn't support very long lines");
        return j > 1 ? str : 0;
    }
    if( fs->file )
    {
        char* ptr = fgets( str, maxCount, fs->file );
        if (ptr && maxCount > 256 && !(fs->flags & cv::FileStorage::BASE64))
        {
            size_t sz = strnlen(ptr, maxCount);
            CV_Assert(sz < (size_t)(maxCount - 1) && "OpenCV persistence doesn't support very long lines");
        }
        return ptr;
    }
#if USE_ZLIB
    if( fs->gzfile )
    {
        char* ptr = gzgets( fs->gzfile, str, maxCount );
        if (ptr && maxCount > 256 && !(fs->flags & cv::FileStorage::BASE64))
        {
            size_t sz = strnlen(ptr, maxCount);
            CV_Assert(sz < (size_t)(maxCount - 1) && "OpenCV persistence doesn't support very long lines");
        }
        return ptr;
    }
#endif
    CV_Error(CV_StsError, "The storage is not opened");
}

int icvEof( CvFileStorage* fs )
{
    if( fs->strbuf )
        return fs->strbufpos >= fs->strbufsize;
    if( fs->file )
        return feof(fs->file);
#if USE_ZLIB
    if( fs->gzfile )
        return gzeof(fs->gzfile);
#endif
    return false;
}

void icvCloseFile( CvFileStorage* fs )
{
    if( fs->file )
        fclose( fs->file );
#if USE_ZLIB
    else if( fs->gzfile )
        gzclose( fs->gzfile );
#endif
    fs->file = 0;
    fs->gzfile = 0;
    fs->strbuf = 0;
    fs->strbufpos = 0;
    fs->is_opened = false;
}

void icvRewind( CvFileStorage* fs )
{
    if( fs->file )
        rewind(fs->file);
#if USE_ZLIB
    else if( fs->gzfile )
        gzrewind(fs->gzfile);
#endif
    fs->strbufpos = 0;
}

CvGenericHash* cvCreateMap( int flags, int header_size, int elem_size, CvMemStorage* storage, int start_tab_size )
{
    if( header_size < (int)sizeof(CvGenericHash) )
        CV_Error( CV_StsBadSize, "Too small map header_size" );

    if( start_tab_size <= 0 )
        start_tab_size = 16;

    CvGenericHash* map = (CvGenericHash*)cvCreateSet( flags, header_size, elem_size, storage );

    map->tab_size = start_tab_size;
    start_tab_size *= sizeof(map->table[0]);
    map->table = (void**)cvMemStorageAlloc( storage, start_tab_size );
    memset( map->table, 0, start_tab_size );

    return map;
}

void icvParseError(const CvFileStorage* fs, const char* func_name,
               const char* err_msg, const char* source_file, int source_line )
{
    cv::String msg = cv::format("%s(%d): %s", fs->filename, fs->lineno, err_msg);
    //cv::errorNoReturn(cv::Error::StsParseError, func_name, msg.c_str(), source_file, source_line );
    cv::error(cv::Error::StsParseError, func_name, msg.c_str(), source_file, source_line );
}

void icvFSCreateCollection( CvFileStorage* fs, int tag, CvFileNode* collection )
{
    if( CV_NODE_IS_MAP(tag) )
    {
        if( collection->tag != CV_NODE_NONE )
        {
            assert( fs->fmt == CV_STORAGE_FORMAT_XML );
            CV_PARSE_ERROR( "Sequence element should not have name (use <_></_>)" );
        }

        collection->data.map = cvCreateMap( 0, sizeof(CvFileNodeHash),
                            sizeof(CvFileMapNode), fs->memstorage, 16 );
    }
    else
    {
        CvSeq* seq;
        seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvFileNode), fs->memstorage );

        // if <collection> contains some scalar element, add it to the newly created collection
        if( CV_NODE_TYPE(collection->tag) != CV_NODE_NONE )
            cvSeqPush( seq, collection );

        collection->data.seq = seq;
    }

    collection->tag = tag;
    cvSetSeqBlockSize( collection->data.seq, 8 );
}

static char* icvFSDoResize( CvFileStorage* fs, char* ptr, int len )
{
    char* new_ptr = 0;
    int written_len = (int)(ptr - fs->buffer_start);
    int new_size = (int)((fs->buffer_end - fs->buffer_start)*3/2);
    new_size = MAX( written_len + len, new_size );
    new_ptr = (char*)cvAlloc( new_size + 256 );
    fs->buffer = new_ptr + (fs->buffer - fs->buffer_start);
    if( written_len > 0 )
        memcpy( new_ptr, fs->buffer_start, written_len );
    fs->buffer_start = new_ptr;
    fs->buffer_end = fs->buffer_start + new_size;
    new_ptr += written_len;
    return new_ptr;
}

char* icvFSResizeWriteBuffer( CvFileStorage* fs, char* ptr, int len )
{
    return ptr + len < fs->buffer_end ? ptr : icvFSDoResize( fs, ptr, len );
}

char* icvFSFlush( CvFileStorage* fs )
{
    char* ptr = fs->buffer;
    int indent;

    if( ptr > fs->buffer_start + fs->space )
    {
        ptr[0] = '\n';
        ptr[1] = '\0';
        icvPuts( fs, fs->buffer_start );
        fs->buffer = fs->buffer_start;
    }

    indent = fs->struct_indent;

    if( fs->space != indent )
    {
        memset( fs->buffer_start, ' ', indent );
        fs->space = indent;
    }

    ptr = fs->buffer = fs->buffer_start + fs->space;

    return ptr;
}

void icvClose( CvFileStorage* fs, cv::String* out )
{
    if( out )
        out->clear();

    if( !fs )
        CV_Error( CV_StsNullPtr, "NULL double pointer to file storage" );

    if( fs->is_opened )
    {
        if( fs->write_mode && (fs->file || fs->gzfile || fs->outbuf) )
        {
            if( fs->write_stack )
            {
                while( fs->write_stack->total > 0 )
                    cvEndWriteStruct(fs);
            }
            icvFSFlush(fs);
            if( fs->fmt == CV_STORAGE_FORMAT_XML )
                icvPuts( fs, "</opencv_storage>\n" );
            else if ( fs->fmt == CV_STORAGE_FORMAT_JSON )
                icvPuts( fs, "}\n" );
        }
    }

    icvCloseFile(fs);

    if( fs->outbuf && out )
    {
        *out = cv::String(fs->outbuf->begin(), fs->outbuf->end());
    }
}

char* icvDoubleToString( char* buf, double value )
{
    Cv64suf val;
    unsigned ieee754_hi;

    val.f = value;
    ieee754_hi = (unsigned)(val.u >> 32);

    if( (ieee754_hi & 0x7ff00000) != 0x7ff00000 )
    {
        int ivalue = cvRound(value);
        if( ivalue == value )
            sprintf( buf, "%d.", ivalue );
        else
        {
            static const char* fmt = "%.16e";
            char* ptr = buf;
            sprintf( buf, fmt, value );
            if( *ptr == '+' || *ptr == '-' )
                ptr++;
            for( ; cv_isdigit(*ptr); ptr++ )
                ;
            if( *ptr == ',' )
                *ptr = '.';
        }
    }
    else
    {
        unsigned ieee754_lo = (unsigned)val.u;
        if( (ieee754_hi & 0x7fffffff) + (ieee754_lo != 0) > 0x7ff00000 )
            strcpy( buf, ".Nan" );
        else
            strcpy( buf, (int)ieee754_hi < 0 ? "-.Inf" : ".Inf" );
    }

    return buf;
}

char* icvFloatToString( char* buf, float value )
{
    Cv32suf val;
    unsigned ieee754;
    val.f = value;
    ieee754 = val.u;

    if( (ieee754 & 0x7f800000) != 0x7f800000 )
    {
        int ivalue = cvRound(value);
        if( ivalue == value )
            sprintf( buf, "%d.", ivalue );
        else
        {
            static const char* fmt = "%.8e";
            char* ptr = buf;
            sprintf( buf, fmt, value );
            if( *ptr == '+' || *ptr == '-' )
                ptr++;
            for( ; cv_isdigit(*ptr); ptr++ )
                ;
            if( *ptr == ',' )
                *ptr = '.';
        }
    }
    else
    {
        if( (ieee754 & 0x7fffffff) != 0x7f800000 )
            strcpy( buf, ".Nan" );
        else
            strcpy( buf, (int)ieee754 < 0 ? "-.Inf" : ".Inf" );
    }

    return buf;
}

static void icvProcessSpecialDouble( CvFileStorage* fs, char* buf, double* value, char** endptr )
{
    char c = buf[0];
    int inf_hi = 0x7ff00000;

    if( c == '-' || c == '+' )
    {
        inf_hi = c == '-' ? 0xfff00000 : 0x7ff00000;
        c = *++buf;
    }

    if( c != '.' )
        CV_PARSE_ERROR( "Bad format of floating-point constant" );

    union{double d; uint64 i;} v;
    v.d = 0.;
    if( toupper(buf[1]) == 'I' && toupper(buf[2]) == 'N' && toupper(buf[3]) == 'F' )
        v.i = (uint64)inf_hi << 32;
    else if( toupper(buf[1]) == 'N' && toupper(buf[2]) == 'A' && toupper(buf[3]) == 'N' )
        v.i = (uint64)-1;
    else
        CV_PARSE_ERROR( "Bad format of floating-point constant" );
    *value = v.d;

    *endptr = buf + 4;
}


double icv_strtod( CvFileStorage* fs, char* ptr, char** endptr )
{
    double fval = strtod( ptr, endptr );
    if( **endptr == '.' )
    {
        char* dot_pos = *endptr;
        *dot_pos = ',';
        double fval2 = strtod( ptr, endptr );
        *dot_pos = '.';
        if( *endptr > dot_pos )
            fval = fval2;
        else
            *endptr = dot_pos;
    }

    if( *endptr == ptr || cv_isalpha(**endptr) )
        icvProcessSpecialDouble( fs, ptr, &fval, endptr );

    return fval;
}

void switch_to_Base64_state( CvFileStorage* fs, base64::fs::State state )
{
    const char * err_unkonwn_state = "Unexpected error, unable to determine the Base64 state.";
    const char * err_unable_to_switch = "Unexpected error, unable to switch to this state.";

    /* like a finite state machine */
    switch (fs->state_of_writing_base64)
    {
    case base64::fs::Uncertain:
        switch (state)
        {
        case base64::fs::InUse:
            CV_DbgAssert( fs->base64_writer == 0 );
            fs->base64_writer = new base64::Base64Writer( fs );
            break;
        case base64::fs::Uncertain:
            break;
        case base64::fs::NotUse:
            break;
        default:
            CV_Error( CV_StsError, err_unkonwn_state );
            break;
        }
        break;
    case base64::fs::InUse:
        switch (state)
        {
        case base64::fs::InUse:
        case base64::fs::NotUse:
            CV_Error( CV_StsError, err_unable_to_switch );
            break;
        case base64::fs::Uncertain:
            delete fs->base64_writer;
            fs->base64_writer = 0;
            break;
        default:
            CV_Error( CV_StsError, err_unkonwn_state );
            break;
        }
        break;
    case base64::fs::NotUse:
        switch (state)
        {
        case base64::fs::InUse:
        case base64::fs::NotUse:
            CV_Error( CV_StsError, err_unable_to_switch );
            break;
        case base64::fs::Uncertain:
            break;
        default:
            CV_Error( CV_StsError, err_unkonwn_state );
            break;
        }
        break;
    default:
        CV_Error( CV_StsError, err_unkonwn_state );
        break;
    }

    fs->state_of_writing_base64 = state;
}

void check_if_write_struct_is_delayed( CvFileStorage* fs, bool change_type_to_base64 )
{
    if ( fs->is_write_struct_delayed )
    {
        /* save data to prevent recursive call errors */
        std::string struct_key;
        std::string type_name;
        int struct_flags = fs->delayed_struct_flags;

        if ( fs->delayed_struct_key != 0 && *fs->delayed_struct_key != '\0' )
        {
            struct_key.assign(fs->delayed_struct_key);
        }
        if ( fs->delayed_type_name != 0 && *fs->delayed_type_name != '\0' )
        {
            type_name.assign(fs->delayed_type_name);
        }

        /* reset */
        delete[] fs->delayed_struct_key;
        delete[] fs->delayed_type_name;
        fs->delayed_struct_key   = 0;
        fs->delayed_struct_flags = 0;
        fs->delayed_type_name    = 0;

        fs->is_write_struct_delayed = false;

        /* call */
        if ( change_type_to_base64 )
        {
            fs->start_write_struct( fs, struct_key.c_str(), struct_flags, "binary");
            if ( fs->state_of_writing_base64 != base64::fs::Uncertain )
                switch_to_Base64_state( fs, base64::fs::Uncertain );
            switch_to_Base64_state( fs, base64::fs::InUse );
        }
        else
        {
            fs->start_write_struct( fs, struct_key.c_str(), struct_flags, type_name.c_str());
            if ( fs->state_of_writing_base64 != base64::fs::Uncertain )
                switch_to_Base64_state( fs, base64::fs::Uncertain );
            switch_to_Base64_state( fs, base64::fs::NotUse );
        }
    }
}

void make_write_struct_delayed( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name )
{
    CV_Assert( fs->is_write_struct_delayed == false );
    CV_DbgAssert( fs->delayed_struct_key   == 0 );
    CV_DbgAssert( fs->delayed_struct_flags == 0 );
    CV_DbgAssert( fs->delayed_type_name    == 0 );

    fs->delayed_struct_flags = struct_flags;

    if ( key != 0 )
    {
        fs->delayed_struct_key = new char[strlen(key) + 1U];
        strcpy(fs->delayed_struct_key, key);
    }

    if ( type_name != 0 )
    {
        fs->delayed_type_name = new char[strlen(type_name) + 1U];
        strcpy(fs->delayed_type_name, type_name);
    }

    fs->is_write_struct_delayed = true;
}

static const char symbols[9] = "ucwsifdr";

char icvTypeSymbol(int depth)
{
    CV_Assert(depth >=0 && depth < 9);
    return symbols[depth];
}

static int icvSymbolToType(char c)
{
    const char* pos = strchr( symbols, c );
    if( !pos )
        CV_Error( CV_StsBadArg, "Invalid data type specification" );
    return static_cast<int>(pos - symbols);
}

char* icvEncodeFormat( int elem_type, char* dt )
{
    sprintf( dt, "%d%c", CV_MAT_CN(elem_type), icvTypeSymbol(CV_MAT_DEPTH(elem_type)) );
    return dt + ( dt[2] == '\0' && dt[0] == '1' );
}

int icvDecodeFormat( const char* dt, int* fmt_pairs, int max_len )
{
    int fmt_pair_count = 0;
    int i = 0, k = 0, len = dt ? (int)strlen(dt) : 0;

    if( !dt || !len )
        return 0;

    assert( fmt_pairs != 0 && max_len > 0 );
    fmt_pairs[0] = 0;
    max_len *= 2;

    for( ; k < len; k++ )
    {
        char c = dt[k];

        if( cv_isdigit(c) )
        {
            int count = c - '0';
            if( cv_isdigit(dt[k+1]) )
            {
                char* endptr = 0;
                count = (int)strtol( dt+k, &endptr, 10 );
                k = (int)(endptr - dt) - 1;
            }

            if( count <= 0 )
                CV_Error( CV_StsBadArg, "Invalid data type specification" );

            fmt_pairs[i] = count;
        }
        else
        {
            int depth = icvSymbolToType(c);
            if( fmt_pairs[i] == 0 )
                fmt_pairs[i] = 1;
            fmt_pairs[i+1] = depth;
            if( i > 0 && fmt_pairs[i+1] == fmt_pairs[i-1] )
                fmt_pairs[i-2] += fmt_pairs[i];
            else
            {
                i += 2;
                if( i >= max_len )
                    CV_Error( CV_StsBadArg, "Too long data type specification" );
            }
            fmt_pairs[i] = 0;
        }
    }

    fmt_pair_count = i/2;
    return fmt_pair_count;
}


int icvCalcElemSize( const char* dt, int initial_size )
{
    int size = 0;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], i, fmt_pair_count;
    int comp_size;

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    fmt_pair_count *= 2;
    for( i = 0, size = initial_size; i < fmt_pair_count; i += 2 )
    {
        comp_size = CV_ELEM_SIZE(fmt_pairs[i+1]);
        size = cvAlign( size, comp_size );
        size += comp_size * fmt_pairs[i];
    }
    if( initial_size == 0 )
    {
        comp_size = CV_ELEM_SIZE(fmt_pairs[1]);
        size = cvAlign( size, comp_size );
    }
    return size;
}


int icvCalcStructSize( const char* dt, int initial_size )
{
    int size = icvCalcElemSize( dt, initial_size );
    size_t elem_max_size = 0;
    for ( const char * type = dt; *type != '\0'; type++ ) {
        switch ( *type )
        {
        case 'u': { elem_max_size = std::max( elem_max_size, sizeof(uchar ) ); break; }
        case 'c': { elem_max_size = std::max( elem_max_size, sizeof(schar ) ); break; }
        case 'w': { elem_max_size = std::max( elem_max_size, sizeof(ushort) ); break; }
        case 's': { elem_max_size = std::max( elem_max_size, sizeof(short ) ); break; }
        case 'i': { elem_max_size = std::max( elem_max_size, sizeof(int   ) ); break; }
        case 'f': { elem_max_size = std::max( elem_max_size, sizeof(float ) ); break; }
        case 'd': { elem_max_size = std::max( elem_max_size, sizeof(double) ); break; }
        default: break;
        }
    }
    size = cvAlign( size, static_cast<int>(elem_max_size) );
    return size;
}

int icvDecodeSimpleFormat( const char* dt )
{
    int elem_type = -1;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], fmt_pair_count;

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    if( fmt_pair_count != 1 || fmt_pairs[0] >= CV_CN_MAX)
        CV_Error( CV_StsError, "Too complex format for the matrix" );

    elem_type = CV_MAKETYPE( fmt_pairs[1], fmt_pairs[0] );

    return elem_type;
}

void icvWriteCollection( CvFileStorage* fs, const CvFileNode* node )
{
    int i, total = node->data.seq->total;
    int elem_size = node->data.seq->elem_size;
    int is_map = CV_NODE_IS_MAP(node->tag);
    CvSeqReader reader;

    cvStartReadSeq( node->data.seq, &reader, 0 );

    for( i = 0; i < total; i++ )
    {
        CvFileMapNode* elem = (CvFileMapNode*)reader.ptr;
        if( !is_map || CV_IS_SET_ELEM(elem) )
        {
            const char* name = is_map ? elem->key->str.ptr : 0;
            icvWriteFileNode( fs, name, &elem->value );
        }
        CV_NEXT_SEQ_ELEM( elem_size, reader );
    }
}

void icvWriteFileNode( CvFileStorage* fs, const char* name, const CvFileNode* node )
{
    switch( CV_NODE_TYPE(node->tag) )
    {
    case CV_NODE_INT:
        fs->write_int( fs, name, node->data.i );
        break;
    case CV_NODE_REAL:
        fs->write_real( fs, name, node->data.f );
        break;
    case CV_NODE_STR:
        fs->write_string( fs, name, node->data.str.ptr, 0 );
        break;
    case CV_NODE_SEQ:
    case CV_NODE_MAP:
        cvStartWriteStruct( fs, name, CV_NODE_TYPE(node->tag) +
                (CV_NODE_SEQ_IS_SIMPLE(node->data.seq) ? CV_NODE_FLOW : 0),
                node->info ? node->info->type_name : 0 );
        icvWriteCollection( fs, node );
        cvEndWriteStruct( fs );
        break;
    case CV_NODE_NONE:
        cvStartWriteStruct( fs, name, CV_NODE_SEQ, 0 );
        cvEndWriteStruct( fs );
        break;
    default:
        CV_Error( CV_StsBadFlag, "Unknown type of file node" );
    }
}


static inline bool cv_strcasecmp(const char * s1, const char * s2)
{
    if ( s1 == 0 && s2 == 0 )
        return true;
    else if ( s1 == 0 || s2 == 0 )
        return false;

    size_t len1 = strlen(s1);
    size_t len2 = strlen(s2);
    if ( len1 != len2 )
        return false;

    for ( size_t i = 0U; i < len1; i++ )
        if ( tolower( static_cast<int>(s1[i]) ) != tolower( static_cast<int>(s2[i]) ) )
            return false;

    return true;
}

// this function will convert "aa?bb&cc&dd" to {"aa", "bb", "cc", "dd"}
static std::vector<std::string> analyze_file_name( std::string const & file_name )
{
    static const char not_file_name       = '\n';
    static const char parameter_begin     = '?';
    static const char parameter_separator = '&';
    std::vector<std::string> result;

    if ( file_name.find(not_file_name, 0U) != std::string::npos )
        return result;

    size_t beg = file_name.find_last_of(parameter_begin);
    size_t end = file_name.size();
    result.push_back(file_name.substr(0U, beg));

    if ( beg != std::string::npos )
    {
        beg ++;
        for ( size_t param_beg = beg, param_end = beg;
              param_end < end;
              param_beg = param_end + 1U )
        {
            param_end = file_name.find_first_of( parameter_separator, param_beg );
            if ( (param_end == std::string::npos || param_end != param_beg) && param_beg + 1U < end )
            {
                result.push_back( file_name.substr( param_beg, param_end - param_beg ) );
            }
        }
    }

    return result;
}

static bool is_param_exist( const std::vector<std::string> & params, const std::string & param )
{
    if ( params.size() < 2U )
        return false;

    return std::find(params.begin(), params.end(), param) != params.end();
}

//===========================================================================================

static
void cvOpenFileStorage_(CvFileStorage*& fs, const char* query, CvMemStorage* dststorage, int flags, const char* encoding)
{
    int default_block_size = 1 << 18;
    bool append = (flags & 3) == CV_STORAGE_APPEND;
    bool mem = (flags & CV_STORAGE_MEMORY) != 0;
    bool write_mode = (flags & 3) != 0;
    bool write_base64 = (write_mode || append) && (flags & CV_STORAGE_BASE64) != 0;
    bool isGZ = false;
    size_t fnamelen = 0;
    const char * filename = query;

    std::vector<std::string> params;
    if ( !mem )
    {
        params = analyze_file_name( query );
        if ( !params.empty() )
            filename = params.begin()->c_str();

        if ( write_base64 == false && is_param_exist( params, "base64" ) )
            write_base64 = (write_mode || append);
    }

    if( !filename || filename[0] == '\0' )
    {
        if( !write_mode )
            CV_Error( CV_StsNullPtr, mem ? "NULL or empty filename" : "NULL or empty buffer" );
        mem = true;
    }
    else
        fnamelen = strlen(filename);

    if( mem && append )
        CV_Error( CV_StsBadFlag, "CV_STORAGE_APPEND and CV_STORAGE_MEMORY are not currently compatible" );

    fs->memstorage = cvCreateMemStorage( default_block_size );
    fs->dststorage = dststorage ? dststorage : fs->memstorage;

    fs->flags = CV_FILE_STORAGE;
    fs->write_mode = write_mode;

    if( !mem )
    {
        fs->filename = (char*)cvMemStorageAlloc( fs->memstorage, fnamelen+1 );
        strcpy( fs->filename, filename );

        char* dot_pos = strrchr(fs->filename, '.');
        char compression = '\0';

        if( dot_pos && dot_pos[1] == 'g' && dot_pos[2] == 'z' &&
            (dot_pos[3] == '\0' || (cv_isdigit(dot_pos[3]) && dot_pos[4] == '\0')) )
        {
            if( append )
            {
                cvReleaseFileStorage( &fs );
                CV_Error(CV_StsNotImplemented, "Appending data to compressed file is not implemented" );
            }
            isGZ = true;
            compression = dot_pos[3];
            if( compression )
                dot_pos[3] = '\0', fnamelen--;
        }

        if( !isGZ )
        {
            fs->file = fopen(fs->filename, !fs->write_mode ? "rt" : !append ? "wt" : "a+t" );
            if( !fs->file )
                goto _exit_;
        }
        else
        {
            #if USE_ZLIB
            char mode[] = { fs->write_mode ? 'w' : 'r', 'b', compression ? compression : '3', '\0' };
            fs->gzfile = gzopen(fs->filename, mode);
            if( !fs->gzfile )
                goto _exit_;
            #else
            cvReleaseFileStorage( &fs );
            CV_Error(CV_StsNotImplemented, "There is no compressed file storage support in this configuration");
            #endif
        }
    }

    fs->roots = 0;
    fs->struct_indent = 0;
    fs->struct_flags = 0;
    fs->wrap_margin = 71;

    if( fs->write_mode )
    {
        int fmt = flags & CV_STORAGE_FORMAT_MASK;

        if( mem )
            fs->outbuf = new std::deque<char>;

        if( fmt == CV_STORAGE_FORMAT_AUTO && filename )
        {
            const char* dot_pos = NULL;
            const char* dot_pos2 = NULL;
            // like strrchr() implementation, but save two last positions simultaneously
            for (const char* pos = filename; pos[0] != 0; pos++)
            {
                if (pos[0] == '.')
                {
                    dot_pos2 = dot_pos;
                    dot_pos = pos;
                }
            }
            if (cv_strcasecmp(dot_pos, ".gz") && dot_pos2 != NULL)
            {
                dot_pos = dot_pos2;
            }
            fs->fmt
                = (cv_strcasecmp(dot_pos, ".xml") || cv_strcasecmp(dot_pos, ".xml.gz"))
                ? CV_STORAGE_FORMAT_XML
                : (cv_strcasecmp(dot_pos, ".json") || cv_strcasecmp(dot_pos, ".json.gz"))
                ? CV_STORAGE_FORMAT_JSON
                : CV_STORAGE_FORMAT_YAML
                ;
        }
        else if ( fmt != CV_STORAGE_FORMAT_AUTO )
        {
            fs->fmt = fmt;
        }
        else
        {
            fs->fmt = CV_STORAGE_FORMAT_XML;
        }

        // we use factor=6 for XML (the longest characters (' and ") are encoded with 6 bytes (&apos; and &quot;)
        // and factor=4 for YAML ( as we use 4 bytes for non ASCII characters (e.g. \xAB))
        int buf_size = CV_FS_MAX_LEN*(fs->fmt == CV_STORAGE_FORMAT_XML ? 6 : 4) + 1024;

        if (append)
        {
            fseek( fs->file, 0, SEEK_END );
            if (ftell(fs->file) == 0)
                append = false;
        }

        fs->write_stack = cvCreateSeq( 0, sizeof(CvSeq), fs->fmt == CV_STORAGE_FORMAT_XML ?
                sizeof(CvXMLStackRecord) : sizeof(int), fs->memstorage );
        fs->is_first = 1;
        fs->struct_indent = 0;
        fs->struct_flags = CV_NODE_EMPTY;
        fs->buffer_start = fs->buffer = (char*)cvAlloc( buf_size + 1024 );
        fs->buffer_end = fs->buffer_start + buf_size;

        fs->base64_writer           = 0;
        fs->is_default_using_base64 = write_base64;
        fs->state_of_writing_base64 = base64::fs::Uncertain;

        fs->is_write_struct_delayed = false;
        fs->delayed_struct_key      = 0;
        fs->delayed_struct_flags    = 0;
        fs->delayed_type_name       = 0;

        if( fs->fmt == CV_STORAGE_FORMAT_XML )
        {
            size_t file_size = fs->file ? (size_t)ftell( fs->file ) : (size_t)0;
            fs->strstorage = cvCreateChildMemStorage( fs->memstorage );
            if( !append || file_size == 0 )
            {
                if( encoding )
                {
                    if( strcmp( encoding, "UTF-16" ) == 0 ||
                        strcmp( encoding, "utf-16" ) == 0 ||
                        strcmp( encoding, "Utf-16" ) == 0 )
                    {
                        cvReleaseFileStorage( &fs );
                        CV_Error( CV_StsBadArg, "UTF-16 XML encoding is not supported! Use 8-bit encoding\n");
                    }

                    CV_Assert( strlen(encoding) < 1000 );
                    char buf[1100];
                    sprintf(buf, "<?xml version=\"1.0\" encoding=\"%s\"?>\n", encoding);
                    icvPuts( fs, buf );
                }
                else
                    icvPuts( fs, "<?xml version=\"1.0\"?>\n" );
                icvPuts( fs, "<opencv_storage>\n" );
            }
            else
            {
                int xml_buf_size = 1 << 10;
                char substr[] = "</opencv_storage>";
                int last_occurence = -1;
                xml_buf_size = MIN(xml_buf_size, int(file_size));
                fseek( fs->file, -xml_buf_size, SEEK_END );
                char* xml_buf = (char*)cvAlloc( xml_buf_size+2 );
                // find the last occurrence of </opencv_storage>
                for(;;)
                {
                    int line_offset = (int)ftell( fs->file );
                    char* ptr0 = icvGets( fs, xml_buf, xml_buf_size ), *ptr;
                    if( !ptr0 )
                        break;
                    ptr = ptr0;
                    for(;;)
                    {
                        ptr = strstr( ptr, substr );
                        if( !ptr )
                            break;
                        last_occurence = line_offset + (int)(ptr - ptr0);
                        ptr += strlen(substr);
                    }
                }
                cvFree( &xml_buf );
                if( last_occurence < 0 )
                {
                    cvReleaseFileStorage( &fs );
                    CV_Error( CV_StsError, "Could not find </opencv_storage> in the end of file.\n" );
                }
                icvCloseFile( fs );
                fs->file = fopen( fs->filename, "r+t" );
                CV_Assert(fs->file);
                fseek( fs->file, last_occurence, SEEK_SET );
                // replace the last "</opencv_storage>" with " <!-- resumed -->", which has the same length
                icvPuts( fs, " <!-- resumed -->" );
                fseek( fs->file, 0, SEEK_END );
                icvPuts( fs, "\n" );
            }
            fs->start_write_struct = icvXMLStartWriteStruct;
            fs->end_write_struct = icvXMLEndWriteStruct;
            fs->write_int = icvXMLWriteInt;
            fs->write_real = icvXMLWriteReal;
            fs->write_string = icvXMLWriteString;
            fs->write_comment = icvXMLWriteComment;
            fs->start_next_stream = icvXMLStartNextStream;
        }
        else if( fs->fmt == CV_STORAGE_FORMAT_YAML )
        {
            if( !append)
                icvPuts( fs, "%YAML:1.0\n---\n" );
            else
                icvPuts( fs, "...\n---\n" );
            fs->start_write_struct = icvYMLStartWriteStruct;
            fs->end_write_struct = icvYMLEndWriteStruct;
            fs->write_int = icvYMLWriteInt;
            fs->write_real = icvYMLWriteReal;
            fs->write_string = icvYMLWriteString;
            fs->write_comment = icvYMLWriteComment;
            fs->start_next_stream = icvYMLStartNextStream;
        }
        else
        {
            if( !append )
                icvPuts( fs, "{\n" );
            else
            {
                bool valid = false;
                long roffset = 0;
                for ( ;
                      fseek( fs->file, roffset, SEEK_END ) == 0;
                      roffset -= 1 )
                {
                    const char end_mark = '}';
                    if ( fgetc( fs->file ) == end_mark )
                    {
                        fseek( fs->file, roffset, SEEK_END );
                        valid = true;
                        break;
                    }
                }

                if ( valid )
                {
                    icvCloseFile( fs );
                    fs->file = fopen( fs->filename, "r+t" );
                    CV_Assert(fs->file);
                    fseek( fs->file, roffset, SEEK_END );
                    fputs( ",", fs->file );
                }
                else
                {
                    CV_Error( CV_StsError, "Could not find '}' in the end of file.\n" );
                }
            }
            fs->struct_indent = 4;
            fs->start_write_struct = icvJSONStartWriteStruct;
            fs->end_write_struct = icvJSONEndWriteStruct;
            fs->write_int = icvJSONWriteInt;
            fs->write_real = icvJSONWriteReal;
            fs->write_string = icvJSONWriteString;
            fs->write_comment = icvJSONWriteComment;
            fs->start_next_stream = icvJSONStartNextStream;
        }
    }
    else
    {
        if( mem )
        {
            fs->strbuf = filename;
            fs->strbufsize = fnamelen;
        }

        size_t buf_size = 1 << 20;
        const char* yaml_signature = "%YAML";
        const char* json_signature = "{";
        const char* xml_signature  = "<?xml";
        char buf[16] = { 0 };
        char* bufPtr = icvGets( fs, buf, sizeof(buf)-2);
        if (!bufPtr)
            CV_Error(CV_StsBadArg, "Can't read from input stream or input stream is empty");
        bufPtr = cv_skip_BOM(bufPtr);
        CV_Assert(bufPtr);
        size_t bufOffset = bufPtr - buf;

        if(strncmp( bufPtr, yaml_signature, strlen(yaml_signature) ) == 0)
            fs->fmt = CV_STORAGE_FORMAT_YAML;
        else if(strncmp( bufPtr, json_signature, strlen(json_signature) ) == 0)
            fs->fmt = CV_STORAGE_FORMAT_JSON;
        else if(strncmp( bufPtr, xml_signature, strlen(xml_signature) ) == 0)
            fs->fmt = CV_STORAGE_FORMAT_XML;
        else if(fs->strbufsize  == bufOffset)
            CV_Error(CV_StsBadArg, "Input file is empty");
        else
            CV_Error(CV_StsBadArg, "Unsupported file storage format");

        if( !isGZ )
        {
            if( !mem )
            {
                fseek( fs->file, 0, SEEK_END );
                buf_size = ftell( fs->file );
            }
            else
                buf_size = fs->strbufsize;
            buf_size = MIN( buf_size, (size_t)(1 << 20) );
            buf_size = MAX( buf_size, (size_t)(CV_FS_MAX_LEN*2 + 1024) );
        }
        icvRewind(fs);
        fs->strbufpos = bufOffset;

        fs->str_hash = cvCreateMap( 0, sizeof(CvStringHash),
                        sizeof(CvStringHashNode), fs->memstorage, 256 );

        fs->roots = cvCreateSeq( 0, sizeof(CvSeq),
                        sizeof(CvFileNode), fs->memstorage );

        fs->buffer = fs->buffer_start = (char*)cvAlloc( buf_size + 256 );
        fs->buffer_end = fs->buffer_start + buf_size;
        fs->buffer[0] = '\n';
        fs->buffer[1] = '\0';

        //mode = cvGetErrMode();
        //cvSetErrMode( CV_ErrModeSilent );
        switch (fs->fmt)
        {
        case CV_STORAGE_FORMAT_XML : { icvXMLParse ( fs ); break; }
        case CV_STORAGE_FORMAT_YAML: { icvYMLParse ( fs ); break; }
        case CV_STORAGE_FORMAT_JSON: { icvJSONParse( fs ); break; }
        default: break;
        }
        //cvSetErrMode( mode );

        // release resources that we do not need anymore
        cvFree( &fs->buffer_start );
        fs->buffer = fs->buffer_end = 0;
    }
    fs->is_opened = true;

_exit_:
    if( fs )
    {
        if( cvGetErrStatus() < 0 || (!fs->file && !fs->gzfile && !fs->outbuf && !fs->strbuf) )
        {
            cvReleaseFileStorage( &fs );
        }
        else if( !fs->write_mode )
        {
            icvCloseFile(fs);
            // we close the file since it's not needed anymore. But icvCloseFile() resets is_opened,
            // which may be misleading. Since we restore the value of is_opened.
            fs->is_opened = true;
        }
    }

    return;
}

CV_IMPL CvFileStorage*
cvOpenFileStorage(const char* query, CvMemStorage* dststorage, int flags, const char* encoding)
{
    CvFileStorage* fs = (CvFileStorage*)cvAlloc( sizeof(*fs) );
    CV_Assert(fs);
    memset( fs, 0, sizeof(*fs));

    try
    {
        cvOpenFileStorage_(fs, query, dststorage, flags, encoding);
        return fs;
    }
    catch (...)
    {
        if (fs)
            cvReleaseFileStorage(&fs);
        throw;
    }
}

/* closes file storage and deallocates buffers */
CV_IMPL  void
cvReleaseFileStorage( CvFileStorage** p_fs )
{
    if( !p_fs )
        CV_Error( CV_StsNullPtr, "NULL double pointer to file storage" );

    if( *p_fs )
    {
        CvFileStorage* fs = *p_fs;
        *p_fs = 0;

        icvClose(fs, 0);

        cvReleaseMemStorage( &fs->strstorage );
        cvFree( &fs->buffer_start );
        cvReleaseMemStorage( &fs->memstorage );

        delete fs->outbuf;
        delete fs->base64_writer;
        delete[] fs->delayed_struct_key;
        delete[] fs->delayed_type_name;

        memset( fs, 0, sizeof(*fs) );
        cvFree( &fs );
    }
}


CV_IMPL void*
cvLoad( const char* filename, CvMemStorage* memstorage,
        const char* name, const char** _real_name )
{
    void* ptr = 0;
    const char* real_name = 0;
    CvFileStorage* fs = cvOpenFileStorage(filename, memstorage, CV_STORAGE_READ);

    CvFileNode* node = 0;

    if( !fs )
        return 0;

    if( name )
    {
        node = cvGetFileNodeByName( fs, 0, name );
    }
    else
    {
        int i, k;
        for( k = 0; k < fs->roots->total; k++ )
        {
            CvSeq* seq;
            CvSeqReader reader;

            node = (CvFileNode*)cvGetSeqElem( fs->roots, k );
            CV_Assert(node != NULL);
            if( !CV_NODE_IS_MAP( node->tag ))
                return 0;
            seq = node->data.seq;
            node = 0;

            cvStartReadSeq( seq, &reader, 0 );

            // find the first element in the map
            for( i = 0; i < seq->total; i++ )
            {
                if( CV_IS_SET_ELEM( reader.ptr ))
                {
                    node = (CvFileNode*)reader.ptr;
                    goto stop_search;
                }
                CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
            }
        }

stop_search:
        ;
    }

    if( !node )
        CV_Error( CV_StsObjectNotFound, "Could not find the/an object in file storage" );

    real_name = cvGetFileNodeName( node );
    ptr = cvRead( fs, node, 0 );

    // sanity check
    if( !memstorage && (CV_IS_SEQ( ptr ) || CV_IS_SET( ptr )) )
        CV_Error( CV_StsNullPtr,
        "NULL memory storage is passed - the loaded dynamic structure can not be stored" );

    if( cvGetErrStatus() < 0 )
    {
        cvRelease( (void**)&ptr );
        real_name = 0;
    }

    if( _real_name)
    {
    if (real_name)
    {
        *_real_name = (const char*)cvAlloc(strlen(real_name));
            memcpy((void*)*_real_name, real_name, strlen(real_name));
    } else {
        *_real_name = 0;
    }
    }

    cvReleaseFileStorage(&fs);
    return ptr;
}

CV_IMPL const char*
cvAttrValue( const CvAttrList* attr, const char* attr_name )
{
    while( attr && attr->attr )
    {
        int i;
        for( i = 0; attr->attr[i*2] != 0; i++ )
        {
            if( strcmp( attr_name, attr->attr[i*2] ) == 0 )
                return attr->attr[i*2+1];
        }
        attr = attr->next;
    }

    return 0;
}


#define CV_HASHVAL_SCALE 33

CV_IMPL CvString cvMemStorageAllocString( CvMemStorage* storage, const char* ptr, int len )
{
    CvString str;
    memset(&str, 0, sizeof(CvString));
    str.len = len >= 0 ? len : (int)strlen(ptr);
    str.ptr = (char*)cvMemStorageAlloc( storage, str.len + 1 );
    memcpy( str.ptr, ptr, str.len );
    str.ptr[str.len] = '\0';
    return str;
}

CV_IMPL CvStringHashNode*
cvGetHashedKey( CvFileStorage* fs, const char* str, int len, int create_missing )
{
    CvStringHashNode* node = 0;
    unsigned hashval = 0;
    int i, tab_size;

    if( !fs )
        return 0;

    CvStringHash* map = fs->str_hash;

    if( len < 0 )
    {
        for( i = 0; str[i] != '\0'; i++ )
            hashval = hashval*CV_HASHVAL_SCALE + (unsigned char)str[i];
        len = i;
    }
    else for( i = 0; i < len; i++ )
        hashval = hashval*CV_HASHVAL_SCALE + (unsigned char)str[i];

    hashval &= INT_MAX;
    tab_size = map->tab_size;
    if( (tab_size & (tab_size - 1)) == 0 )
        i = (int)(hashval & (tab_size - 1));
    else
        i = (int)(hashval % tab_size);

    for( node = (CvStringHashNode*)(map->table[i]); node != 0; node = node->next )
    {
        if( node->hashval == hashval &&
            node->str.len == len &&
            memcmp( node->str.ptr, str, len ) == 0 )
            break;
    }

    if( !node && create_missing )
    {
        node = (CvStringHashNode*)cvSetNew( (CvSet*)map );
        node->hashval = hashval;
        node->str = cvMemStorageAllocString( map->storage, str, len );
        node->next = (CvStringHashNode*)(map->table[i]);
        map->table[i] = node;
    }

    return node;
}

CV_IMPL CvFileNode*
cvGetFileNode( CvFileStorage* fs, CvFileNode* _map_node,
               const CvStringHashNode* key,
               int create_missing )
{
    CvFileNode* value = 0;
    int k = 0, attempts = 1;

    if( !fs )
        return 0;

    CV_CHECK_FILE_STORAGE(fs);

    if( !key )
        CV_Error( CV_StsNullPtr, "Null key element" );

    if( _map_node )
    {
        if( !fs->roots )
            return 0;
        attempts = fs->roots->total;
    }

    for( k = 0; k < attempts; k++ )
    {
        int i, tab_size;
        CvFileNode* map_node = _map_node;
        CvFileMapNode* another;
        CvFileNodeHash* map;

        if( !map_node )
            map_node = (CvFileNode*)cvGetSeqElem( fs->roots, k );
        CV_Assert(map_node != NULL);
        if( !CV_NODE_IS_MAP(map_node->tag) )
        {
            if( (!CV_NODE_IS_SEQ(map_node->tag) || map_node->data.seq->total != 0) &&
                CV_NODE_TYPE(map_node->tag) != CV_NODE_NONE )
                CV_Error( CV_StsError, "The node is neither a map nor an empty collection" );
            return 0;
        }

        map = map_node->data.map;
        tab_size = map->tab_size;

        if( (tab_size & (tab_size - 1)) == 0 )
            i = (int)(key->hashval & (tab_size - 1));
        else
            i = (int)(key->hashval % tab_size);

        for( another = (CvFileMapNode*)(map->table[i]); another != 0; another = another->next )
            if( another->key == key )
            {
                if( !create_missing )
                {
                    value = &another->value;
                    return value;
                }
                CV_PARSE_ERROR( "Duplicated key" );
            }

        if( k == attempts - 1 && create_missing )
        {
            CvFileMapNode* node = (CvFileMapNode*)cvSetNew( (CvSet*)map );
            node->key = key;

            node->next = (CvFileMapNode*)(map->table[i]);
            map->table[i] = node;
            value = (CvFileNode*)node;
        }
    }

    return value;
}

CV_IMPL CvFileNode*
cvGetFileNodeByName( const CvFileStorage* fs, const CvFileNode* _map_node, const char* str )
{
    CvFileNode* value = 0;
    int i, len, tab_size;
    unsigned hashval = 0;
    int k = 0, attempts = 1;

    if( !fs )
        return 0;

    CV_CHECK_FILE_STORAGE(fs);

    if( !str )
        CV_Error( CV_StsNullPtr, "Null element name" );

    for( i = 0; str[i] != '\0'; i++ )
        hashval = hashval*CV_HASHVAL_SCALE + (unsigned char)str[i];
    hashval &= INT_MAX;
    len = i;

    if( !_map_node )
    {
        if( !fs->roots )
            return 0;
        attempts = fs->roots->total;
    }

    for( k = 0; k < attempts; k++ )
    {
        CvFileNodeHash* map;
        const CvFileNode* map_node = _map_node;
        CvFileMapNode* another;

        if( !map_node )
            map_node = (CvFileNode*)cvGetSeqElem( fs->roots, k );

        if( !CV_NODE_IS_MAP(map_node->tag) )
        {
            if( (!CV_NODE_IS_SEQ(map_node->tag) || map_node->data.seq->total != 0) &&
                CV_NODE_TYPE(map_node->tag) != CV_NODE_NONE )
                CV_Error( CV_StsError, "The node is neither a map nor an empty collection" );
            return 0;
        }

        map = map_node->data.map;
        tab_size = map->tab_size;

        if( (tab_size & (tab_size - 1)) == 0 )
            i = (int)(hashval & (tab_size - 1));
        else
            i = (int)(hashval % tab_size);

        for( another = (CvFileMapNode*)(map->table[i]); another != 0; another = another->next )
        {
            const CvStringHashNode* key = another->key;

            if( key->hashval == hashval &&
                key->str.len == len &&
                memcmp( key->str.ptr, str, len ) == 0 )
            {
                value = &another->value;
                return value;
            }
        }
    }

    return value;
}

CV_IMPL CvFileNode*
cvGetRootFileNode( const CvFileStorage* fs, int stream_index )
{
    CV_CHECK_FILE_STORAGE(fs);

    if( !fs->roots || (unsigned)stream_index >= (unsigned)fs->roots->total )
        return 0;

    return (CvFileNode*)cvGetSeqElem( fs->roots, stream_index );
}

CV_IMPL void
cvStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags,
                    const char* type_name, CvAttrList /*attributes*/ )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    check_if_write_struct_is_delayed( fs );
    if ( fs->state_of_writing_base64 == base64::fs::NotUse )
        switch_to_Base64_state( fs, base64::fs::Uncertain );

    if ( fs->state_of_writing_base64 == base64::fs::Uncertain
        &&
        CV_NODE_IS_SEQ(struct_flags)
        &&
        fs->is_default_using_base64
        &&
        type_name == 0
       )
    {
        /* Uncertain whether output Base64 data */
        make_write_struct_delayed( fs, key, struct_flags, type_name );
    }
    else if ( type_name && memcmp(type_name, "binary", 6) == 0 )
    {
        /* Must output Base64 data */
        if ( !CV_NODE_IS_SEQ(struct_flags) )
            CV_Error( CV_StsBadArg, "must set 'struct_flags |= CV_NODE_SEQ' if using Base64.");
        else if ( fs->state_of_writing_base64 != base64::fs::Uncertain )
            CV_Error( CV_StsError, "function \'cvStartWriteStruct\' calls cannot be nested if using Base64.");

        fs->start_write_struct( fs, key, struct_flags, type_name );

        if ( fs->state_of_writing_base64 != base64::fs::Uncertain )
            switch_to_Base64_state( fs, base64::fs::Uncertain );
        switch_to_Base64_state( fs, base64::fs::InUse );
    }
    else
    {
        /* Won't output Base64 data */
        if ( fs->state_of_writing_base64 == base64::fs::InUse )
            CV_Error( CV_StsError, "At the end of the output Base64, `cvEndWriteStruct` is needed.");

        fs->start_write_struct( fs, key, struct_flags, type_name );

        if ( fs->state_of_writing_base64 != base64::fs::Uncertain )
            switch_to_Base64_state( fs, base64::fs::Uncertain );
        switch_to_Base64_state( fs, base64::fs::NotUse );
    }
}


CV_IMPL void
cvEndWriteStruct( CvFileStorage* fs )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    check_if_write_struct_is_delayed( fs );

    if ( fs->state_of_writing_base64 != base64::fs::Uncertain )
        switch_to_Base64_state( fs, base64::fs::Uncertain );

    fs->end_write_struct( fs );
}


CV_IMPL void
cvWriteInt( CvFileStorage* fs, const char* key, int value )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->write_int( fs, key, value );
}


CV_IMPL void
cvWriteReal( CvFileStorage* fs, const char* key, double value )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->write_real( fs, key, value );
}


CV_IMPL void
cvWriteString( CvFileStorage* fs, const char* key, const char* value, int quote )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->write_string( fs, key, value, quote );
}


CV_IMPL void
cvWriteComment( CvFileStorage* fs, const char* comment, int eol_comment )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->write_comment( fs, comment, eol_comment );
}


CV_IMPL void
cvStartNextStream( CvFileStorage* fs )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->start_next_stream( fs );
}

CV_IMPL void
cvWriteRawData( CvFileStorage* fs, const void* _data, int len, const char* dt )
{
    if (fs->is_default_using_base64 ||
        fs->state_of_writing_base64 == base64::fs::InUse )
    {
        cvWriteRawDataBase64( fs, _data, len, dt );
        return;
    }
    else if ( fs->state_of_writing_base64 == base64::fs::Uncertain )
    {
        switch_to_Base64_state( fs, base64::fs::NotUse );
    }

    const char* data0 = (const char*)_data;
    int offset = 0;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS*2], k, fmt_pair_count;
    char buf[256] = "";

    CV_CHECK_OUTPUT_FILE_STORAGE( fs );

    if( len < 0 )
        CV_Error( CV_StsOutOfRange, "Negative number of elements" );

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );

    if( !len )
        return;

    if( !data0 )
        CV_Error( CV_StsNullPtr, "Null data pointer" );

    if( fmt_pair_count == 1 )
    {
        fmt_pairs[0] *= len;
        len = 1;
    }

    for(;len--;)
    {
        for( k = 0; k < fmt_pair_count; k++ )
        {
            int i, count = fmt_pairs[k*2];
            int elem_type = fmt_pairs[k*2+1];
            int elem_size = CV_ELEM_SIZE(elem_type);
            const char* data, *ptr;

            offset = cvAlign( offset, elem_size );
            data = data0 + offset;

            for( i = 0; i < count; i++ )
            {
                switch( elem_type )
                {
                case CV_8U:
                    ptr = icv_itoa( *(uchar*)data, buf, 10 );
                    data++;
                    break;
                case CV_8S:
                    ptr = icv_itoa( *(char*)data, buf, 10 );
                    data++;
                    break;
                case CV_16U:
                    ptr = icv_itoa( *(ushort*)data, buf, 10 );
                    data += sizeof(ushort);
                    break;
                case CV_16S:
                    ptr = icv_itoa( *(short*)data, buf, 10 );
                    data += sizeof(short);
                    break;
                case CV_32S:
                    ptr = icv_itoa( *(int*)data, buf, 10 );
                    data += sizeof(int);
                    break;
                case CV_32F:
                    ptr = icvFloatToString( buf, *(float*)data );
                    data += sizeof(float);
                    break;
                case CV_64F:
                    ptr = icvDoubleToString( buf, *(double*)data );
                    data += sizeof(double);
                    break;
                //case CV_USRTYPE1: /* reference */
                //    ptr = icv_itoa( (int)*(size_t*)data, buf, 10 );
                //    data += sizeof(size_t);
                //    break;
                default:
                    CV_Error( CV_StsUnsupportedFormat, "Unsupported type" );
                    return;
                }

                if( fs->fmt == CV_STORAGE_FORMAT_XML )
                {
                    int buf_len = (int)strlen(ptr);
                    icvXMLWriteScalar( fs, 0, ptr, buf_len );
                }
                else if ( fs->fmt == CV_STORAGE_FORMAT_YAML )
                {
                    icvYMLWrite( fs, 0, ptr );
                }
                else
                {
                    if( elem_type == CV_32F || elem_type == CV_64F )
                    {
                        size_t buf_len = strlen(ptr);
                        if( buf_len > 0 && ptr[buf_len-1] == '.' )
                        {
                            // append zero if CV_32F or CV_64F string ends with decimal place to match JSON standard
                            // ptr will point to buf, so can write to buf given ptr is const
                            buf[buf_len] = '0';
                            buf[buf_len+1] = '\0';
                        }
                    }
                    icvJSONWrite( fs, 0, ptr );
                }
            }

            offset = (int)(data - data0);
        }
    }
}

CV_IMPL void
cvStartReadRawData( const CvFileStorage* fs, const CvFileNode* src, CvSeqReader* reader )
{
    int node_type;
    CV_CHECK_FILE_STORAGE( fs );

    if( !src || !reader )
        CV_Error( CV_StsNullPtr, "Null pointer to source file node or reader" );

    node_type = CV_NODE_TYPE(src->tag);
    if( node_type == CV_NODE_INT || node_type == CV_NODE_REAL )
    {
        // emulate reading from 1-element sequence
        reader->ptr = (schar*)src;
        reader->block_max = reader->ptr + sizeof(*src)*2;
        reader->block_min = reader->ptr;
        reader->seq = 0;
    }
    else if( node_type == CV_NODE_SEQ )
    {
        cvStartReadSeq( src->data.seq, reader, 0 );
    }
    else if( node_type == CV_NODE_NONE )
    {
        memset( reader, 0, sizeof(*reader) );
    }
    else
        CV_Error( CV_StsBadArg, "The file node should be a numerical scalar or a sequence" );
}


CV_IMPL void
cvReadRawDataSlice( const CvFileStorage* fs, CvSeqReader* reader,
                    int len, void* _data, const char* dt )
{
    char* data0 = (char*)_data;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS*2], k = 0, fmt_pair_count;
    int i = 0, count = 0;

    CV_CHECK_FILE_STORAGE( fs );

    if( !reader || !data0 )
        CV_Error( CV_StsNullPtr, "Null pointer to reader or destination array" );

    if( !reader->seq && len != 1 )
        CV_Error( CV_StsBadSize, "The read sequence is a scalar, thus len must be 1" );

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    size_t step = ::icvCalcStructSize(dt, 0);

    for(;;)
    {
        int offset = 0;
        for( k = 0; k < fmt_pair_count; k++ )
        {
            int elem_type = fmt_pairs[k*2+1];
            int elem_size = CV_ELEM_SIZE(elem_type);
            char* data;

            count = fmt_pairs[k*2];
            offset = cvAlign( offset, elem_size );
            data = data0 + offset;

            for( i = 0; i < count; i++ )
            {
                CvFileNode* node = (CvFileNode*)reader->ptr;
                if( CV_NODE_IS_INT(node->tag) )
                {
                    int ival = node->data.i;

                    switch( elem_type )
                    {
                    case CV_8U:
                        *(uchar*)data = cv::saturate_cast<uchar>(ival);
                        data++;
                        break;
                    case CV_8S:
                        *(char*)data = cv::saturate_cast<schar>(ival);
                        data++;
                        break;
                    case CV_16U:
                        *(ushort*)data = cv::saturate_cast<ushort>(ival);
                        data += sizeof(ushort);
                        break;
                    case CV_16S:
                        *(short*)data = cv::saturate_cast<short>(ival);
                        data += sizeof(short);
                        break;
                    case CV_32S:
                        *(int*)data = ival;
                        data += sizeof(int);
                        break;
                    case CV_32F:
                        *(float*)data = (float)ival;
                        data += sizeof(float);
                        break;
                    case CV_64F:
                        *(double*)data = (double)ival;
                        data += sizeof(double);
                        break;
                    //case CV_USRTYPE1: /* reference */
                    //    *(size_t*)data = ival;
                    //    data += sizeof(size_t);
                    //    break;
                    default:
                        CV_Error( CV_StsUnsupportedFormat, "Unsupported type" );
                        return;
                    }
                }
                else if( CV_NODE_IS_REAL(node->tag) )
                {
                    double fval = node->data.f;
                    int ival;

                    switch( elem_type )
                    {
                    case CV_8U:
                        ival = cvRound(fval);
                        *(uchar*)data = cv::saturate_cast<uchar>(ival);
                        data++;
                        break;
                    case CV_8S:
                        ival = cvRound(fval);
                        *(char*)data = cv::saturate_cast<schar>(ival);
                        data++;
                        break;
                    case CV_16U:
                        ival = cvRound(fval);
                        *(ushort*)data = cv::saturate_cast<ushort>(ival);
                        data += sizeof(ushort);
                        break;
                    case CV_16S:
                        ival = cvRound(fval);
                        *(short*)data = cv::saturate_cast<short>(ival);
                        data += sizeof(short);
                        break;
                    case CV_32S:
                        ival = cvRound(fval);
                        *(int*)data = ival;
                        data += sizeof(int);
                        break;
                    case CV_32F:
                        *(float*)data = (float)fval;
                        data += sizeof(float);
                        break;
                    case CV_64F:
                        *(double*)data = fval;
                        data += sizeof(double);
                        break;
                    //case CV_USRTYPE1: /* reference */
                    //    ival = cvRound(fval);
                    //    *(size_t*)data = ival;
                    //    data += sizeof(size_t);
                    //    break;
                    default:
                        CV_Error( CV_StsUnsupportedFormat, "Unsupported type" );
                        return;
                    }
                }
                else
                    CV_Error( CV_StsError,
                    "The sequence element is not a numerical scalar" );

                CV_NEXT_SEQ_ELEM( sizeof(CvFileNode), *reader );
                if( !--len )
                    goto end_loop;
            }

            offset = (int)(data - data0);
        }
        data0 += step;
    }

end_loop:
    if( i != count - 1 || k != fmt_pair_count - 1 )
        CV_Error( CV_StsBadSize,
        "The sequence slice does not fit an integer number of records" );

    if( !reader->seq )
        reader->ptr -= sizeof(CvFileNode);
}


CV_IMPL void
cvReadRawData( const CvFileStorage* fs, const CvFileNode* src,
               void* data, const char* dt )
{
    CvSeqReader reader;

    if( !src || !data )
        CV_Error( CV_StsNullPtr, "Null pointers to source file node or destination array" );

    cvStartReadRawData( fs, src, &reader );
    cvReadRawDataSlice( fs, &reader, CV_NODE_IS_SEQ(src->tag) ?
                        src->data.seq->total : 1, data, dt );
}


CV_IMPL void
cvWriteFileNode( CvFileStorage* fs, const char* new_node_name,
                 const CvFileNode* node, int embed )
{
    CvFileStorage* dst = 0;
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);

    if( !node )
        return;

    if( CV_NODE_IS_COLLECTION(node->tag) && embed )
    {
        icvWriteCollection( fs, node );
    }
    else
    {
        icvWriteFileNode( fs, new_node_name, node );
    }
    /*
    int i, stream_count;
    stream_count = fs->roots->total;
    for( i = 0; i < stream_count; i++ )
    {
        CvFileNode* node = (CvFileNode*)cvGetSeqElem( fs->roots, i, 0 );
        icvDumpCollection( dst, node );
        if( i < stream_count - 1 )
            dst->start_next_stream( dst );
    }*/
    cvReleaseFileStorage( &dst );
}


CV_IMPL const char*
cvGetFileNodeName( const CvFileNode* file_node )
{
    return file_node && CV_NODE_HAS_NAME(file_node->tag) ?
        ((CvFileMapNode*)file_node)->key->str.ptr : 0;
}



CV_IMPL  void
cvRegisterType( const CvTypeInfo* _info )
{
    CvTypeInfo* info = 0;
    int i, len;
    char c;

    //if( !CvType::first )
    //    icvCreateStandardTypes();

    if( !_info || _info->header_size != sizeof(CvTypeInfo) )
        CV_Error( CV_StsBadSize, "Invalid type info" );

    if( !_info->is_instance || !_info->release ||
        !_info->read || !_info->write )
        CV_Error( CV_StsNullPtr,
        "Some of required function pointers "
        "(is_instance, release, read or write) are NULL");

    c = _info->type_name[0];
    if( !cv_isalpha(c) && c != '_' )
        CV_Error( CV_StsBadArg, "Type name should start with a letter or _" );

    len = (int)strlen(_info->type_name);

    for( i = 0; i < len; i++ )
    {
        c = _info->type_name[i];
        if( !cv_isalnum(c) && c != '-' && c != '_' )
            CV_Error( CV_StsBadArg,
            "Type name should contain only letters, digits, - and _" );
    }

    info = (CvTypeInfo*)cvAlloc( sizeof(*info) + len + 1 );

    *info = *_info;
    info->type_name = (char*)(info + 1);
    memcpy( (char*)info->type_name, _info->type_name, len + 1 );

    info->flags = 0;
    info->next = CvType::first;
    info->prev = 0;
    if( CvType::first )
        CvType::first->prev = info;
    else
        CvType::last = info;
    CvType::first = info;
}


CV_IMPL void
cvUnregisterType( const char* type_name )
{
    CvTypeInfo* info;

    info = cvFindType( type_name );
    if( info )
    {
        if( info->prev )
            info->prev->next = info->next;
        else
            CvType::first = info->next;

        if( info->next )
            info->next->prev = info->prev;
        else
            CvType::last = info->prev;

        if( !CvType::first || !CvType::last )
            CvType::first = CvType::last = 0;

        cvFree( &info );
    }
}


CV_IMPL CvTypeInfo*
cvFirstType( void )
{
    return CvType::first;
}


CV_IMPL CvTypeInfo*
cvFindType( const char* type_name )
{
    CvTypeInfo* info = 0;

    if (type_name)
      for( info = CvType::first; info != 0; info = info->next )
        if( strcmp( info->type_name, type_name ) == 0 )
      break;

    return info;
}


CV_IMPL CvTypeInfo*
cvTypeOf( const void* struct_ptr )
{
    CvTypeInfo* info = 0;

    if( struct_ptr )
    {
        for( info = CvType::first; info != 0; info = info->next )
            if( info->is_instance( struct_ptr ))
                break;
    }

    return info;
}


/* universal functions */
CV_IMPL void
cvRelease( void** struct_ptr )
{
    CvTypeInfo* info;

    if( !struct_ptr )
        CV_Error( CV_StsNullPtr, "NULL double pointer" );

    if( *struct_ptr )
    {
        info = cvTypeOf( *struct_ptr );
        if( !info )
            CV_Error( CV_StsError, "Unknown object type" );
        if( !info->release )
            CV_Error( CV_StsError, "release function pointer is NULL" );

        info->release( struct_ptr );
        *struct_ptr = 0;
    }
}


void* cvClone( const void* struct_ptr )
{
    void* struct_copy = 0;
    CvTypeInfo* info;

    if( !struct_ptr )
        CV_Error( CV_StsNullPtr, "NULL structure pointer" );

    info = cvTypeOf( struct_ptr );
    if( !info )
        CV_Error( CV_StsError, "Unknown object type" );
    if( !info->clone )
        CV_Error( CV_StsError, "clone function pointer is NULL" );

    struct_copy = info->clone( struct_ptr );
    return struct_copy;
}


/* reads matrix, image, sequence, graph etc. */
CV_IMPL void*
cvRead( CvFileStorage* fs, CvFileNode* node, CvAttrList* list )
{
    void* obj = 0;
    CV_CHECK_FILE_STORAGE( fs );

    if( !node )
        return 0;

    if( !CV_NODE_IS_USER(node->tag) || !node->info )
        CV_Error( CV_StsError, "The node does not represent a user object (unknown type?)" );

    obj = node->info->read( fs, node );
    if( list )
        *list = cvAttrList(0,0);

    return obj;
}


/* writes matrix, image, sequence, graph etc. */
CV_IMPL void
cvWrite( CvFileStorage* fs, const char* name,
         const void* ptr, CvAttrList attributes )
{
    CvTypeInfo* info;

    CV_CHECK_OUTPUT_FILE_STORAGE( fs );

    if( !ptr )
        CV_Error( CV_StsNullPtr, "Null pointer to the written object" );

    info = cvTypeOf( ptr );
    if( !info )
        CV_Error( CV_StsBadArg, "Unknown object" );

    if( !info->write )
        CV_Error( CV_StsBadArg, "The object does not have write function" );

    info->write( fs, name, ptr, attributes );
}


/* simple API for reading/writing data */
CV_IMPL void
cvSave( const char* filename, const void* struct_ptr,
        const char* _name, const char* comment, CvAttrList attributes )
{
    CvFileStorage* fs = 0;

    if( !struct_ptr )
        CV_Error( CV_StsNullPtr, "NULL object pointer" );

    fs = cvOpenFileStorage( filename, 0, CV_STORAGE_WRITE );
    if( !fs )
        CV_Error( CV_StsError, "Could not open the file storage. Check the path and permissions" );

    cv::String name = _name ? cv::String(_name) : cv::FileStorage::getDefaultObjectName(filename);

    if( comment )
        cvWriteComment( fs, comment, 0 );
    cvWrite( fs, name.c_str(), struct_ptr, attributes );
    cvReleaseFileStorage( &fs );
}

