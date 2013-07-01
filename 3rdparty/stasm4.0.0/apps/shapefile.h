// shapefile.h: read and use shape files
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_SHAPEFILE_H
#define STASM_SHAPEFILE_H

using std::string;

// define hash_map (currently compiler dependent)
#ifdef _MSC_VER // microsoft
  #include <hash_map>
  using namespace stdext;
#else  // gcc
  #include <ext/hash_map>
  using namespace __gnu_cxx;
  namespace __gnu_cxx
  {
  template<> struct hash<string> // specialization for hash_map<string>
      {
          size_t operator()(const string& s) const
          {
              const std::collate<char>& c =
                  std::use_facet<std::collate<char> >(std::locale::classic());
              return c.hash(s.c_str(), s.c_str() + s.size());
          }
      };
  }
#endif

namespace stasm
{
static const int MAX_MAT_DIM = 10000; // arb, for sanity checking

typedef hash_map<string, MAT> map_mat;

class ShapeFile                   // all the data from a shape file
{
public:
    char             shapepath_[SLEN]; // shapefile path
    char             dirs_[SBIG];      // image dirs at start of shape file
    int              nshapes_;         // number of shapes
    vector<Shape>    shapes_;          // the shapes
    vector<string>   bases_;           // basename of each shape
    vector<unsigned> bits_;            // set of bits for each shape e.g. AT_Pose
    map_mat          poses_;           // pose of each shape
    int              nchar_;           // number of chars in longest basename

    ShapeFile()                                // default constructor
    {
        shapepath_[0] = dirs_[0] = 0;
        nshapes_ = nchar_ = 0;
    }
    ShapeFile(const ShapeFile& sh)             // copy constructor
    {
        Copy_(sh);
    }
    ShapeFile& operator=(const ShapeFile& rhs) // assignment operator
    {
        Copy_(rhs);
        return *this;
    }
    void Open_(                   // read shape file from disk
        const char* shapepath);   // in: shape file path

    void Subset_(                 // select a subset of the shapes
        int         nshapes,      // in: number of shapes to select
        int         seed,         // in: if 0 use first nshapes shapes,
                                  //     else rand subset of nshapes shapes
        const char* sregex);      // in: regex string, NULL or "" to match any

    const MAT Pose_(              // return all INVALID if pose not available
        const string& base)       // in
    const
    {
        static const MAT nopose = (MAT(1, 4) << INVALID, INVALID, INVALID, INVALID);
        map_mat::const_iterator it(poses_.find(base));
        return it == poses_.end()? nopose: it->second;
    }

private:
    void Copy_(const ShapeFile& sh)
    {
        STRCPY(shapepath_, sh.shapepath_);
        STRCPY(dirs_, sh.dirs_);
        nshapes_    = sh.nshapes_;
        shapes_     = sh.shapes_;
        bases_      = sh.bases_;
        bits_       = sh.bits_;
        poses_      = sh.poses_;
        nchar_      = sh.nchar_;
    }
    void SubsetRegex_(             // select shapes matching regex
        const char* sregex);       // in: regex string, NULL or "" to match any

    void SubsetN_(                 // select nshapes
        int        nshapes,        // in: number of shapes to select
        int        seed,           // in: if 0 use first nshapes,
                                   //     else rand subset of nshapes
        const char* sregex);       // in: regex string (used only for err msgs)
};

// Stasm 3.1 tags (used only to convert to and from old format shapefiles)
static const unsigned FA_BadImage   = 0x01;   // image is "bad" in some way (blurred, face tilted, etc.)
static const unsigned FA_Glasses    = 0x02;   // face is wearing specs
static const unsigned FA_Beard      = 0x04;   // beard including possible mustache
static const unsigned FA_Mustache   = 0x08;   // mustache but no beard occluding chin or cheeks
static const unsigned FA_Obscured   = 0x10;   // faces is obscured e.g. by subject's hand
static const unsigned FA_EyesClosed = 0x20;   // eyes closed (partially open is not considered closed)
static const unsigned FA_Expression = 0x40;   // non-neutral expression on face
static const unsigned FA_NnFailed   = 0x80;   // Rowley search failed (does not return 1 face with 2 eyes)
static const unsigned FA_Synthesize = 0x100;  // synthesize eye points from twin landmark
static const unsigned FA_VjFailed   = 0x200;  // Viola Jones detector failed (no face found)
static const unsigned FA_ViolaJones = 0x1000; // Viola Jones detector results
static const unsigned FA_Rowley     = 0x2000; // Rowley detector results
static const unsigned FA_Meta       = 0xF000; // "meta shape" (face detector results)

} // namespace stasm
#endif // STASM_SHAPEFILE_H
