function writeUT(handle, matrix, imageID, roi_x, roi_y, roi_width, roi_height, label, url)
% write a single matrix of supported datatype (i.e. not int32) to the file
% handle in UT format, algorithmID 7
% inputs: file handle, single layer matrix, imageID (16 char md5 hash, can
% be empty, in which case we generated a null 16 byte string), roix, roiy,
% roiw, roih (bounding box of matrix roi), label (class label for training)
% and url (can be empty)
% 
% computed values: fvSize, urlSize (url encoded as null-terminated 8-bit
% string, urlSize includes null terminator), a null terminator will be
% added to url by this function
%
% For performance reasons, handle should be opened in 'W', i.e. buffered
% mode.

% 512 -- max supported channels in cv::Mat
if (size(matrix,3) > 512)
    disp('Cannot serialize matrix, 512 is the max depth supported');
    return;
end


% UT format; 
% struct br_universal_template 
% { 
%     unsigned char imageID[16]; /*!< MD5 hash of the undecoded origin file. */ 
%     int32_t  algorithmID; /*!< interpretation of _data_ after _urlSize_. */ 
%     int32_t x;      /*!< region of interest horizontal offset (pixels). */ 
%     int32_t y;      /*!< region of interest vertical offset (pixels). */ 
%     uint32_t width;  /*!< region of interest horizontal size (pixels). */ 
%     uint32_t height; /*!< region of interest vertical size (pixels). */ 
%     uint32_t label; /*!< supervised training class or manually annotated ground truth. */ 
%     uint32_t urlSize; /*!< length of null-terminated URL at the beginning of _data_, 
%                            including the null-terminator character. */ 
%     uint32_t fvSize; /*!< length of the feature vector after the URL in _data_. */ 
%     unsigned char data[]; /*!< (_urlSize_ + _fvSize_)-byte buffer. 
%                                The first _urlSize_ bytes represent the URL. 
%                                The remaining _fvSize_ bytes represent the feature vector. */ 
% }; 

% algorithm 7 binary data format:
% uint16 datatype code (copied from opencv, base datatype codes, single
% channel is assumed)
% uint32 matrix rows
% uint32 matrix cols
% uint16 channel count (max valid is 512)
% channels->rows->columns

% opencv data type definitions 
% #define CV_8U   0
% #define CV_8S   1
% #define CV_16U  2
% #define CV_16S  3
% #define CV_32S  4
% #define CV_32F  5
% #define CV_64F  6
% #define CV_USRTYPE1 7

if (numel(imageID) ~= 16)
    imageID = uint8(zeros(16,1));
end

% fixed for this function 
algorithmID = 7;

% append null terminator
url = [url, '\0'];
% calculate complete string length
urlSize = uint32(length(url));

% figure out datatype code based on the input matrix's data type
matlab_type = class(matrix(1,1));

type_code = uint32(0);
% bytes per element
typeSize = 1;

switch(matlab_type)
    case 'uint8'
        type_code = 0;
        typeSize = 1;
    case 'int8'
        type_code = 1;
        typeSize = 1;
    case 'uint16'
        type_code = 2;
        typeSize = 2;
    case 'int16'
        type_code = 3;
        typeSize = 2;
    case 'uint32'
        disp(' uint32 datatype not supported, please try again');
        return;
    case 'int32'
        type_code = 4;
        typeSize = 4;
        
    case 'single'
        type_code = 5;
        typeSize = 4;
    case 'double'
        type_code = 6;
        typeSize = 8;
    otherwise
        disp(['Unrecognized matlab datatype, ', matlab_type]);
        return;
end

% total size of feature vecotr in bytes, plus 12 byte header encoding
% [uint32 datatype, copied from opencv codes; uint32(matrix width);
% uint32(matrix height)]
fvSize = uint32(typeSize * numel(matrix) + 4*3);

% imageID
fwrite(handle, imageID, 'uint8');

% algorithmID
fwrite(handle, algorithmID, 'int32');

% roi x
fwrite(handle, roi_x, 'uint32');
% roi y
fwrite(handle, roi_y, 'uint32');
% roi width
fwrite(handle, roi_width, 'uint32');

% roi height
fwrite(handle, roi_height, 'uint32');

% label
fwrite(handle, label, 'uint32');

% url size 
fwrite(handle, urlSize, 'uint32');

% feature vector size
fwrite(handle, fvSize, 'uint32');

% url (just writing a single null byte)
fwrite(handle, url, 'uint8');

% binary data header -- datatype code, row count, col count, channel count
% (max 512). Datatype and channel count are uint16, dimensions are uint32
fwrite(handle, type_code,'uint16');
fwrite(handle, uint32(size(matrix,1)), 'uint32');
fwrite(handle, uint32(size(matrix,2)), 'uint32');
fwrite(handle, uint16(size(matrix,3)), 'uint16');

% write data, explicit row-major enumeration, matrix(:) is col-major,
% followed by depth. By permuting the dimensions, we can put the bytes in
% an appropriate order:
permuted = permute(matrix,[3,2,1]);

fwrite(handle, permuted(:), matlab_type);
