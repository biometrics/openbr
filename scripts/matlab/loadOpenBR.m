% This script will read a matrix container format where the first two entires are 32-bit ints
% specifying the number of rows and columns (respectively). The remaining buffer is 32-bit floats
% with all the matrix contents, in row-major order.
function [x] = loadOpenBR(filename,reverse)
   
    if nargin < 2,
        % Use this option if the data was serlialized in column-major order.
        reverse = false;
    end

    fileStream = fopen(filename,'r');
    if fileStream == -1,
        fprintf('Error opening file %s\n',filename);
        x = 0;
        return
    end
    
    r = fread(fileStream,1,'int32');
    c = fread(fileStream,1,'int32');
    x = fread(fileStream,[c  r],'float32');
    x = x';  % Matlab reads the file in column-major order
    fclose(fileStream);

    if reverse,
        sz = size(x);
        x = x';
        x = reshape(x,sz);
    end
