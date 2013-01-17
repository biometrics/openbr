function [x2] = loadBin(filename,reverse)
% [x2] = loadBin(filename,reverse)
   
    if nargin < 2,
        reverse = false;
    end

    z = fopen(filename,'r');
    if z == -1,
        fprintf('Error opening file %s\n',filename);
        x2 = 0;
        return
    end
    
    x = fread(z,1,'int32');
    x1 = fread(z,1,'int32');
    x2 = fread(z,[x1  x],'float32');
    x2 = x2';
    fclose(z);

    if reverse,
        sz = size(x2);
        x2 = x2';
        x2 = reshape(x2,sz);
    end
