function writeUTFVectors(handle, fvectors)
% write the rows of fvectors as separate ut format 7 templates. Dummy
% values will be used for roi settings/imageID/url/etc.
% handle will not be opened/closed by this function.
% 
% see also writeUT

dummy_ID = [];
roi_x = uint32(0);
roi_y = uint32(0);
roi_width = uint32(0);
roi_height = uint32(0);
label = uint32(0);

urlTotal = '';
for i = 1:size(fvectors,1)
    writeUT(handle, fvectors(i,:), dummy_ID, roi_x, roi_y, roi_width, roi_height, label, urlTotal);
end
