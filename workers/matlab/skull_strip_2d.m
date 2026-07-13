function Iss = skull_strip_2d(I)
Id = mat2gray(I);
level = graythresh(Id);
bw = imbinarize(Id, level);
bw = imfill(bw, 'holes');
bw = keep_largest(bw);
bw_eroded = imerode(bw, strel('disk', 18));
bw_brain = keep_largest(bw_eroded);
bw_brain = imdilate(bw_brain, strel('disk', 6));
bw_brain = imfill(bw_brain, 'holes');
bw_brain = imopen(bw_brain, strel('disk', 2));
if nnz(bw_brain) < 0.02 * numel(bw_brain)
    Iss = I;
    return
end
Iss = I;
Iss(~bw_brain) = 0;
end

function bw_out = keep_largest(bw_in)
cc = bwconncomp(bw_in);
if cc.NumObjects == 0
    bw_out = bw_in;
    return
end
numPixels = cellfun(@numel, cc.PixelIdxList);
[~, idx] = max(numPixels);
bw_out = false(size(bw_in));
bw_out(cc.PixelIdxList{idx}) = true;
end