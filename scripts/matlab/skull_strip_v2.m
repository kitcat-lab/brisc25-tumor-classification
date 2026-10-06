function Iss = skull_strip_v2(I)
% **********************************************************************
% skull_strip_v2 - Multi-level Otsu + Active Contours skull stripping
%
% Improved 2D skull stripping for T1 MRI slices. Compared to the simple
% morphological method (skull_strip_2d), this version:
%   1. Uses multi-level Otsu (multithresh, 2 thresholds) to separate
%      background / brain / skull+fat by intensity, instead of a single
%      threshold that merges brain and skull.
%   2. Builds an initial brain mask from the MIDDLE intensity band
%      (brain), excluding the bright outer skull/scalp.
%   3. Refines the mask with active contours (Chan-Vese), which snap to
%      the real brain boundary instead of a fixed erosion radius.
%
% LIMITATION: still a classical method. In sagittal/coronal views where
% facial/neck tissue has brain-like intensity and is connected to the
% brain, it can still leak. Meant for comparison, not clinical grade.
%
% Input  : I  (uint8 grayscale, resized + denoised)
% Output : Iss (uint8, non-brain pixels set to 0)
% *********************************************************************

Id = mat2gray(I);                 % double [0,1]

% 1. Multi-level Otsu: 2 thresholds -> 3 intensity classes
%    (background, brain, skull/fat). If it fails (flat image),
%    fall back to a single Otsu threshold.
try
    t = multithresh(Id, 2);
    Q = imquantize(Id, t);        % labels 1 (dark), 2 (mid), 3 (bright)
catch
    Q = (Id > graythresh(Id)) + 1;
end

% 2. Initial brain mask = middle band (label 2) + bright band (label 3)
%    minus the outer ring. Start with everything that is not background.
bw = Q >= 2;
bw = imfill(bw, 'holes');
bw = keep_largest(bw);            % whole head (brain + skull)

% 3. Peel the bright outer skull/fat band: remove label-3 pixels that
%    sit on the boundary of the head mask.
skull = (Q == 3);
% erode the head slightly and keep only interior -> drops outer skull
head_interior = imerode(bw, strel('disk', 8));
brain_init = bw & ~skull;         % remove bright skull/fat
brain_init = brain_init | head_interior;  % keep deep interior brain
brain_init = imfill(brain_init, 'holes');
brain_init = keep_largest(brain_init);

% Safety: if init mask collapsed, use eroded head as init
if nnz(brain_init) < 0.03 * numel(brain_init)
    brain_init = imerode(bw, strel('disk', 12));
    brain_init = keep_largest(brain_init);
end

% 4. Active contours (Chan-Vese) refine the boundary
%    'Chan-Vese' uses region intensity, good when edges are weak.
%    SmoothFactor keeps the contour from leaking into thin structures.
n_iter = 80;
try
    bw_brain = activecontour(Id, brain_init, n_iter, 'Chan-Vese', ...
        'SmoothFactor', 1.5);
catch
    bw_brain = brain_init;        % fallback if toolbox call fails
end

% 5. Clean up: largest component, fill, light opening
bw_brain = keep_largest(bw_brain);
bw_brain = imfill(bw_brain, 'holes');
bw_brain = imopen(bw_brain, strel('disk', 2));

% Fallback: if mask almost empty, return original (stripping failed)
if nnz(bw_brain) < 0.02 * numel(bw_brain)
    Iss = I;
    return
end

% Apply mask
Iss = I;
Iss(~bw_brain) = 0;
end


function bw_out = keep_largest(bw_in)
% Keeps only the largest connected component of a binary mask.
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
