function batch_pre_processed_brisc25_clean_skullstrip(input_folder, output_folder)
% Portable entry point; input_folder contains class folders with official filenames.
arguments
 input_folder (1,:) char
 output_folder (1,:) char
end
if isfolder(output_folder) && ~isempty(dir(fullfile(output_folder, '*.xlsx')))
 error('Use an output directory without existing workbooks.');
end
if ~isfolder(output_folder), mkdir(output_folder); end
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'artifacts', 'third_party'));
% **********************************************************************
% Batch feature extraction + Excel exporter (with skull stripping)
%
% Postgraduate in Data Science for Biotechnology, ESB-UCP
% Catarina Bota  s-cbota@ucp.pt
%
% BRISC 2025 â€” Brain Tumor MRI Classification
%   Wavelet: bior1.1 | Levels: 8, 16, 32 | GLCM offsets: [2 0;1 0;0 1;0 2]
%
% Input:  classification_task/{glioma,meningioma,pituitary,no_tumor}/*.jpg
% Pre-processing: grayscale, resize 256x256, 3x3 median filter,
%                 morphological 2D skull stripping (toggle do_skull_strip)
% Output: features_BRISC_skullstripped.xlsx (sheets NL8, NL16, NL32)
% **********************************************************************

close all; clc;


%% *********************************************************************
%  PARAMETERS
% **********************************************************************

% '.' = current MATLAB folder
root_folder = input_folder;

% Group folder names must match exact names on disk
groups = {'glioma', 'meningioma', 'pituitary', 'no_tumor'};

% â”€â”€ SKULL STRIPPING TOGGLE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
% false -> no skull stripping  -> features_BRISC_preprocessed.xlsx
% true  -> morphological 2D skull stripping -> features_BRISC_skullstripped.xlsx
% Run the script twice (false, then true) to compare the two feature sets.
% Everything else stays identical, so any difference is due only to skull
% stripping.
do_skull_strip = true;

% Wavelet mother: bior1.1 -> computes averages and differences
wavelet_mae = 'bior1.1';

% NumLevels: number of GLCM quantization levels
% ->Controls GLCM size: NL x NL x 4
% 8 -> dense, less detail
% 16 -> balanced
% 32 -> more detail, sparser
num_levels = [8, 16, 32];

% Offsets: 4 adjacency directions for GLCM
offsets = [2 0; 1 0;0 1;0 2];

% Target image size
img_size = [256 256];

% Names of the 22 Haralick features returned by GLCM_Features
feature_names = {'autoc','contr','corrm','corrp','cprom','cshad','dissi', ...
                 'energ','entro','homom','homop','maxpr','sosvh','savgh', ...
                 'svarh','senth','dvarh','denth','inf1h','inf2h','indnc', ...
                 'idmnc'};

% The 9 origins where GLCM is calculated
subband_names = {'img','LL1','LH1','HL1','HH1','LL2','LH2','HL2','HH2'};

% Output Excel file name depends on the skull stripping toggle
if do_skull_strip
    output_excel = 'features_BRISC_skullstripped.xlsx';
else
    output_excel = 'features_BRISC_preprocessed.xlsx';
end


%% ********************************************************************
%  BUILD COLUMN NAMES (9 x 22 = 198 feature columns)
% *********************************************************************

feat_col_names = {};
for s = 1:length(subband_names)
    for f = 1:length(feature_names)
        feat_col_names{end+1} = [subband_names{s} '_' feature_names{f}];
    end
end
n_feat_cols = length(feat_col_names);  % 198


%% *********************************************************************
%  IMAGE INVENTORY
% *********************************************************************

fprintf('Inventorying images...\n');
if do_skull_strip
    fprintf('Skull stripping: ON\n');
else
    fprintf('Skull stripping: OFF\n');
end

image_list = struct('filepath', {}, 'filename', {}, 'group', {});

for g = 1:length(groups)
    group  = groups{g};
    folder = fullfile(root_folder, group);

    if ~isfolder(folder)
        warning('Folder not found: %s -> skipping.', folder);
        continue
    end

    files = dir(fullfile(folder, '*.jpg'));
    fprintf('  %s: %d images\n', group, length(files));

    for k = 1:length(files)
        image_list(end+1).filepath = fullfile(folder, files(k).name);
        image_list(end).filename   = files(k).name;
        image_list(end).group      = group;
    end
end

n_images = length(image_list);
fprintf('Total: %d images\n\n', n_images);

if n_images == 0
    error('No images found. Check root_folder and folder names.');
end


%% *********************************************************************
%  OUTER LOOP: one Excel sheet per NumLevels
% *********************************************************************

for nl_idx = 1:length(num_levels)

    NL = num_levels(nl_idx);
    fprintf('****************************\n');
    fprintf('NumLevels = %d\n', NL);
    fprintf('****************************\n');

    meta_data    = cell(n_images, 5);
    feature_data = zeros(n_images, n_feat_cols);


    %% ****************************************************************
    %  INNER LOOP:  process each image
    % ****************************************************************

    for i = 1:n_images

        entry    = image_list(i);
        filepath = entry.filepath;
        filename = entry.filename;
        group    = entry.group;

        fprintf('[%d/%d] %s\n', i, n_images, filename);


        % --- Parse file name ----------------------------------------
        tokens = regexp(filename, ...
            'brisc2025_(train|test)_(\d+)_(gl|me|pi|no)_(ax|co|sa)_t1', ...
            'tokens');

        if isempty(tokens)
            warning('Unexpected name: %s -> skipping.', filename);
            continue
        end

        t = tokens{1};
        split     = t{1};
        index     = t{2};
        tumor_code = t{3};
        view_code  = t{4};

        view_map = containers.Map({'ax','co','sa'},{'axial','coronal','sagittal'});
        view = view_map(view_code);

        meta_data(i,:) = {filename, split, group, view, index};


        % --- Load and pre-process -----------------------------------
        % Step 1: Read image
        I_raw = imread(filepath);

        % Step 2: RGB -> grayscale if needed
        if ndims(I_raw) == 3
            I_gray = rgb2gray(I_raw);
        else
            I_gray = I_raw;
        end

        % Step 3: Resize to 256x256
        I_resized = imresize(I_gray, img_size);

        % Step 4: 3x3 median filter (noise reduction)
        I_filt = medfilt2(I_resized, [3 3]);  % uint8

        % Step 5 (optional): morphological 2D skull stripping
        % Toggle with do_skull_strip in PARAMETERS. See skull_strip_2d below.
        if do_skull_strip
            I = skull_strip_2d(I_filt);
        else
            I = I_filt;
        end

        % Step 6: Normalize to double [0,1] for DWT
        In = mat2gray(I, [1 256]);


        % --- GLCM of direct image -----------------------------------
        GLCM_img = graycomatrix(I, 'NumLevels', NL, 'Offset', offsets);
        feat_img  = GLCM_Features(GLCM_img, 0);


        % --- DWT2 at 2 levels ---------------------------------------
        [ILL1, ILH1, IHL1, IHH1] = dwt2(In, wavelet_mae);
        [ILL2, ILH2, IHL2, IHH2] = dwt2(ILL1, wavelet_mae);


        % --- GLCM and features for each subband ---------------------
        GLCM_LL1 = graycomatrix(ILL1, 'NumLevels', NL, 'Offset', offsets);
        GLCM_LH1 = graycomatrix(ILH1, 'NumLevels', NL, 'Offset', offsets);
        GLCM_HL1 = graycomatrix(IHL1, 'NumLevels', NL, 'Offset', offsets);
        GLCM_HH1 = graycomatrix(IHH1, 'NumLevels', NL, 'Offset', offsets);
        GLCM_LL2 = graycomatrix(ILL2, 'NumLevels', NL, 'Offset', offsets);
        GLCM_LH2 = graycomatrix(ILH2, 'NumLevels', NL, 'Offset', offsets);
        GLCM_HL2 = graycomatrix(IHL2, 'NumLevels', NL, 'Offset', offsets);
        GLCM_HH2 = graycomatrix(IHH2, 'NumLevels', NL, 'Offset', offsets);

        feat_LL1 = GLCM_Features(GLCM_LL1, 0);
        feat_LH1 = GLCM_Features(GLCM_LH1, 0);
        feat_HL1 = GLCM_Features(GLCM_HL1, 0);
        feat_HH1 = GLCM_Features(GLCM_HH1, 0);
        feat_LL2 = GLCM_Features(GLCM_LL2, 0);
        feat_LH2 = GLCM_Features(GLCM_LH2, 0);
        feat_HL2 = GLCM_Features(GLCM_HL2, 0);
        feat_HH2 = GLCM_Features(GLCM_HH2, 0);


        % --- Build 198-feature row vector ---------------------------
        feat_row = extract_feat_vector(feat_img,  feature_names);
        feat_row = [feat_row, extract_feat_vector(feat_LL1, feature_names)];
        feat_row = [feat_row, extract_feat_vector(feat_LH1, feature_names)];
        feat_row = [feat_row, extract_feat_vector(feat_HL1, feature_names)];
        feat_row = [feat_row, extract_feat_vector(feat_HH1, feature_names)];
        feat_row = [feat_row, extract_feat_vector(feat_LL2, feature_names)];
        feat_row = [feat_row, extract_feat_vector(feat_LH2, feature_names)];
        feat_row = [feat_row, extract_feat_vector(feat_HL2, feature_names)];
        feat_row = [feat_row, extract_feat_vector(feat_HH2, feature_names)];

        feature_data(i, :) = feat_row;

    end % end inner loop


    %% ****************************************************************
    %  BUILD PER-IMAGE TABLE
    % ****************************************************************

    T_meta = cell2table(meta_data, ...
        'VariableNames', {'filename','split','class','view','index'});
    T_feat = array2table(feature_data, ...
        'VariableNames', feat_col_names);
    T_all  = [T_meta, T_feat];

    fprintf('\nNL=%d: %d images x %d columns\n', ...
        NL, height(T_all), width(T_all));
    for g = 1:length(groups)
        fprintf('  %-4s: %d images\n', groups{g}, ...
            sum(strcmp(T_all.class, groups{g})));
    end


    %% ****************************************************************
    % 1 ROW PER IMAGE
    % ****************************************************************

    T_out = T_all;
    fprintf('Images to export: %d\n', height(T_out));


    %% ***************************************************************
    %  WRITE SHEET TO EXCEL
    % ***************************************************************

    sheet_name = sprintf('NL%d', NL);
    writetable(T_out, fullfile(output_folder, output_excel), 'Sheet', sheet_name);

    fprintf('Sheet "%s" written to %s\n', sheet_name, output_excel);
    fprintf('  Rows    : %d images\n', height(T_out));
    fprintf('  Columns : %d (metadata + 198 features)\n\n', width(T_out));

end % end outer loop


fprintf('******************************************\n');
fprintf('Done! Output file: %s\n', output_excel);
fprintf('  Each sheet: 1 row per image, 198 features\n');
fprintf('******************************************\n');


%% *********************************************************************
%  AUXILIARY FUNCTION extract_feat_vector
% *********************************************************************

end

function v = extract_feat_vector(feat_struct, feat_names)
    n = length(feat_names);
    v = zeros(1, n);
    for k = 1:n
        vals = feat_struct.(feat_names{k});  % 1x4 vector
        v(k) = mean(vals);                   % mean across 4 directions
    end
end


%% *********************************************************************
%  FUNCTION skull_strip_2d
%  Simple morphological 2D skull stripping for T1 MRI slices.
%
%  Pipeline:
%   1. Otsu threshold -> binary mask (tissue vs dark background)
%   2. Fill holes, keep largest connected component (whole head)
%   3. Strong erosion (disk 18) to break the skull-brain link and peel
%      the outer skull/scalp
%   4. Keep the largest component again (now the brain)
%   5. Dilate back partially (disk 6, less than erosion) to recover the
%      brain boundary without re-including the skull
%   6. Fill + open to smooth, then apply mask to the image
%
%  LIMITATION: in T1 images brain and skull often touch and share similar
%  intensities, so Otsu may merge them. Aggressive erosion can also remove
%  genuine brain tissue. This method is therefore UNRELIABLE on some slices
%  (may leave skull or remove brain). It is included only to empirically
%  compare WITH vs WITHOUT skull stripping and to justify, with data, why a
%  simple morphological method is not adopted. It is NOT a clinical-grade
%  replacement for SPM12 or FSL BET. Inspect masked images and report
%  representative good/bad cases.
%
%  Input  : I  (uint8 grayscale, resized + denoised)
%  Output : Iss (uint8, non-brain pixels set to 0)
% *********************************************************************
function Iss = skull_strip_2d(I)

    Id = mat2gray(I);                 % double [0,1]

    % 1. Otsu threshold
    level = graythresh(Id);
    bw = imbinarize(Id, level);

    % 2. Fill holes, keep largest component (whole head)
    bw = imfill(bw, 'holes');
    bw = keep_largest(bw);

    % 3. Strong erosion to break skull-brain connection
    bw_eroded = imerode(bw, strel('disk', 18));

    % 4. Keep largest component (brain) after erosion
    bw_brain = keep_largest(bw_eroded);

    % 5. Dilate back partially (less than erosion radius)
    bw_brain = imdilate(bw_brain, strel('disk', 6));

    % 6. Fill and smooth the mask
    bw_brain = imfill(bw_brain, 'holes');
    bw_brain = imopen(bw_brain, strel('disk', 2));

    % Fallback: if mask almost empty (stripping failed), keep original
    if nnz(bw_brain) < 0.02 * numel(bw_brain)
        Iss = I;
        return
    end

    Iss = I;
    Iss(~bw_brain) = 0;
end


%% *********************************************************************
%  FUNCTION keep_largest
%  Keeps only the largest connected component of a binary mask.
% *********************************************************************
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
