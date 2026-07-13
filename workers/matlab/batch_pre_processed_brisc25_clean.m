% **********************************************************************
% Batch feature extraction + Excel exporter
%
% Postgraduate in Data Science for Biotechnology, ESB-UCP
% Catarina Bota  s-cbota@ucp.pt
%
% BRISC 2025 — Brain Tumor MRI Classification
%   Wavelet: bior1.1 | Levels: 8, 16, 32 | GLCM offsets: [2 0;1 0;0 1;0 2]
%
% Input:  classification_task/{glioma,meningioma,pituitary,no_tumor}/*.jpg
% Pre-processing: grayscale, resize 256x256, 3x3 median filter
% Output: features_BRISC_pre_processed.xlsx (sheets NL8, NL16, NL32)
% **********************************************************************
 
clear all; close all; clc;
 
 
%% *********************************************************************
%  PARAMETERS 
% **********************************************************************
 
% '.' = current MATLAB folder
root_folder = '.';
 
% Group folder names must match exact names on disk
groups = {'glioma', 'meningioma', 'pituitary', 'no_tumor'};
 
% Wavelet mother: bior1.1 -> computes averages and differences
wavelet_mae = 'bior1.1';
 
% NumLevels: number of GLCM quantization levels 
% ->Controls GLCM size: NL x NL x 4
% 8 -> dense, less detail 
% 16 -> balanced
% 32 -> more detail, sparser
num_levels = [8, 16, 32];
 
% Offsets: 4 adjacency directions for GLCM
% [row, col] = how far to move to reach the neighbor
% [2 0] -> 2 pixels down ->(vertical, distance 2)
% [1 0] -> 1 pixel down  ->(vertical, distance 1)
% [0 1] -> 1 pixel right ->(horizontal, distance 1)
% [0 2] -> 2 pixels right ->(horizontal, distance 2)


% Result: GLCM of size NumLevels x NumLevels x 4
% Feature values are averaged across the 4 directions to produce
% one representative value per feature (rotation-invariant approach)

% Image 1 → DWT → GLCM → 198 features |
% Image 2 → DWT → GLCM → 198 features │
% ...                                 |-> groupsummary -> mean -> 1 row
%Image 24 → DWT → GLCM → 198 features |

% An approach maintaining the four values separately,
% as in [Oliveira et al., 2024], was considered but not adopted.
%
%% Feature values were averaged across the four offset directions to produce
% a single representative value per feature.
% It ensures the model is computationally efficient and focuses
% on the impact of gray-level quantization without the "noise" of 
% directional variability that could lead to overfitting on a 
% multi-plane dataset like BRISC 2025.


% Target image size: resize all images to this before processing
% -> ensures all GLCM and DWT operate on same spatial scale
% -> 256x256 is standard in brain tumor MRI classification literature
img_size = [256 256];

offsets = [2 0; 1 0;0 1;0 2];
 
% Names of the 22 Haralick features returned by GLCM_Features
% This order defines the column order in the output Excel
feature_names = {'autoc','contr','corrm','corrp','cprom','cshad','dissi', ...
                 'energ','entro','homom','homop','maxpr','sosvh','savgh', ...
                 'svarh','senth','dvarh','denth','inf1h','inf2h','indnc', ...
                 'idmnc'};
 
% The 9 origins where GLCM is calculated:
% -> img = original image (no DWT)
% Each image is independent (no patient aggregation)
% -> LL1..HH1 = 4 level-1 subbands (size N/2 x N/2)
% -> LL2..HH2 = 4 level-2 subbands (size N/4 x N/4)
subband_names = {'img','LL1','LH1','HL1','HH1','LL2','LH2','HL2','HH2'};
 
% Output Excel file name
output_excel = 'features_BRISC_preprocessed.xlsx';
 
 
%% ********************************************************************
%  BUILD COLUMN NAMES
%  For each subband (9) and feature (22): "LL1_energ", "HH2_entro", etc
%  These become the column headers in the Excel output
%  Total: 9 x 22 = 198 feature columns
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
%  Go through glioma/, meningioma/, pituitary/, no_tumor/ and build
%  a complete file list upfront (train + test together):
%  -> Knowing the total count allows pre-allocating result arrays at once,
%     which is faster than growing arrays inside the loop
%
%  Each entry in image_list has:
%  .filepath = full path (used by imread)
%  .filename = file name only (used by regexp)
%  .group = "glioma", "meningioma", "pituitary" or "no_tumor"
% *********************************************************************
 
fprintf('Inventorying images...\n');
 
image_list = struct('filepath', {}, 'filename', {}, 'group', {});
 
for g = 1:length(groups)
    group  = groups{g};
    folder = fullfile(root_folder, group);
 
    % isfolder checks if the folder exists; ~ means NOT 
    if ~isfolder(folder)
        warning('Folder not found: %s -> skipping.', folder);
        continue
    end
 
    % dir lists all .jpg files (* is a wildcard)
    files = dir(fullfile(folder, '*.jpg'));
    fprintf('  %s: %d images\n', group, length(files));
 
    % Add each file to the global list:
    % -> use end+1 then fill end (same new entry) to avoid
    % concatenation issues with temporary arrays
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
%
%  -> The full pipeline runs 3 times (NL = 8, 16, 32)
%  -> Each run writes one sheet to features_BRISC_preprocessed.xlsx
% *********************************************************************
 
for nl_idx = 1:length(num_levels)
 
    NL = num_levels(nl_idx);
    fprintf('****************************\n');
    fprintf('NumLevels = %d\n', NL);
    fprintf('****************************\n');
 
    % Pre-allocate result arrays for all images at once
    % cell(n,m) -> n x m cell array of [] -> for string metadata
    % zeros(n,m) -> n x m numeric matrix -> for feature values
    meta_data    = cell(n_images, 5);
    feature_data = zeros(n_images, n_feat_cols);
 
 
    %% ********************************************************************
    %  INNER LOOP:  process each image
    % *********************************************************************
 
    for i = 1:n_images
 
        entry    = image_list(i);
        filepath = entry.filepath;
        filename = entry.filename;
        group    = entry.group;
 
        fprintf('[%d/%d] %s\n', i, n_images, filename);
 
 
        % --- Parse file name ----------------------------------------
        %-> regexp extracts capture groups () from the file name
        %-> \d+ = one or more digits
        %
        % Example: 'brisc2025_train_00010_gl_ax_t1.jpg'
        %  t{1} = "train"  -> split (train or test)
        %  t{2} = "00010" -> index
        %  t{3} = "gl"    -> tumor code (gl/me/pi/nt)
        %  t{4} = "ax"    -> view code (ax/co/sa)
        %  t{3} = "axial" ->anatomical view
        %  t{4} = "3"-> slice number
        tokens = regexp(filename, ...
            'brisc2025_(train|test)_(\d+)_(gl|me|pi|no)_(ax|co|sa)_t1', ...
            'tokens');
 
        if isempty(tokens)
            warning('Unexpected name: %s -> skipping.', filename);
            continue
        end
 
        t = tokens{1};
        split     = t{1};   % train or test
        index     = t{2};   % image index
        tumor_code = t{3};  % gl | me | pi | no
        view_code  = t{4};  % ax | co | sa

        % Map short codes to full names for readability
        view_map = containers.Map({'ax','co','sa'},{'axial','coronal','sagittal'});
        view = view_map(view_code);

        meta_data(i,:) = {filename, split, group, view, index};
 
 
        % --- Load and pre-process ------------------------------------------
        % Step 1: Read image
        I_raw = imread(filepath);

        % Step 2: Convert RGB to grayscale if needed
        % BRISC JPGs may be stored as RGB even though they are grayscale MRI
        % rgb2gray uses luminosity weights: 0.2989R + 0.5870G + 0.1140B
        if ndims(I_raw) == 3
            I_gray = rgb2gray(I_raw);
        else
            I_gray = I_raw;
        end

        % Step 3: Resize to 256x256
        % Ensures all images have the same spatial resolution before feature
        % extraction. Differences in image size would make GLCM and DWT
        % features incomparable across images.
        % Reference: standard practice in brain tumor MRI classification
        I_resized = imresize(I_gray, img_size);

        % Step 4: Noise reduction — 3x3 median filter
        % Median filter preserves edges (unlike Gaussian) while removing
        % salt-and-pepper noise. Standard choice for MRI pre-processing.
        % Reference: Netshamutshedzi et al. (2025), Frontiers in AI
        I = medfilt2(I_resized, [3 3]);  % uint8, used for GLCM direct image

        % Step 5: Normalize to double [0,1] for DWT
        % bior1.1 needs double input - uint8 causes coefficient collapse
        In = mat2gray(I, [1 256]);
 
 
        % --- GLCM of direct image 

        % -> graycomatrix quantizes I into NL levels and counts co-occurrences 
        % in the 4 offset directions
        % -> Returns NL x NL x 4 matrix
        % -> GLCM_Features computes 22 Haralick values -> struct
        % -> Argument 0 = pairs=0: symmetry included with opposite offsets
        GLCM_img = graycomatrix(I, 'NumLevels', NL, 'Offset', offsets);
        feat_img  = GLCM_Features(GLCM_img, 0);
 
 
        % --- DWT2 at 2 levels 
        % Level 1: decomposes In into 4 subbands (size N/2 x N/2)
        %  ILL1 = approximation (low freq.) -> global structure
        %  ILH1 = horizontal details -> horizontal edges
        %  IHL1 = vertical details  -> vertical edges
        %  IHH1 = diagonal details  -> corners, noise
        [ILL1, ILH1, IHL1, IHH1] = dwt2(In, wavelet_mae);
 
        % Level 2: decomposes ILL1 again (size N/4 x N/4)
        [ILL2, ILH2, IHL2, IHH2] = dwt2(ILL1, wavelet_mae);
 
 
        % --- GLCM and features for each subband 
        % -> graycomatrix sets GrayLimits automatically from each subband's
        %  value range
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
 
 
        % --- Build 198-feature row vector 
        % extract_feat_vector (see function at bottom of file):
        %  -> iterates through the 22 struct fields in order
        %  -> averages the 4 offset values per feature into 1 number
        %  -> returns a 1x22 row vector
        % concatenate all 9 vectors: 9 x 22 = 198 values per image
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
 
 
    %% ********************************************************************
    %  BUILD PER-IMAGE TABLE
    % -> cell2table converts string cell array to table
    % -> array2table converts numeric matrix to table
    % -> Join side by side with []
    % *********************************************************************
 
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
 
    feat_vars = T_all.Properties.VariableNames(6:end);  % columns 1-5 are metadata
 
 
    %% ********************************************************************
    % 1 ROW PER IMAGE
    % *********************************************************************

    T_out = T_all;
    fprintf('Images to export: %d\n', height(T_out));
 
 
    %% *********************************************************************
    %  WRITE SHEET TO EXCEL
    %
    %  One sheet per NumLevels in a single Excel file
    %  Sheet name: "NL8", "NL16" or "NL32"
    %
    %  writetable with 'Sheet' writes to a named sheet
    %  -> If the sheet does not exist, MATLAB creates it automatically
    %  -> If the file does not exist, MATLAB creates it too
    % *********************************************************************
 
    sheet_name = sprintf('NL%d', NL);
    writetable(T_out, output_excel, 'Sheet', sheet_name);

    fprintf('Sheet "%s" written to %s\n', sheet_name, output_excel);
    fprintf('  Rows    : %d images\n', height(T_out));
    fprintf('  Columns : %d (metadata + 198 features)\n\n', width(T_out));
 
end % end outer loop
 
 
fprintf('******************************************\n');
fprintf('Done! Output file: %s\n', output_excel);
fprintf('  Sheet NL8  -> NumLevels = 8\n');
fprintf('  Sheet NL16 -> NumLevels = 16\n');
fprintf('  Sheet NL32 -> NumLevels = 32\n');
fprintf('  Each sheet: 1 row per patient, 198 features\n');
fprintf('******************************************\n');
 
 
%% *********************************************************************
%  AUXILIARY FUNCTION extract_feat_vector
%
%  Converts a GLCM_Features struct into a 1x22 row vector.
%
%  INPUT:
%    feat_struct -> struct with 22 fields (output of GLCM_Features)
%                   each field is a 1x4 vector (one value per offset)
%    feat_names  -> cell array of field names in desired order
%
%  OUTPUT:
%    v -> 1x22 row vector: one averaged value per feature
%
%  feat_struct.(feat_names{k}) is dynamic field access that allows
%  iterating over field names stored in a variable instead of writing each
%  field name hardcoded
%
%  mean(vals) averages the 4 direction values into one number,
%  making the feature direction-independent
% *********************************************************************
 
function v = extract_feat_vector(feat_struct, feat_names)
    n = length(feat_names);
    v = zeros(1, n);
    for k = 1:n
        vals = feat_struct.(feat_names{k});  % 1x4 vector
        v(k) = mean(vals);                   % mean across 4 directions
    end
end

%% *********************************************************************
% REFERENCES 
% Netshamutshedzi, N. et al. (2025). A systematic review of the hybrid
% machine learning models for brain tumour segmentation and detection.
% Frontiers in Artificial Intelligence.
%
%Alibabaei, S., Rahmani, M., Tahmasbi, M., Birgani, M. J. T., & Razmjoo, 
% S. (2023).Evaluating the gray level co-occurrence matrix-based texture 
% features of magnetic resonance images for glioblastoma multiform 
% patients' treatment response assessment. Journal of Medical Signals 
% & Sensors, 13(4), 261–271.
%
% Chen, H., Li, W., & Zhu, Y. (2021). Improved window adaptive gray level 
% co-occurrence matrix for extraction and analysis of texture 
% characteristics of pulmonary nodules. Computer Methods and Programs 
% in Biomedicine, 208, 106263.
%
% Oliveira, M. J., Ribeiro, P., & Rodrigues, P. M. (2024). Machine 
% learning-driven GLCM analysis of structural MRI for Alzheimer's 
% disease diagnosis. Bioengineering, 11(11), 1153.
%
%