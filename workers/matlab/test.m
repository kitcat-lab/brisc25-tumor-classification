%File: VISUAL TEST OF SKULL STRIPPING
%
% Shows 2 example images per class (glioma, meningioma, pituitary,
% no_tumor) with Original vs Skull-stripped side by side.
%
% Requires:
%   - skull_strip_2d.m in the same folder (and keep_largest inside it)
%   - the classification_task/all folder structure with class subfolders
%
% Purpose: visually inspect whether morphological skull stripping works
% on BRISC T1 slices. Use the figures to pick good/bad cases for the
% article and to justify the WITH vs WITHOUT comparison.
% *********************************************************************
 
clear; close all; clc;
 
root_folder = '.';                  % folder containing the class subfolders
groups   = {'glioma','meningioma','pituitary','no_tumor'};
n_per_class = 2;                    % examples per class
img_size = [256 256];
 
% Figure: 4 classes x 2 examples = 8 rows, 2 columns (orig | stripped)
figure('Name','Skull stripping comparison','Color','w', ...
       'Position',[100 100 600 1400]);
 
plot_idx = 1;
total_rows = length(groups) * n_per_class;
 
for g = 1:length(groups)
    group  = groups{g};
    folder = fullfile(root_folder, group);
 
    if ~isfolder(folder)
        warning('Folder not found: %s -> skipping.', folder);
        continue
    end
 
    files = dir(fullfile(folder, '*.jpg'));
    if isempty(files)
        warning('No images in %s -> skipping.', folder);
        continue
    end
 
    % Pick n_per_class images spread across the folder (not just the first)
    n_avail = length(files);
    pick = unique(round(linspace(1, n_avail, n_per_class)));
 
    for j = 1:length(pick)
        fname = files(pick(j)).name;
        I_raw = imread(fullfile(folder, fname));
 
        % Same pre-processing as the extraction script
        if ndims(I_raw) == 3
            I_gray = rgb2gray(I_raw);
        else
            I_gray = I_raw;
        end
        I = medfilt2(imresize(I_gray, img_size), [3 3]);
 
        % Skull stripping
        Iss = skull_strip_2d(I);
 
        % --- Original ---
        subplot(total_rows, 2, plot_idx);
        imshow(I);
        title(sprintf('%s (orig)', group), 'Interpreter','none', 'FontSize',8);
        plot_idx = plot_idx + 1;
 
        % --- Skull stripped ---
        subplot(total_rows, 2, plot_idx);
        imshow(Iss);
        % Percentage of pixels kept (rough indicator of how much was removed)
        pct_kept = 100 * nnz(Iss) / numel(Iss);
        title(sprintf('stripped (%.0f%% kept)', pct_kept), 'FontSize',8);
        plot_idx = plot_idx + 1;
    end
end
 
sgtitle('Original vs Skull-stripped - 2 examples per class', ...
        'FontWeight','bold');
 
% Optional: save the figure for the article
% exportgraphics(gcf, 'skull_strip_examples.png', 'Resolution', 200);
 
fprintf('Done. Inspect the figure.\n');
fprintf('Tip: uncomment the exportgraphics line to save it as PNG.\n');
 