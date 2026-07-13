% **********************************************************************
% File: RANDOM SKULL STRIPPING EXAMPLES -> A4 PNG + PDF
%
% Picks random example images (balanced across the 4 classes) and shows
% Original vs Skull-stripped, laid out to fit a single A4 page (landscape).
% Exports to PNG and PDF for the article. All text in black.
%
% Layout (landscape A4):
%   4 columns (one per class) x 4 rows
%   Each class column shows 2 examples, each as an Original|Stripped pair
%   stacked vertically -> 2 examples x 2 images = 4 rows per column.
%
% Requires:
%   - skull_strip_2d.m in the same folder (with keep_largest inside it)
%   - class subfolders: glioma/ meningioma/ pituitary/ no_tumor/
% *********************************************************************

clear; close all; clc;

% ── Settings ─────────────────────────────────────────────────────────────
root_folder = '.';
groups       = {'glioma','meningioma','pituitary','no_tumor'};
class_labels = {'Glioma','Meningioma','Pituitary','No Tumor'};
n_per_class  = 2;            % random examples per class
img_size     = [256 256];
rng(42);                    % reproducible selection (comment out for new ones)

out_png = 'skull_strip_examples.png';
out_pdf = 'skull_strip_examples.pdf';

% ── Figure sized as A4 landscape (297 x 210 mm) ──────────────────────────
fig = figure('Name','Skull stripping - random examples', ...
             'Color','w', 'Units','centimeters', ...
             'Position',[1 1 29.7 21.0]);

% Grid: rows = n_per_class * 2 (orig+stripped), cols = number of classes
n_rows = n_per_class * 2;          % 4 rows
n_cols = length(groups);           % 4 columns

tl = tiledlayout(n_rows, n_cols, 'TileSpacing','compact', 'Padding','compact');
title(tl, 'Original vs Skull-Stripped MRI (random examples per class)', ...
      'FontWeight','bold', 'FontSize',13, 'Color','k');

% Pre-load random picks per class so we can fill the grid column by column
picks = cell(1, n_cols);
file_lists = cell(1, n_cols);
for g = 1:n_cols
    folder = fullfile(root_folder, groups{g});
    files  = dir(fullfile(folder, '*.jpg'));
    file_lists{g} = files;
    if isempty(files)
        warning('No images in %s.', folder);
        picks{g} = [];
    else
        k = min(n_per_class, numel(files));
        picks{g} = randperm(numel(files), k);
    end
end

% Fill the grid: for each example (1..n_per_class), one row for Original
% and the next row for Stripped, across all class columns.
for ex = 1:n_per_class
    % --- Row for ORIGINAL images of this example index ---
    for g = 1:n_cols
        nexttile;
        files = file_lists{g};
        if isempty(files) || ex > numel(picks{g})
            axis off; continue
        end
        fname = files(picks{g}(ex)).name;
        I_raw = imread(fullfile(root_folder, groups{g}, fname));
        if ndims(I_raw) == 3, I_gray = rgb2gray(I_raw); else, I_gray = I_raw; end
        I = medfilt2(imresize(I_gray, img_size), [3 3]);

        imshow(I);
        % Column header (class name) only on the very top row
        if ex == 1
            title(class_labels{g}, 'FontSize',10, 'FontWeight','bold', 'Color','k');
        end
        % Row label on the leftmost column
        if g == 1
            ylabel(sprintf('Ex.%d  Original', ex), 'FontSize',8, ...
                   'FontWeight','bold', 'Color','k', 'Rotation',90);
        end
    end

    % --- Row for SKULL-STRIPPED images of this example index ---
    for g = 1:n_cols
        nexttile;
        files = file_lists{g};
        if isempty(files) || ex > numel(picks{g})
            axis off; continue
        end
        fname = files(picks{g}(ex)).name;
        I_raw = imread(fullfile(root_folder, groups{g}, fname));
        if ndims(I_raw) == 3, I_gray = rgb2gray(I_raw); else, I_gray = I_raw; end
        I = medfilt2(imresize(I_gray, img_size), [3 3]);
        Iss = skull_strip_2d(I);
        pct_kept = 100 * nnz(Iss) / numel(Iss);

        imshow(Iss);
        title(sprintf('%.0f%% kept', pct_kept), 'FontSize',8, 'Color','k');
        if g == 1
            ylabel(sprintf('Ex.%d  Stripped', ex), 'FontSize',8, ...
                   'FontWeight','bold', 'Color','k', 'Rotation',90);
        end
    end
end

% ── Export as A4 landscape ────────────────────────────────────────────────
exportgraphics(fig, out_png, 'Resolution', 200);

% For the PDF, set the paper to A4 landscape so it fills one page
set(fig, 'PaperType','A4', 'PaperOrientation','landscape', ...
         'PaperUnits','normalized', 'PaperPosition',[0 0 1 1]);
exportgraphics(fig, out_pdf, 'ContentType','vector');

fprintf('Saved (A4 landscape):\n  %s\n  %s\n', out_png, out_pdf);