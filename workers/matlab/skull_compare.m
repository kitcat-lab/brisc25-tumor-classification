% **********************************************************************
% File: COMPARE SKULL STRIPPING METHODS -> A4 PNG + PDF
%
% Shows, for random examples (1 per class), three columns:
%   Original | v1 (morphological) | v2 (multilevel + active contours)
% so you can judge which method works better.
%
% Requires in the same folder:
%   - skull_strip_2d.m   (v1, morphological)
%   - skull_strip_v2.m   (v2, multilevel + active contours)
% and the class subfolders glioma/ meningioma/ pituitary/ no_tumor/
% *********************************************************************

clear; close all; clc;

root_folder  = '.';
groups       = {'glioma','meningioma','pituitary','no_tumor'};
class_labels = {'Glioma','Meningioma','Pituitary','No Tumor'};
img_size     = [256 256];
rng(42);                    % reproducible; comment out for new picks

out_png = 'skull_strip_comparison.png';
out_pdf = 'skull_strip_comparison.pdf';

% A4 landscape
fig = figure('Color','w','Units','centimeters','Position',[1 1 29.7 21.0]);

% rows = 4 classes, cols = 3 (orig | v1 | v2)
tl = tiledlayout(length(groups), 3, 'TileSpacing','compact','Padding','compact');
title(tl, 'Skull Stripping Comparison: Original vs v1 (morphological) vs v2 (multilevel + active contours)', ...
    'FontWeight','bold','FontSize',12,'Color','k');

for g = 1:length(groups)
    folder = fullfile(root_folder, groups{g});
    files  = dir(fullfile(folder,'*.jpg'));
    if isempty(files)
        warning('No images in %s', folder);
        for c=1:3, nexttile; axis off; end
        continue
    end

    % one random image for this class
    idx = randi(numel(files));
    I_raw = imread(fullfile(folder, files(idx).name));
    if ndims(I_raw)==3, I_gray = rgb2gray(I_raw); else, I_gray = I_raw; end
    I = medfilt2(imresize(I_gray, img_size), [3 3]);

    % two methods
    I_v1 = skull_strip_2d(I);
    I_v2 = skull_strip_v2(I);
    p1 = 100*nnz(I_v1)/numel(I_v1);
    p2 = 100*nnz(I_v2)/numel(I_v2);

    % --- Original ---
    nexttile; imshow(I);
    ylabel(class_labels{g}, 'FontWeight','bold','FontSize',10, ...
        'Color','k','Rotation',90);
    if g==1, title('Original','FontSize',10,'Color','k'); end

    % --- v1 ---
    nexttile; imshow(I_v1);
    if g==1
        title('v1 morphological','FontSize',10,'Color','k');
    else
        title(sprintf('%.0f%% kept', p1),'FontSize',8,'Color','k');
    end
    if g==1
        text(0.5,-0.08,sprintf('%.0f%% kept',p1),'Units','normalized', ...
            'HorizontalAlignment','center','FontSize',8,'Color','k');
    end

    % --- v2 ---
    nexttile; imshow(I_v2);
    if g==1
        title('v2 active contours','FontSize',10,'Color','k');
    else
        title(sprintf('%.0f%% kept', p2),'FontSize',8,'Color','k');
    end
    if g==1
        text(0.5,-0.08,sprintf('%.0f%% kept',p2),'Units','normalized', ...
            'HorizontalAlignment','center','FontSize',8,'Color','k');
    end
end

% Export A4 landscape
exportgraphics(fig, out_png, 'Resolution', 200);
set(fig,'PaperType','A4','PaperOrientation','landscape', ...
    'PaperUnits','normalized','PaperPosition',[0 0 1 1]);
exportgraphics(fig, out_pdf, 'ContentType','vector');

fprintf('Saved:\n  %s\n  %s\n', out_png, out_pdf);