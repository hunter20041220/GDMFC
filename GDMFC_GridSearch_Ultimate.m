%==========================================================================
% GDMFC Ultimate Grid Search Script (Full Metrics & Logging)
%==========================================================================
% Author: Assistant (Simulating IEEE Fellow Professor)
% Environment: MATLAB R2024b
% Description: 
%   Performs exhaustive grid search over 15,000 parameter combinations.
%   Metrics recorded: ACC, NMI, Purity.
%   Search Space:
%     1. Preprocess: [1,2,3,4,5] (5)
%     2. Lambda1:    [1e-4, 1e-3, 1e-2, 1e-1, 1] (5)
%     3. Lambda2:    [1e-4, 1e-3, 1e-2, 1e-1, 1] (5)
%     4. Gamma:      [1.1, 1.3, 1.5, 2, 5, 10] (6)
%     5. Graph K:    [3, 5, 7, 9, 11] (5)
%     6. Tol:        [1e-3, 1e-4, 1e-5, 1e-6] (4)
%==========================================================================

clear; clc; close all;

%% 1. Setup & Data Loading
fprintf('============================================================\n');
fprintf('Step 0: Initializing and Loading Data...\n');
fprintf('============================================================\n');

addpath(genpath('./')); 
addpath(genpath('../DMF_MVC')); 

% --- Load ORL Dataset ---
dataPath = '../../dataset/orl'; 
numSubjects = 40;
imagesPerSubject = 10;
imageHeight = 112;
imageWidth = 92;
numSamples = numSubjects * imagesPerSubject;

if ~isfolder(dataPath)
    error('Error: Dataset path "%s" not found.', dataPath);
end

allImages = zeros(numSamples, imageHeight * imageWidth);
labels = zeros(numSamples, 1);
sampleIdx = 1;

for subjID = 1:numSubjects
    subjFolder = fullfile(dataPath, sprintf('s%d', subjID));
    for imgID = 1:imagesPerSubject
        imgPath = fullfile(subjFolder, sprintf('%d.pgm', imgID));
        if ~isfile(imgPath), continue; end
        img = imread(imgPath);
        allImages(sampleIdx, :) = double(img(:)');
        labels(sampleIdx) = subjID;
        sampleIdx = sampleIdx + 1;
    end
end
numSamples = sampleIdx - 1;
labels = labels(1:numSamples);
allImages = allImages(1:numSamples, :);

% --- Construct Features ---
X1 = zeros(numSamples, (imageHeight/2)*(imageWidth/2));
for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    imgDown = imresize(img, [imageHeight/2, imageWidth/2], 'bilinear');
    X1(i, :) = imgDown(:)';
end

X2 = zeros(numSamples, floor(imageHeight/8) * floor(imageWidth/8) * 4);
blockSize = 8;
numBlocksH = floor(imageHeight / blockSize);
numBlocksW = floor(imageWidth / blockSize);
for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    featIdx = 1;
    for bh = 1:numBlocksH
        for bw = 1:numBlocksW
            rS = (bh-1)*blockSize+1; rE = min(bh*blockSize, imageHeight);
            cS = (bw-1)*blockSize+1; cE = min(bw*blockSize, imageWidth);
            blk = img(rS:rE, cS:cE);
            X2(i, featIdx:featIdx+3) = [mean(blk(:)), std(blk(:)), min(blk(:)), max(blk(:))];
            featIdx = featIdx + 4;
        end
    end
end

X_RAW_BASE = cell(1, 2);
X_RAW_BASE{1} = X1;
X_RAW_BASE{2} = X2;
y = labels;
numCluster = numSubjects;

fprintf('Data Loaded. Pre-calculating normalized data versions...\n');

%% 2. Pre-calculate Data Versions
data_versions = cell(1, 5);
for mode = 1:5
    X_temp = data_guiyi_choos(X_RAW_BASE, mode);
    for v = 1:length(X_temp)
        row_norms = sqrt(sum(X_temp{v}.^2, 2));
        row_norms(row_norms == 0) = 1;
        X_temp{v} = bsxfun(@rdivide, X_temp{v}, row_norms);
    end
    data_versions{mode} = X_temp;
end
fprintf('Pre-calculation complete.\n\n');

%% 3. Define Search Grid
param_ranges = struct();
param_ranges.preprocess = [1, 2, 3, 4, 5];               % 5
param_ranges.lambda1    = [0.0001, 0.001, 0.01, 0.1, 1]; % 5
param_ranges.lambda2    = [0.0001, 0.001, 0.01, 0.1, 1]; % 5
param_ranges.gamma      = [1.1, 1.3, 1.5, 2, 5, 10];     % 6
param_ranges.graph_k    = [3, 5, 7, 9, 11];              % 5
param_ranges.tol        = [1e-3, 1e-4, 1e-5, 1e-6];      % 4

fixed_beta = 115;
fixed_layers = [100, 50];

[P, L1, L2, G, K, T] = ndgrid(...
    param_ranges.preprocess, ...
    param_ranges.lambda1, ...
    param_ranges.lambda2, ...
    param_ranges.gamma, ...
    param_ranges.graph_k, ...
    param_ranges.tol);

combinations = [P(:), L1(:), L2(:), G(:), K(:), T(:)];
total_runs = size(combinations, 1);

%% 4. Initialize CSV Logging (Real-time Excel Storage)
logFileName = 'GDMFC_RealTime_Log.csv';

% If file doesn't exist, write headers. If it exists, we append (safety).
if ~isfile(logFileName)
    fid = fopen(logFileName, 'w');
    fprintf(fid, 'RunID,Mode,Lambda1,Lambda2,Gamma,GraphK,Tol,ACC,NMI,Purity,TimeSec,Status\n');
    fclose(fid);
else
    fprintf('Warning: %s already exists. Appending to it.\n', logFileName);
end

fprintf('============================================================\n');
fprintf('STARTING VERBOSE GRID SEARCH\n');
fprintf('Total Runs: %d\n', total_runs);
fprintf('Results will be saved to: %s (Real-time)\n', logFileName);
fprintf('============================================================\n');
pause(2);

%% 5. Main Loop (Verbose)
% IMPORTANT: Cannot use 'parfor' if we want ordered printing and sequential file writing.
% Using standard 'for' loop to allow visualization.

total_timer = tic;

for idx = 1:total_runs
    % Get Params
    p_mode = combinations(idx, 1);
    l1     = combinations(idx, 2);
    l2     = combinations(idx, 3);
    gam    = combinations(idx, 4);
    gk     = combinations(idx, 5);
    tol_val= combinations(idx, 6);
    
    % --- VISUALIZATION: Print Header BEFORE Run ---
    fprintf('\n');
    fprintf('>>>>> [Run %d / %d] <<<<<\n', idx, total_runs);
    fprintf('Params: Mode=%d | L1=%.4f | L2=%.4f | Gam=%.1f | K=%d | Tol=%.1e\n', ...
            p_mode, l1, l2, gam, gk, tol_val);
    fprintf('------------------------------------------------------------\n');
    
    t_start = tic;
    acc = 0; nmi = 0; purity = 0; status = "Success";
    
    try
        % 1. Data Selection
        X_curr = data_versions{p_mode};
        
        % 2. Options
        opts = struct();
        opts.lambda1 = l1; 
        opts.lambda2 = l2;
        opts.beta    = fixed_beta; 
        opts.gamma   = gam;
        opts.graph_k = gk; 
        opts.maxIter = 100; 
        opts.tol     = tol_val;
        
        % 3. Run GDMFC (Output is NOT suppressed now)
        % This will show the "Iter X: obj=..." lines directly in command window
        [H, ~, ~, ~] = GDMFC(X_curr, numCluster, fixed_layers, opts);
        
        % 4. Evaluation
        S = H * H'; 
        S = (S + S')/2; 
        S = max(S,0);
        rng(42); 
        predict_labels = SpectralClustering(S, numCluster);
        
        res = bestMap(y, predict_labels);
        acc = length(find(y == res)) / length(y);
        nmi = MutualInfo(y, predict_labels);
        purity = compute_purity(y, predict_labels);
        
        % Visual Feedback for End of Run
        fprintf('<<<<< Result: ACC=%.2f%% | NMI=%.4f | Time=%.2fs\n', acc*100, nmi, toc(t_start));
        
    catch ME
        fprintf('!!!!! ERROR in Run %d: %s\n', idx, ME.message);
        status = "Failed";
    end
    
    t_cost = toc(t_start);
    
    % --- REAL-TIME LOGGING: Write to CSV immediately ---
    % Open, Append, Close (Safe against crash)
    fid = fopen(logFileName, 'a');
    fprintf(fid, '%d,%d,%.6f,%.6f,%.2f,%d,%.1e,%.6f,%.6f,%.6f,%.2f,%s\n', ...
            idx, p_mode, l1, l2, gam, gk, tol_val, acc, nmi, purity, t_cost, status);
    fclose(fid);
end

total_time_hr = toc(total_timer) / 3600;
fprintf('\n============================================================\n');
fprintf('ALL FINISHED in %.2f hours.\n', total_time_hr);
fprintf('Check %s for full results table.\n', logFileName);
fprintf('============================================================\n');