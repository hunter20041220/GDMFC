% Test script: load ORL images and run preprocessing (data_guiyi_choos)
% Run this in MATLAB to verify preprocessing succeeds without errors.

clear; clc;
addpath(genpath('../DMF_MVC'));

dataPath = '../../dataset/orl';
numSubjects = 40;
imagesPerSubject = 10;
imageHeight = 112;
imageWidth = 92;
numSamples = numSubjects * imagesPerSubject;

allImages = zeros(numSamples, imageHeight * imageWidth);
labels = zeros(numSamples, 1);

sampleIdx = 1;
for subjID = 1:numSubjects
    subjFolder = fullfile(dataPath, sprintf('s%d', subjID));
    for imgID = 1:imagesPerSubject
        imgPath = fullfile(subjFolder, sprintf('%d.pgm', imgID));
        img = imread(imgPath);
        allImages(sampleIdx, :) = double(img(:)');
        labels(sampleIdx) = subjID;
        sampleIdx = sampleIdx + 1;
    end
end

% Construct simple views
X = cell(1,2);
% View1: downsampled pixels
downsampleFactor = 2;
newH = imageHeight / downsampleFactor;
newW = imageWidth / downsampleFactor;
X1 = zeros(numSamples, newH * newW);
for i = 1:numSamples
    img = reshape(allImages(i,:), imageHeight, imageWidth);
    imgDown = imresize(img, [newH, newW], 'bilinear');
    X1(i,:) = imgDown(:)';
end
X{1} = X1;
% View2: block stats
blockSize = 8;
numBlocksH = floor(imageHeight / blockSize);
numBlocksW = floor(imageWidth / blockSize);
lbpFeatDim = numBlocksH * numBlocksW * 4;
X2 = zeros(numSamples, lbpFeatDim);
for i = 1:numSamples
    img = reshape(allImages(i,:), imageHeight, imageWidth);
    featIdx = 1;
    for bh = 1:numBlocksH
        for bw = 1:numBlocksW
            rowStart = (bh-1)*blockSize + 1;
            rowEnd = min(bh*blockSize, imageHeight);
            colStart = (bw-1)*blockSize + 1;
            colEnd = min(bw*blockSize, imageWidth);
            block = img(rowStart:rowEnd, colStart:colEnd);
            X2(i, featIdx) = mean(block(:));
            X2(i, featIdx+1) = std(block(:));
            X2(i, featIdx+2) = min(block(:));
            X2(i, featIdx+3) = max(block(:));
            featIdx = featIdx + 4;
        end
    end
end
X{2} = X2;

% Run preprocessing
preprocess_mode = 3; % column-wise L2
Xp = data_guiyi_choos(X, preprocess_mode);

% Apply NormalizeFea
for v = 1:length(Xp)
    Xp{v} = NormalizeFea(Xp{v}, 0);
end

save('test_preprocess_ORL.mat', 'Xp', 'labels');
fprintf('Preprocessing test completed. Saved test_preprocess_ORL.mat\n');
