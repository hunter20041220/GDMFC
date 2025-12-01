function data = data_guiyi_choos(data, data_g)
%DATA_GUIYI_CHOOS Preprocess a cell-array of views according to mode
%
%   data = data_guiyi_choos(data, data_g)
%
% Modes (data_g):
%   1 - Min-max scale each feature to [0,1] (apply mapminmax to rows)
%   2 - Min-max scale when data is organized transposed (apply mapminmax to columns)
%   3 - Column-wise L2 normalization (each feature scaled by its L2 norm)
%   4 - Column-wise sum normalization (divide each column by its column sum)
%   5 - Global normalization (divide entire matrix by its total sum)
%
% Input:
%   data   - cell array, data{v} is n x d_v matrix for view v
%   data_g - integer mode selector
%
% Output:
%   data   - preprocessed cell array

if nargin < 2
    error('data_guiyi_choos requires two inputs: data and data_g');
end

for v = 1:length(data)
    X = data{v};
    switch data_g
        case 1
            % Min-max scale each feature to [0,1]
            X = mapminmax(X, 0, 1);
        case 2
            % If features are in rows, transpose before mapminmax
            X = mapminmax(X', 0, 1)';
        case 3
            % Column-wise L2 normalization (avoid divide-by-zero)
            norms = sqrt(sum(X.^2, 1));
            norms(norms == 0) = 1;
            X = X * diag(1./norms);
        case 4
            % Column-wise sum normalization
            colsum = sum(X, 1);
            colsum(colsum == 0) = 1;
            X = X ./ colsum;
        case 5
            % Global normalization by total sum
            total = sum(X(:));
            if total == 0
                total = 1;
            end
            X = X ./ total;
        otherwise
            error('Unknown data_g mode: %d', data_g);
    end
    data{v} = X;
end

end