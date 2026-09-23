function [meanValue, stdValue, residual] = time_average_stats(data)
%TIME_AVERAGE_STATS Compute the time-weighted mean, standard deviation, and residuals
%
% Inputs:
%   data(:,1) : time t
%   data(:,2) : data y
%
% Outputs:
%   meanValue : time-weighted mean
%   stdValue  : time-weighted population standard deviation
%   residual  : y - meanValue
%
% Example:
%   data = load('data.txt');
%   [meanValue, stdValue, residual] = time_average_stats(data);

    t = data(:,1);
    y = data(:,2);

    % Validate data
    assert(size(data,2) == 2, '输入数据必须是两列：[时间, 数据]');
    assert(length(t) >= 2, '至少需要两个数据点');
    assert(all(diff(t) > 0), '时间必须严格递增');

    % Total time span
    T = t(end) - t(1);

    % Time-weighted mean
    meanValue = trapz(t, y) / T;

    % Residual: instantaneous value minus the time mean
    residual = y - meanValue;

    % Time-weighted population standard deviation
    stdValue = sqrt(max(trapz(t, residual.^2) / T, 0));

end