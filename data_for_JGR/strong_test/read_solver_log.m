function T = read_solver_log(filename, maxLines)
%READ_SOLVER_LOG Read per-step timings and iteration counts from a solver log.
%   T = read_solver_log('output.txt');          % Read all lines
%   T = read_solver_log('output.txt', 100000);  % Read first 100000 lines
% maxLines counts all physical lines, including blank/header lines.
% Omitted, empty, or Inf means no line limit.
%   disp(T);
%   writetable(T, 'solver_statistics.csv');
%
% Step is the order in the log, not the simulation's physical step index.
% Iterations is the final printed iteration index (Iteration 0 is initial).
% For incomplete steps, Iterations and SolveTime_s are NaN; LastIteration
% retains the last observed index. Complete means the timing line was found,
% not that convergence was independently verified.
% Assumes one outer iteration sequence per assembly block.
% Save the input log as UTF-8 if Chinese labels are not recognized.

if nargin < 2 || isempty(maxLines)
    maxLines = Inf;
end
validateattributes(maxLines, {'numeric'}, {'scalar','real','positive'}, ...
    mfilename, 'maxLines', 2);
if ~isinf(maxLines) && maxLines ~= fix(maxLines)
    error('read_solver_log:InvalidLimit', 'maxLines must be a positive integer or Inf.');
end

fid = fopen(filename, 'rt', 'n', 'UTF-8');
if fid < 0
    error('read_solver_log:OpenFailed', 'Cannot open file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));

num = '[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?';
patAssembly = ['组装用时\s*[,，:：]*\s*(' num ')'];
patSolve = ['1\s*个时间步求解用时\s*[,，:：]*\s*(' num ')'];
patIteration = 'Iteration\s+(\d+)\s*[,，]';

assembly = zeros(0,1);
solveTime = zeros(0,1);
lastIteration = zeros(0,1);
complete = false(0,1);
n = 0;
lineCount = 0;
while lineCount < maxLines
    line = fgetl(fid);
    if ~ischar(line)
        break
    end
    lineCount = lineCount + 1;
    token = regexp(line, patAssembly, 'tokens', 'once');
    if ~isempty(token)
        n = n + 1;
        assembly(n,1) = str2double(regexprep(token{1}, '[dD]', 'e'));
        solveTime(n,1) = NaN;
        lastIteration(n,1) = NaN;
        complete(n,1) = false;
        continue
    end
    if n == 0 || complete(n)
        continue
    end
    token = regexp(line, patIteration, 'tokens', 'once');
    if ~isempty(token)
        k = str2double(token{1});
        if ~isnan(lastIteration(n)) && k < lastIteration(n)
            error('read_solver_log:IterationReset', ...
                'Iteration index reset in log block %d; multiple solves may be present.', n);
        end
        lastIteration(n) = k;
        continue
    end
    token = regexp(line, patSolve, 'tokens', 'once');
    if ~isempty(token)
        solveTime(n) = str2double(regexprep(token{1}, '[dD]', 'e'));
        complete(n) = true;
    end
end
iterations = lastIteration;
iterations(~complete) = NaN;
T = table((1:n)', assembly, solveTime, iterations, lastIteration, complete, ...
    'VariableNames', {'Step','AssemblyTime_s','SolveTime_s', ...
    'Iterations','LastIteration','Complete'});
if n == 0
    warning('read_solver_log:NoSteps', ...
        'No assembly labels found within the line limit. Check maxLines, log format and UTF-8 encoding.');
end
end
