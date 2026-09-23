clc;
clear;

% Replace with your log-file name
T = read_solver_log('job.out\job.39163336.out',20000);

% Display information for each time step
disp(T);

% Include only complete time steps
Tc = T(T.Complete, :);

fprintf('完整时间步数：%d\n', height(Tc)); 

fprintf('平均迭代次数：%.2f\n', mean(Tc.Iterations)); 

% Export results
writetable(T, 'solver_statistics.csv');