function [X, sigmaGamma, fitInfo] = fit_torque_scaling(dataDir)
%FIT_TORQUE_SCALING
% Compute only:
%   
%   X = epsilon * Re^2 * E^(2/3)
%
% and the temporal standard deviation sigmaGamma of torque Gamma_z,
% then fit
%
%   sigmaGamma = C * X^alpha
%
% Input files:
%   case.txt        : time  Gamma_x  Gamma_y  Gamma_z
%   ekin_case.txt   : time  kinetic_energy
%
% Outputs:
%   X          : scaling abscissa for each case
%   sigmaGamma : temporal standard deviation of Gamma_z
%   fitInfo    : free log-log fit results

if nargin < 1 || isempty(dataDir)
    dataDir = fileparts(mfilename('fullpath'));
end


%% ================= Configuration =================

names = ["e4", "e4_rra60", "5e5_rra40", "5e6"];

E       = [1e-4, 1e-4, 5e-5, 5e-6];
epsilon = [0.2,  0.2,  0.2,  0.2];

% Spherical-shell parameters
ro = 20/13;
ri = 7/13;
betaInverse = 40;

% If no time window is specified:
% automatically discard the first 20% of the common time interval
tStart = [NaN, NaN, NaN, NaN];
tEnd   = [NaN, NaN, NaN, NaN];

discardFraction = 0.00;

%% ========================================

n = numel(names);

X = nan(n,1);
sigmaGamma = nan(n,1);
Re = nan(n,1);


for k = 1:n

    %% ---------- Read torque ----------
    A = readNumericLines( ...
        fullfile(dataDir, names(k) + ".txt"), 4);

    tg = A(:,1);
    Gz = A(:,4);


    %% ---------- Read kinetic energy ----------
    B = readNumericLines( ...
        fullfile(dataDir, "ekin_" + names(k) + ".txt"), 2);

    te = B(:,1);
    energy = B(:,2);


    %% ---------- Compute fluid volume ----------
    radius = @(mu) ...
        ro - epsilon(k) .* ...
        exp(-2*betaInverse*(1-mu));

    V = 2*pi/3 * integral( ...
        @(mu) radius(mu).^3 - ri^3, ...
        -1, 1, ...
        'RelTol',1e-10, ...
        'AbsTol',1e-12);


    %% ---------- Derive Re from kinetic energy ----------
    %
    % Current definition:
    %   Ekin = 1/2 ∫ |u|^2 dV    ×
    %   Ekin = sqrt(∫ |u|^2 dV)  √
    %   
    % Therefore
    %
    %   Re^2 = 2 Ekin / V   ×
    %   Re^2 = Ekin**2 / V  √
    
    
    Re2 = energy.^2 / V;


    %% ---------- Common statistical time interval ----------

    lo = max(tg(1), te(1));
    hi = min(tg(end), te(end));

    a = tStart(k);
    b = tEnd(k);

    if isnan(a)
        a = lo + discardFraction*(hi-lo);
    end

    if isnan(b)
        b = hi;
    end


    %% ---------- Subset data ----------

    [tt, gg] = cutWindow(tg, Gz, a, b);
    [et, rr] = cutWindow(te, Re2, a, b);


    %% ---------- sigma_Gamma ----------

    meanGamma = timeMean(tt, gg);

    sigmaGamma(k) = sqrt( ...
        timeMean(tt, (gg-meanGamma).^2) );


    %% ---------- Re ----------

    Re(k) = sqrt(timeMean(et, rr));


    %% ---------- Scaling variable X ----------

    X(k) = epsilon(k) ...
         * Re(k)^2 ...
         * E(k)^(2/3);

end


%% ============================================================
%                    X-sigmaGamma fit
%% ============================================================

valid = isfinite(X) & isfinite(sigmaGamma) ...
      & X > 0 & sigmaGamma > 0;

lx = log10(X(valid));
ly = log10(sigmaGamma(valid));

% Free fit:
%
% log10(sigma) = alpha log10(X) + b
%
p = polyfit(lx, ly, 1);

alpha = p(1);
b = p(2);

C = 10^b;

yfit = polyval(p,lx);

SStot = sum((ly-mean(ly)).^2);
SSres = sum((ly-yfit).^2);

R2 = 1 - SSres/SStot;


fitInfo.slope = alpha;
fitInfo.coefficient = C;
fitInfo.interceptLog10 = b;
fitInfo.R2log = R2;


fprintf('\n---------------------------------\n');
fprintf('sigma = C * X^alpha\n');
fprintf('alpha = %.6f\n', alpha);
fprintf('C     = %.6e\n', C);
fprintf('R2    = %.6f\n', R2);
fprintf('---------------------------------\n\n');


%% ---------- Plotting ----------

figure('Color','w');
hold on;

loglog(X, sigmaGamma, ...
    'o', ...
    'MarkerSize',8, ...
    'MarkerFaceColor','auto');


xx = logspace( ...
    log10(min(X(valid))), ...
    log10(max(X(valid))), ...
    200);

yy = C * xx.^alpha;

loglog(xx, yy, ...
    'k-', ...
    'LineWidth',1.5);


xlabel('\epsilon Re^2 E^{2/3}');
ylabel('\sigma_\Gamma');

legend( ...
    'Simulation', ...
    sprintf('Fit: \\sigma_\\Gamma = %.3g X^{%.3f}',C,alpha), ...
    'Location','best');

grid on;
box on;

set(gca, ...
    'XScale','log', ...
    'YScale','log');

end


%% ============================================================
%                     Helper functions
%% ============================================================

function A = readNumericLines(path,ncol)

assert(isfile(path),'找不到文件：%s',path);

linesText = splitlines(string(fileread(path)));

A = zeros(numel(linesText),ncol);

count = 0;

for j = 1:numel(linesText)

    s = strtrim(linesText(j));

    if strlength(s)==0 || ...
       startsWith(s,'#') || ...
       startsWith(s,'%')
        continue
    end

    s = regexprep(s,'[\[\],;]',' ');
    s = regexprep(s,'[dD]','e');

    v = sscanf(char(s),'%f');

    if numel(v) ~= ncol
        error('文件 %s 第 %d 行格式错误。',path,j);
    end

    count = count + 1;

    A(count,:) = v(:)';

end

A = A(1:count,:);

assert(all(diff(A(:,1))>0), ...
    '文件 %s 中时间不是严格递增。',path);

end


function [tw,yw] = cutWindow(t,y,a,b)

inside = t>a & t<b;

tw = [a; t(inside); b];

yw = [ ...
    interp1(t,y,a,'linear'); ...
    y(inside,:); ...
    interp1(t,y,b,'linear')];

end


function m = timeMean(t,y)

m = trapz(t,y) / (t(end)-t(1));

end