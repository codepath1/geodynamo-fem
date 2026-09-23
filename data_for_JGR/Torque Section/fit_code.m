clc
clear
[X,sigma] = analyze_torque_scaling;

d1=load('Mkin_tor5e_5_rra40.txt'); 
d2=load('Mkin_tor5E_6.txt'); 
d3=load('Mkin_tore_4_rra60.txt'); 
d4=load('Mkin_torE_4.txt'); 

[meanValue1, stdValue, residual]=time_average_stats(d1);


[meanValue2, stdValue, residual]=time_average_stats(d2);


[meanValue3, stdValue, residual]=time_average_stats(d3);


[meanValue4, stdValue, residual]=time_average_stats(d4);


ro = 20/13;
ri = 7/13;
betaInverse = 40;
epsilon=0.2;

radius = @(mu) ...
        ro - epsilon .* ...
        exp(-2*betaInverse*(1-mu));

V = 2*pi/3 * integral( ...
        @(mu) radius(mu).^3 - ri^3, ...
        -1, 1, ...
        'RelTol',1e-10, ...
        'AbsTol',1e-12);


ekman=[1e-4, 1e-4, 5e-5, 5e-6]';
% cfg.names = ["e4", "e4_rra60", "5e5_rra40", "5e6"];
mag=[meanValue4,meanValue3,meanValue1,meanValue2]';

X=X;
sigmaGamma=sigma.*mag;

figure('Color','w');

%scatter(X, sigmaGamma, 60, 'filled');
%scatter(X, sigmaGamma, 100, 'r', 'p', 'filled');

sigmaGamma = sigma .* mag;

plot_loglog_fit(X, sigmaGamma);

% Remove the original scatter points drawn by the fitting function and retain the fitted line
delete(findobj(gca, 'Type', 'Scatter'));

markers = {'^', 'p', 's', 'o'};

size_t=[400,800,600,300];
% size is  ^', 'p', 's', 'o'
for i = 1:4
    scatter(X(i), sigmaGamma(i), size_t(i), 'r', ...
        markers{i}, 'filled');
end

set(gca, 'XScale','log', 'YScale','log');
xlabel('\epsilon Re^2 E^{2/3}');
ylabel('\sigma_\Gamma');

grid off;
box on;

set(gca, 'XScale','log', 'YScale','log');

xlabel('\epsilon Re^2 E^{2/3}');
ylabel('\sigma_\Gamma');

% Dummy objects used only for the legend
h1 = plot(nan,nan,'^', ...
    'MarkerSize',12, ...
    'MarkerFaceColor','r', ...
    'MarkerEdgeColor','r', ...
    'LineStyle','none');

h2 = plot(nan,nan,'p', ...
    'MarkerSize',14, ...
    'MarkerFaceColor','r', ...
    'MarkerEdgeColor','r', ...
    'LineStyle','none');

h3 = plot(nan,nan,'s', ...
    'MarkerSize',11, ...
    'MarkerFaceColor','r', ...
    'MarkerEdgeColor','r', ...
    'LineStyle','none');

h4 = plot(nan,nan,'o', ...
    'MarkerSize',11, ...
    'MarkerFaceColor','r', ...
    'MarkerEdgeColor','r', ...
    'LineStyle','none');

lgd = legend([h1 h2 h3 h4], ...
    {'e4','e4\_ra60','5e5\_ra40','5e6'});

lgd.FontSize = 18;
lgd.ItemTokenSize = [35 28];

lgd = legend([h1 h2 h3 h4], {'e4','e4\_ra60','5e5\_ra40','5e6'});
lgd.FontSize = 18;
lgd.ItemTokenSize = [28, 20];

exportgraphics(gcf, 'tor_fit.png', 'Resolution', 600); 
