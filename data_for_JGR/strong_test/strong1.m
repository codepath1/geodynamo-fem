%% Strong scaling: journal-style single-panel figure
% MATLAB R2020a or later (exportgraphics).
% n_p: total MPI ranks; n_P: MPI ranks assigned to the Poisson solve.
clear; clc; close all;

p_mid = [64 128 256];
T_mid_fixed64 = [1.1094 0.7525 0.2946];
T_mid_all = [1.1109 0.8556 0.4329];
p_huge = [128 256 512 1024];
T_huge_fixed128 = [6.4298 3.4443 2.4775 1.6991];
T_huge_np8 = [10.1368 4.8509 2.8557 1.6711];

% IMPORTANT: select the label matching the actual measured timing.
% Default retains the label in the supplied figure.
timingLabel = 'Time per preconditioner application (s)';
% timingLabel = 'Time per outer Krylov iteration (s)';

blue = [0.000 0.447 0.698];
orange = [0.835 0.369 0.000];
gray = [0.48 0.48 0.48];
fontName = 'Arial';

fig = figure('Color','w','Units','centimeters', ...
    'Position',[3 3 16.5 11.0],'Renderer','painters');
ax = axes(fig,'Position',[0.135 0.15 0.835 0.81]);
hold(ax,'on');
set(ax,'XScale','log','YScale','log');

% Ideal references anchored to the first FIXED-RANK measurement
% for each mesh. No arbitrary vertical shift.
q_mid = logspace(log10(p_mid(1)),log10(p_mid(end)),100);
q_huge = logspace(log10(p_huge(1)),log10(p_huge(end)),100);
hIdeal = plot(ax,q_mid,T_mid_fixed64(1)*p_mid(1)./q_mid, ...
    ':','Color',gray,'LineWidth',1.25);
plot(ax,q_huge,T_huge_fixed128(1)*p_huge(1)./q_huge, ...
    ':','Color',gray,'LineWidth',1.25,'HandleVisibility','off');

% Color distinguishes mesh; markers and line styles distinguish allocation.
style = {'LineWidth',1.45,'MarkerSize',6,'MarkerFaceColor','w'};
h1 = plot(ax,p_mid,T_mid_fixed64,'-o','Color',blue,style{:});
h2 = plot(ax,p_mid,T_mid_all,'--s','Color',blue,style{:});
h3 = plot(ax,p_huge,T_huge_fixed128,'-d','Color',orange,style{:});
h4 = plot(ax,p_huge,T_huge_np8,'--^','Color',orange,style{:});

set(ax,'FontName',fontName,'FontSize',10,'LineWidth',0.8, ...
    'TickDir','out','TickLength',[0.012 0.012],'Box','on', ...
    'XLim',[56 1180],'YLim',[0.22 13], ...
    'XTick',[64 128 256 512 1024], ...
    'XTickLabel',{'64','128','256','512','1024'}, ...
    'YTick',[0.25 0.5 1 2 4 8], ...
    'YTickLabel',{'0.25','0.5','1','2','4','8'}, ...
    'XMinorTick','off','YMinorTick','off', ...
    'XGrid','on','YGrid','on','XMinorGrid','off','YMinorGrid','off', ...
    'GridLineStyle',':','GridColor',[0.65 0.65 0.65], ...
    'GridAlpha',0.22,'Layer','top','TickLabelInterpreter','tex');
xlabel(ax,'Number of MPI processes, n_p', ...
    'Interpreter','tex','FontName',fontName,'FontSize',11);
ylabel(ax,timingLabel,'FontName',fontName,'FontSize',11);

lgd = legend(ax,[h1 h2 h3 h4 hIdeal], ...
    {'GMid: n_P = 64','GMid: n_P = n_p', ...
     'GHuge: n_P = 128','GHuge: n_P = n_p/8', ...
     'Ideal scaling: T \propto n_p^{-1}'}, ...
    'Location','southwest','Interpreter','tex', ...
    'FontName',fontName,'FontSize',9,'Box','off');
lgd.ItemTokenSize = [26 12]; 
drawnow; 

% Files are written alongside this script.
outDir = fileparts(mfilename('fullpath'));
if isempty(outDir), outDir = pwd; end
exportgraphics(fig,fullfile(outDir,'strong_scaling_JGR.pdf'), ...
    'ContentType','vector','BackgroundColor','white');
exportgraphics(fig,fullfile(outDir,'strong_scaling_JGR.png'), ...
    'Resolution',600,'BackgroundColor','white');
