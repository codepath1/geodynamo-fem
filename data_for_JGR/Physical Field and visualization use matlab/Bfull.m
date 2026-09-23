clc
clear

filename = 'B_rra40_2.505s.csv';
data = csvread(filename,1);     %% 偏移一行向下读入数据

ro = 20/13;
ri = 7/13;

%% ==================== 地形参数 ====================
epsilon      = 0.2;
theta0       = 2.40;
phi0         = 0.0;
beta_inverse = 40.0;

n0 = [sin(theta0)*cos(phi0), ...
      sin(theta0)*sin(phi0), ...
      cos(theta0)];

r_outer = @(nx,ny,nz) ro - epsilon .* exp( ...
    -beta_inverse .* ((nx-n0(1)).^2 + ...
                      (ny-n0(2)).^2 + ...
                      (nz-n0(3)).^2));

%% ==================== 读取数据 ====================
idx = find((data(:,4).^2 + data(:,5).^2 + data(:,6).^2) <= (ro).^2 + 0.01);
data0 = data(idx,:);

x = data0(:,4);
y = data0(:,5);
z = data0(:,6);
u = data0(:,1);
v = data0(:,2);
w = data0(:,3);

r = sqrt(x.^2 + y.^2 + z.^2);
ur = u./r.*x + v./r.*y + w./r.*z;

%% ==================== 参考球面上的 Br ====================
r_ref = ro*0.9 + ri*0.1;

[refx,refy,refz] = sphere(400);
refx = r_ref*refx;
refy = r_ref*refy;
refz = r_ref*refz;

Br_ref = griddata(x,y,z,ur,refx,refy,refz,'linear');

%% ==================== 开始绘图 ====================
figure('Color','w')
hold on

% --------------------------------------------------
% 1) Br 参考球面
% --------------------------------------------------
cmbr = surf(refx,refy,refz,Br_ref);
shading flat
caxis([-3,3])

alphad = Br_ref.^2;
set(cmbr,'FaceAlpha','flat');
set(cmbr,'AlphaData',alphad);
set(cmbr,'SpecularStrength',0.2);
alim([0 2])

% --------------------------------------------------
% 2) 内核
% --------------------------------------------------
%% ==================== 内核表面的 Br ====================

[icx,icy,icz] = sphere(300);

icx = ri * icx;
icy = ri * icy;
icz = ri * icz;

% 将径向磁场插值到内核表面
Br_inner = griddata( ...
    x,y,z,ur, ...
    icx,icy,icz, ...
    'linear');

% 绘制内核表面的径向磁场
cmb_inner = surf( ...
    icx,icy,icz,Br_inner, ...
    'EdgeColor','none', ...
    'FaceColor','interp');

set(cmb_inner,'SpecularStrength',0.15);

shading interp
% --------------------------------------------------
% 3) 只标记外边界上“地形起伏区域”
%    不画整个外边界，只高亮那一块
% --------------------------------------------------
[so_x,so_y,so_z] = sphere(300);

% 单位法向
nx = so_x;
ny = so_y;
nz = so_z;

% 地形起伏幅值：规则球半径 ro 与实际外边界半径之差
topo_drop = ro - r_outer(nx,ny,nz);

% 阈值：只标记起伏比较明显的区域
% 你可以改成 0.03*epsilon, 0.05*epsilon, 0.1*epsilon 试试
topo_mask = topo_drop > 0.05*epsilon;

% 在规则外球面 ro 上高亮该区域
Xmark = ro * so_x;
Ymark = ro * so_y;
Zmark = ro * so_z;

% 只保留起伏区域，其余设为 NaN
Xmark(~topo_mask) = NaN;
Ymark(~topo_mask) = NaN;
Zmark(~topo_mask) = NaN;

% 用常数颜色绘制这一块区域
Cmark = ones(size(Xmark));

surf(Xmark,Ymark,Zmark,Cmark, ...
    'EdgeColor','none', ...
    'FaceAlpha',0.85);

% 设置这一块区域的颜色（深红色）
colormap(jet)

% 如果你想单独指定颜色，也可以改用 patch/scatter3；
% 这里用 surf + 单值颜色也可以工作。

% 另外加一个起伏中心点，便于定位
plot3(ro*n0(1), ro*n0(2), ro*n0(3), ...
    'ko', 'MarkerFaceColor','y', 'MarkerSize',7);

%% ==================== 三维磁场插值网格 ====================
N = 100;
[xx,yy,zz] = meshgrid(linspace(-2,2,N), ...
                      linspace(-2,2,N), ...
                      linspace(-2,2,N));

U3d = griddata(x,y,z,u,xx,yy,zz,'linear');
V3d = griddata(x,y,z,v,xx,yy,zz,'linear');
W3d = griddata(x,y,z,w,xx,yy,zz,'linear');

% 使用真实起伏外边界作为掩膜
r_grid = sqrt(xx.^2 + yy.^2 + zz.^2);
r_safe = max(r_grid, eps);

nxg = xx ./ r_safe;
nyg = yy ./ r_safe;
nzg = zz ./ r_safe;

r_outer_grid = r_outer(nxg, nyg, nzg);

mask = (r_grid >= ri) & (r_grid <= r_outer_grid);

U3d(~mask) = NaN;
V3d(~mask) = NaN;
W3d(~mask) = NaN;

alphad3 = sqrt(U3d.^2 + V3d.^2 + W3d.^2);

%% ==================== 种子点 ====================
[x_seed, y_seed, z_seed] = meshgrid(linspace(-ro, ro, 10), ...
                                    linspace(-ro, ro, 10), ...
                                    linspace(-ro, ro, 10));

r_seed = sqrt(x_seed.^2 + y_seed.^2 + z_seed.^2);
r_seed_safe = max(r_seed, eps);

nx_seed = x_seed ./ r_seed_safe;
ny_seed = y_seed ./ r_seed_safe;
nz_seed = z_seed ./ r_seed_safe;

r_outer_seed = r_outer(nx_seed, ny_seed, nz_seed);

shell_mask = (r_seed >= ri) & (r_seed <= r_outer_seed);

sx = x_seed(shell_mask); 
sy = y_seed(shell_mask); 
sz = z_seed(shell_mask); 

sx = sx(:)'; 
sy = sy(:)'; 
sz = sz(:)'; 

%% ==================== 三维磁力线 / 流管 ====================
verts = stream3(xx,yy,zz,U3d,V3d,W3d,sx,sy,sz);
verts = verts(~cellfun(@isempty, verts));

streamtube(verts,xx,yy,zz,alphad3/100,[0 20]);   % 可调：/50

%% ==================== 图形设置 ====================
view(3)
axis equal
axis tight

camlight headlight
camlight right
lighting gouraud
shading flat

grid off
box on
colormap(jet)

set(gca, 'XColor', 'none', 'YColor', 'none', 'ZColor', 'none')

cb = colorbar('eastoutside');
cb.Label.String = 'B_r on reference sphere';


ax = gca;
cb = colorbar('eastoutside');


drawnow   % 等自动布局完成

axPos = ax.Position;
cbPos = cb.Position;

%% 1️⃣ 缩短高度
shrink = 0.5;                     % 0.5    0.7 ->90°
newHeight = cbPos(4) * shrink;
cbPos(4) = newHeight;

%% 2️⃣ 垂直居中
cbPos(2) = axPos(2) + (axPos(4) - newHeight)/2;

%% 3️⃣ 紧贴右侧（关键）
gap = 0.005;                        % 越小越贴
cbPos(1) = axPos(1) + axPos(3) + gap;

cb.Position = cbPos;

exportgraphics(gcf, 'B3d_line_innercore_topomark.png', 'Resolution', 600);

