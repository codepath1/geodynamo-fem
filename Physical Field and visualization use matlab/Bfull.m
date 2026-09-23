clc
clear

filename = 'B_rra40_2.505s.csv';
data = csvread(filename,1);     %% Skip the first row when reading the data

ro = 20/13;
ri = 7/13;

%% ==================== Topography parameters ====================
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

%% ==================== Read data ====================
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

%% ==================== Br on the reference sphere ====================
r_ref = ro*0.9 + ri*0.1;

[refx,refy,refz] = sphere(400);
refx = r_ref*refx;
refy = r_ref*refy;
refz = r_ref*refz;

Br_ref = griddata(x,y,z,ur,refx,refy,refz,'linear');

%% ==================== Start plotting ====================
figure('Color','w')
hold on

% --------------------------------------------------
% 1) Br reference sphere
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
% 2) Inner core
% --------------------------------------------------
%% ==================== Br on the inner-core surface ====================

[icx,icy,icz] = sphere(300);

icx = ri * icx;
icy = ri * icy;
icz = ri * icz;

% Interpolate the radial magnetic field onto the inner-core surface
Br_inner = griddata( ...
    x,y,z,ur, ...
    icx,icy,icz, ...
    'linear');

% Plot the radial magnetic field on the inner-core surface
cmb_inner = surf( ...
    icx,icy,icz,Br_inner, ...
    'EdgeColor','none', ...
    'FaceColor','interp');

set(cmb_inner,'SpecularStrength',0.15);

shading interp
% --------------------------------------------------
% 3) Mark only the topographic-relief region on the outer boundary
%    Do not draw the entire outer boundary; highlight only that region
% --------------------------------------------------
[so_x,so_y,so_z] = sphere(300);

% Unit normal
nx = so_x;
ny = so_y;
nz = so_z;

% Topographic-relief amplitude: difference between the regular-sphere radius ro and the actual outer-boundary radius
topo_drop = ro - r_outer(nx,ny,nz);

% Threshold: mark only regions with appreciable relief
% Try changing this to 0.03*epsilon, 0.05*epsilon, or 0.1*epsilon
topo_mask = topo_drop > 0.05*epsilon;

% Highlight this region on the regular outer sphere ro
Xmark = ro * so_x;
Ymark = ro * so_y;
Zmark = ro * so_z;

% Keep only the relief region and set the rest to NaN
Xmark(~topo_mask) = NaN;
Ymark(~topo_mask) = NaN;
Zmark(~topo_mask) = NaN;

% Draw this region using a constant color
Cmark = ones(size(Xmark));

surf(Xmark,Ymark,Zmark,Cmark, ...
    'EdgeColor','none', ...
    'FaceAlpha',0.85);

% Set this region's color (dark red)
colormap(jet)

% To specify the color separately, patch/scatter3 can also be used;
% surf with a single-valued color also works here.

% Add a relief-center point to aid identification
plot3(ro*n0(1), ro*n0(2), ro*n0(3), ...
    'ko', 'MarkerFaceColor','y', 'MarkerSize',7);

%% ==================== 3-D magnetic-field interpolation grid ====================
N = 100;
[xx,yy,zz] = meshgrid(linspace(-2,2,N), ...
                      linspace(-2,2,N), ...
                      linspace(-2,2,N));

U3d = griddata(x,y,z,u,xx,yy,zz,'linear');
V3d = griddata(x,y,z,v,xx,yy,zz,'linear');
W3d = griddata(x,y,z,w,xx,yy,zz,'linear');

% Use the actual relief outer boundary as the mask
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

%% ==================== Seed points ====================
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

%% ==================== 3-D magnetic field lines / stream tubes ====================
verts = stream3(xx,yy,zz,U3d,V3d,W3d,sx,sy,sz);
verts = verts(~cellfun(@isempty, verts));

streamtube(verts,xx,yy,zz,alphad3/100,[0 20]);   % Adjustable: /50

%% ==================== Figure settings ====================
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


drawnow   % Wait for automatic layout to finish

axPos = ax.Position;
cbPos = cb.Position;

%% 1. Reduce the height
shrink = 0.5;                     % 0.5    0.7 ->90°
newHeight = cbPos(4) * shrink;
cbPos(4) = newHeight;

%% 2. Center vertically
cbPos(2) = axPos(2) + (axPos(4) - newHeight)/2;

%% 3. Align closely with the right side (important)
gap = 0.005;                        % A smaller value gives a tighter alignment
cbPos(1) = axPos(1) + axPos(3) + gap;

cb.Position = cbPos;

exportgraphics(gcf, 'B3d_line_innercore_topomark.png', 'Resolution', 600);

