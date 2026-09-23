clc;
clear;

%filename = 'u_rra40_2.505s.csv';
filename = 'u_e4_rra60.csv';
% Plot three cross-sections
cut_planes = {'x', 'y', 'z'};

%% ==================== Spherical-shell and topography parameters ====================
ro = 20/13;
ri = 7/13;

epsilon      = 0.2;
theta0       = 2.40;
phi0         = 0.0;
beta_inverse = 40.0;

N = 400;

% Streamline-density parameter
% Smaller values produce denser streamlines, e.g., 0.03-0.06
dstart = 0.01;

n0 = [sin(theta0)*cos(phi0), ...
      sin(theta0)*sin(phi0), ...
      cos(theta0)];

r_outer = @(nx,ny,nz) ro - epsilon .* exp( ...
    -beta_inverse .* ((nx-n0(1)).^2 + ...
                      (ny-n0(2)).^2 + ...
                      (nz-n0(3)).^2));

assert(ro - epsilon > ri, '地形与内边界相交。');


%% ==================== Read velocity data ====================
% Columns 1:3: ux, uy, uz
% Columns 4:6: x, y, z

data = readmatrix(filename);

assert(size(data,2) >= 6, ...
    'CSV 至少需要六列：ux,uy,uz,x,y,z。');

data = data(all(isfinite(data(:,1:6)),2),:);

assert(~isempty(data), 'CSV 中没有有效数据。');


%% ==================== 3-D velocity interpolation ====================

Fu = scatteredInterpolant( ...
    data(:,4),data(:,5),data(:,6), ...
    data(:,1),'linear','none');

Fv = scatteredInterpolant( ...
    data(:,4),data(:,5),data(:,6), ...
    data(:,2),'linear','none');

Fw = scatteredInterpolant( ...
    data(:,4),data(:,5),data(:,6), ...
    data(:,3),'linear','none');


%% ==================== 2-D cross-section grid ====================

% Use an even number of points to avoid sampling exactly on the rotation axis
q = linspace(-ro,ro,2*N);

[S,T] = meshgrid(q,q);

R = hypot(S,T);
Rsafe = max(R,eps);

% Used to draw the actual outer boundary
a = linspace(0,2*pi,2400);


%% =====================================================
%                    Three cross-sections
%% =====================================================

for j = 1:numel(cut_planes)

    plane = lower(cut_planes{j});


    %% -------------------------------------------------
    %  Define the cross-section orientation
    %% -------------------------------------------------

    switch plane

        case 'x'
            % x = 0
            % Horizontal coordinate: y
            % Vertical coordinate: z

            e1 = [0,1,0];
            e2 = [0,0,1];

            label_s = 'y';
            label_t = 'z';


        case 'y'
            % y = 0
            % Horizontal coordinate: x
            % Vertical coordinate: z

            e1 = [1,0,0];
            e2 = [0,0,1];

            label_s = 'x';
            label_t = 'z';


        case 'z'
            % z = 0
            % Horizontal coordinate: x
            % Vertical coordinate: y

            e1 = [1,0,0];
            e2 = [0,1,0];

            label_s = 'x';
            label_t = 'y';


        otherwise

            error('截面只能选 x、y 或 z。');

    end


    %% -------------------------------------------------
    %  3-D coordinates of the current cross-section
    %% -------------------------------------------------

    Xq = S*e1(1) + T*e2(1);
    Yq = S*e1(2) + T*e2(2);
    Zq = S*e1(3) + T*e2(3);


    %% -------------------------------------------------
    %  Outer boundary with Gaussian topography
    %% -------------------------------------------------

    Rout = r_outer( ...
        Xq./Rsafe, ...
        Yq./Rsafe, ...
        Zq./Rsafe);

    shell_mask = ...
        (R >= ri) & ...
        (R <= Rout);


    %% -------------------------------------------------
    %  Interpolate Cartesian velocity
    %% -------------------------------------------------

    Uq = Fu(Xq,Yq,Zq);
    Vq = Fv(Xq,Yq,Zq);
    Wq = Fw(Xq,Yq,Zq);


    %% -------------------------------------------------
    %  Compute radial velocity u_r
    %
    %      ur = u · er
    %
    %% -------------------------------------------------

    Phi = atan2(Yq,Xq);

    Theta = atan2( ...
        hypot(Xq,Yq), ...
        Zq);

    Ur = ...
        Uq.*sin(Theta).*cos(Phi) + ...
        Vq.*sin(Theta).*sin(Phi) + ...
        Wq.*cos(Theta);

    Ur(~shell_mask) = NaN;


    %% =================================================
    %              In-plane 2-D velocity
    %
    %   Note:
    %   Ur cannot be used to draw the streamlines.
    %
    %   The streamlines represent the actual velocity projected onto the cross-section.
    %% =================================================

    switch plane

        case 'x'

            % x = 0
            %
            % Horizontal coordinate: y
            % Vertical coordinate: z
            %
            % Therefore, the streamline velocity is:
            %
            % Us = uy
            % Ut = uz

            Us = Vq;
            Ut = Wq;


        case 'y'

            % y = 0
            %
            % Horizontal coordinate: x
            % Vertical coordinate: z

            Us = Uq;
            Ut = Wq;


        case 'z'

            % z = 0
            %
            % Horizontal coordinate: x
            % Vertical coordinate: y

            Us = Uq;
            Ut = Vq;

    end


    % Streamlines cannot exist outside the shell or inside the inner core
    Us(~shell_mask) = NaN;
    Ut(~shell_mask) = NaN;


    %% -------------------------------------------------
    %  Actual outer boundary
    %% -------------------------------------------------

    nx_b = e1(1)*cos(a) + e2(1)*sin(a);
    ny_b = e1(2)*cos(a) + e2(2)*sin(a);
    nz_b = e1(3)*cos(a) + e2(3)*sin(a);

    rb = r_outer(nx_b,ny_b,nz_b);

    s_outer = rb.*cos(a);
    t_outer = rb.*sin(a);

    s_inner = ri*cos(a);
    t_inner = ri*sin(a);


    %% =================================================
    %                  Ur contour map
    %% =================================================

    valid_values = Ur(isfinite(Ur));

    assert(~isempty(valid_values), ...
        '当前截面没有有效插值数据。');

    max_value = max(abs(valid_values));

    if max_value == 0
        max_value = 1;
    end

    levels = linspace(-max_value,max_value,61);


    fig = figure( ...
        'Color','w', ...
        'Position',[100+80*(j-1),100,850,780]);

    ax = axes(fig);

    hold(ax,'on');


    %% -------------------------------------------------
    %  Ur contour map
    %% -------------------------------------------------

    contourf( ...
        ax, ...
        S,T,Ur, ...
        levels, ...
        'LineColor','none');


    %% =================================================
    %                     Streamlines
    %% =================================================

    streamline_sum = my_streamline( ...
        S,T, ...
        Us,Ut, ...
        dstart);


    for kk = 1:numel(streamline_sum)

        sl = streamline_sum{kk};

        if isempty(sl)
            continue;
        end

        % ------------------------------------------------
        % Check again that each streamline lies inside the actual spherical shell
        % Prevent rare streamlines from crossing the topographic boundary
        % ------------------------------------------------

        ss = sl(:,1);
        tt = sl(:,2);

        rr = hypot(ss,tt);

        switch plane

            case 'x'

                xx = zeros(size(ss));
                yy = ss;
                zz = tt;

            case 'y'

                xx = ss;
                yy = zeros(size(ss));
                zz = tt;

            case 'z'

                xx = ss;
                yy = tt;
                zz = zeros(size(ss));

        end


        rr_safe = max(rr,eps);

        rout_sl = r_outer( ...
            xx./rr_safe, ...
            yy./rr_safe, ...
            zz./rr_safe);


        good = ...
            rr >= ri & ...
            rr <= rout_sl;


        % Set values outside the spherical shell to NaN
        % plot automatically breaks the streamline at NaN values
        ss(~good) = NaN;
        tt(~good) = NaN;


        plot( ...
            ax, ...
            ss,tt, ...
            'k-', ...
            'LineWidth',0.7);

    end


    %% -------------------------------------------------
    %  Inner and outer boundaries
    %% -------------------------------------------------

    plot( ...
        ax, ...
        s_outer,t_outer, ...
        'k', ...
        'LineWidth',1.6);

    plot( ...
        ax, ...
        s_inner,t_inner, ...
        'k', ...
        'LineWidth',1.6);


    %% -------------------------------------------------
    %  Figure settings
    %% -------------------------------------------------

    hold(ax,'off');

    axis(ax,'equal');

    xlim(ax,[-1.03*ro,1.03*ro]);
    ylim(ax,[-1.03*ro,1.03*ro]);

    box(ax,'on');

    set( ...
        ax, ...
        'FontSize',15, ...
        'LineWidth',1, ...
        'Layer','top');

    colormap(ax,jet(256));

    clim(ax,[-max_value,max_value]);

    cb = colorbar(ax);
    cb.Label.String = 'u_r';

    xlabel(ax,label_s);
    ylabel(ax,label_t);

    % title( ...
    %     ax, ...
    %     ['u_r + streamlines,  ',plane,' = 0'], ...
    %     'Interpreter','tex');


    %% -------------------------------------------------
    %  Output
    %% -------------------------------------------------

    output_file = sprintf( ...
        'ur_streamline_%s0.png', ...
        plane);

    exportgraphics( ...
        fig, ...
        output_file, ...
        'Resolution',600, ...
        'BackgroundColor','white');

end