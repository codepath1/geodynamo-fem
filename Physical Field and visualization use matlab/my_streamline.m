function streamline_sum=my_streamline(x,y,u,v,dstart)
%0. Preprocessing settings
%Set the grid density (normalized length in the [0, 1] interval)
%dstart = 0.05; default is 0.05
dend=0.5*dstart;

%xmin=min(x,[],'all');xmax=max(x,[],'all');
%ymin=min(y,[],'all');ymax=max(y,[],'all');
xmin=min(min(min(x)));xmax=max(max(max(x)));
ymin=min(min(min(y)));ymax=max(max(max(y)));

%Normalize the flow field to a rectangle in the [0, 1] interval
xn=(x-xmin)/(xmax-xmin);
yn=(y-ymin)/(ymax-ymin);
un=u/(xmax-xmin);
vn=v/(ymax-ymin);

num_start=ceil((0.5-dstart/2)/dstart)*2+1;
num_end=ceil((0.5-dend/2)/dend)*2+1;

%Initialize all grid points: 0 means available and 1 means already occupied
xy_start=zeros(num_start,num_start);
xy_end=zeros(num_end,num_end);

%1. Continue while xy_start contains an available location for a new point
k=0;%Loop count, also the number of streamlines
while ~all(all(xy_start))
    k=k+1;
    %2. Randomly select a grid point in start as a seed
    [start_id_y,start_id_x]=find(xy_start==0);
    randnum=randi(size(start_id_y,1));
    x_pos_i=id2axis(dstart,start_id_x(randnum,1));
    y_pos_i=id2axis(dstart,start_id_y(randnum,1));
    %3. Trace the streamline
    streamline_i_1 = stream2(xn,yn, un, vn,x_pos_i,y_pos_i,0.2);
    streamline_i_2 = stream2(xn,yn,-un,-vn,x_pos_i,y_pos_i,0.2);
    %4. Using xy_end, remove self-intersections or points that are too close, and mark xy_end
    [streamline_i_1,xy_end,xy_start]=delete_self(streamline_i_1{1},xy_end,dend,xy_start,dstart);
    [streamline_i_2,xy_end,xy_start]=delete_self(streamline_i_2{1},xy_end,dend,xy_start,dstart);
    %5. Save
    streamline_k=[flipud(streamline_i_2);streamline_i_1(2:end,:)];%New streamline
    streamline_sum{k}=[xmin+streamline_k(:,1)*(xmax-xmin),ymin+streamline_k(:,2)*(ymax-ymin)];%Transform back from normalized coordinates
end
end