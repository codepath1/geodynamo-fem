function pos_id=axis2id(x,y,distance)
N=ceil((0.5-distance/2)/distance)*2+1;%Number of subdivisions
min_distance=(1-(N-2)*distance)/2;%Minimum distance from both ends
%x position
if x<=min_distance
    pos_id_x=1;
elseif x>=1-min_distance
    pos_id_x=N;
else
    pos_id_x=ceil((x-min_distance)/distance)+1;
end
%y position
if y<=min_distance
    pos_id_y=1;
elseif y>=1-min_distance
    pos_id_y=N;
else
    pos_id_y=ceil((y-min_distance)/distance)+1;
end
%Convert (x, y) to a linear index
% pos_id=sub2ind([N,N],pos_id_y,pos_id_x);

pos_id_x = min(max(pos_id_x,1),N);
pos_id_y = min(max(pos_id_y,1),N);

pos_id = sub2ind([N,N],pos_id_y,pos_id_x);

end


