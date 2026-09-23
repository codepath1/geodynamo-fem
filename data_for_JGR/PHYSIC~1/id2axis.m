function xpoint=id2axis(distance,id)
%Use the grid-cell midpoint
N=ceil((0.5-distance/2)/distance)*2+1;%Number of subdivisions
min_distance=(1-(N-2)*distance)/2;%Minimum distance from both ends
if id==1
    xpoint=min_distance/2;
elseif id==N
    xpoint=1-min_distance/2;
else
    xpoint=min_distance+(id-1.5)*distance;
end
end