function [sl_i,xy_end,xy_start]=delete_self(sl_i,xy_end,dend,xy_start,dstart)
%sl_i streamline, stored as an N-by-2 array
N=size(sl_i,1);
pos_id_last=axis2id(sl_i(1,1),sl_i(1,2),dend);
xy_end(pos_id_last)=1;%Mark the first point

%Also mark xy_start
pos_id_s=axis2id(sl_i(1,1),sl_i(1,2),dstart);
xy_start(pos_id_s)=1;
for j=2:N
    pos_id_now=axis2id(sl_i(j,1),sl_i(j,2),dend);
    if pos_id_now~=pos_id_last
        %Ignore the current point if it lies in the same cell as the previous point
        %If it lies in a different cell, check whether the new cell is occupied
        if xy_end(pos_id_now)==1
            %If occupied, the streamline is too close to another streamline, so stop immediately
            j=j-1;
            break
        else
            %If unoccupied, add the new point
            xy_end(pos_id_now)=1;
            pos_id_last=pos_id_now;
        end
    end
    %Also mark xy_start
    pos_id_s=axis2id(sl_i(j,1),sl_i(j,2),dstart);
    xy_start(pos_id_s)=1;
end
sl_i(j:end,:)=[];
end