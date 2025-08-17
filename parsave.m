
function [] = parsave(dir,x,y)
%save x,y in dir
% so I can save in parfor loop
save(dir,'x','y');
end