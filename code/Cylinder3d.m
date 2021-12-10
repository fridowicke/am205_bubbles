%%This code is to create any shape
% %theta: change t1 and t2 to set the satrting and ednsing values for theta
% t1 = 0;
% t2 = 2*pi;
% theta = linspace(t1,t2);
% 
% %r: change r1 and r2 to set the satrting and ending values for theta
% r1 = 0;
% r2 = 1;
% r = linspace(r1,r2);
% 
% %create meshfrid for inputs:
% [theta,r] = meshgrid(theta,r);
% 
% %Apply the function to create a matrix of z-values. Change teh fucntio
% %to match what you want to plot
% 
% z = ones(100,100);
% 
% %convert to cartesian and plot using mesh:
% [x,y,z] = pol2cart(theta,r,z);
% surf(x,y,z)

%%This section is to use Matlab's DelaunayTraingulation to 
[x,y,z] = cylinder(1);
X = [x(1,:) x(2,:)];
Y = [y(1,:) y(2,:)];
Z = [z(1,:) z(2,:)];

for i = 1:10
    X = [X x(1,:)];
    Y = [Y y(1,:)];
    Z = [Z .1*i*z(2,:)]
end

scatter3(X,Y,Z)
DT = delaunayTriangulation(X',Y',Z')
tetramesh(DT,'FaceAlpha',0.3)