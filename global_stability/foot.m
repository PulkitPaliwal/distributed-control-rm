function [dXdt] = foot(x,l,k1,k2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%e=interp1(t,E,t);

%dPdt=-L1*P+B*E;
% x1dot=-r*sin(X(3))*atan(X(3)+(X(2)-k2)/(X(1)-k1));
% x2dot=r*cos(X(3))*atan(X(3)+(X(2)-k2)/(X(1)-k1));
% thetadot=atan(X(3)+(X(2)-k2)/(X(1)-k1));
% dXdt=[x1dot;x2dot;thetadot];

a=-(x(5)-k1);
b=-(x(6)-k2);
t1=x(7);
t2=x(8);
t3=x(9);
p1=sin(t1);
p2=cos(t1);
p3=sin(t2);
p4=cos(t2);
p5=sin(t3);
p6=cos(t3);
p7=sin(t1+t2);
p8=cos(t1+t2);
p9=sin(t1+t2+t3);
p10=cos(t1+t2+t3);
p11=sin(t2+t3);
p12=cos(t2+t3);
x_prime = b*p9 + a*p10;
y_prime = b*p10 - a*p9;
a = x_prime;
b = y_prime;
x1dot=-l*p1*(atan2((l*p3+l*p11+a*p11+b*p12),(l+l*p4+l*p12+a*p12-b*p11))-atan2((l*p3+l*p11),(l+l*p4+l*p12)));
x2dot=l*p2*(atan2((l*p3+l*p11+a*p11+b*p12),(l+l*p4+l*p12+a*p12-b*p11))-atan2((l*p3+l*p11),(l+l*p4+l*p12)));
x3dot=(-l*p1-l*p7)*(atan2((l*p3+l*p11+a*p11+b*p12),(l+l*p4+l*p12+a*p12-b*p11))-atan2((l*p3+l*p11),(l+l*p4+l*p12)))-l*p7*(atan2((l*p5+a*p5+b*p6),(l+l*p6+a*p6-b*p5))-atan2((l*p5),(l+l*p6)));
x4dot=(l*p2+l*p8)*(atan2((l*p3+l*p11+a*p11+b*p12),(l+l*p4+l*p12+a*p12-b*p11))-atan2((l*p3+l*p11),(l+l*p4+l*p12)))+l*p8*(atan2((l*p5+a*p5+b*p6),(l+l*p6+a*p6-b*p5))-atan2((l*p5),(l+l*p6)));
x5dot=(-l*p1-l*p7-l*p9)*(atan2((l*p3+l*p11+a*p11+b*p12),(l+l*p4+l*p12+a*p12-b*p11))-atan2((l*p3+l*p11),(l+l*p4+l*p12)))+(-l*p7-l*p9)*(atan2((l*p5+a*p5+b*p6),(l+l*p6+a*p6-b*p5))-atan2((l*p5),(l+l*p6)))-l*p9*atan2(b,(l+a));
x6dot=(l*p2+l*p8+l*p10)*(atan2((l*p3+l*p11+a*p11+b*p12),(l+l*p4+l*p12+a*p12-b*p11))-atan2((l*p3+l*p11),(l+l*p4+l*p12)))+(l*p8+l*p10)*(atan2((l*p5+a*p5+b*p6),(l+l*p6+a*p6-b*p5))-atan2((l*p5),(l+l*p6)))+l*p10*atan2(b,(l+a));
t1dot=(atan2((l*p3+l*p11+a*p11+b*p12),(l+l*p4+l*p12+a*p12-b*p11))-atan2((l*p3+l*p11),(l+l*p4+l*p12)));
t2dot=(atan2((l*p5+a*p5+b*p6),(l+l*p6+a*p6-b*p5))-atan2((l*p5),(l+l*p6)));
t3dot=atan2(b,(l+a));
dXdt=[x1dot;x2dot;x3dot;x4dot;x5dot;x6dot;t1dot;t2dot;t3dot];
end