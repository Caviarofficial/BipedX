function T = forward_kinematics(theta, a2, a3, d2, d4)
% 
% 这个函数能够计算对PUMA560的末端相对于世界坐标系的位姿矩阵，其中六个关节均为旋转关节，六个自由度均为角度
% FORWARD_KINEMATICS 计算6自由度机器人的正运动学
%   输入:
%       theta : 1x6 向量，关节角度 [theta1, theta2, theta3, theta4, theta5, theta6] (弧度)
%       a2    : 连杆2长度
%       a3    : 连杆3长度
%       d2    : 连杆2偏距
%       d4    : 连杆4偏距
%   输出:
%       T     : 4x4 齐次变换矩阵，从基坐标系到末端执行器坐标系

    % D-H 参数表 (modified D-H 约定): 每一行 = [alpha, a, d, theta]
    % alpha: 连杆扭角, a: 连杆长度, d: 连杆偏距, theta: 关节角
    DH = [0,        0,       0,  theta(1);
         -pi/2,     0,      d2,  theta(2);
          0,       a2,       0,  theta(3);
         -pi/2,    a3,      d4,  theta(4);
          pi/2,     0,       0,  theta(5);
         -pi/2,     0,       0,  theta(6)];

    T = eye(4);  % 初始化为单位矩阵
    for i = 1:6
        alpha = DH(i,1);
        a     = DH(i,2);
        d     = DH(i,3);
        theta_i = DH(i,4);

        % 根据 modified D-H 约定计算单个连杆变换矩阵
        Ti = [cos(theta_i), -sin(theta_i),           0,             a;
              sin(theta_i)*cos(alpha), cos(theta_i)*cos(alpha), -sin(alpha), -d*sin(alpha);
              sin(theta_i)*sin(alpha), cos(theta_i)*sin(alpha),  cos(alpha),  d*cos(alpha);
              0,                       0,                       0,            1];

        T = T * Ti;  % 累积变换
    end
end