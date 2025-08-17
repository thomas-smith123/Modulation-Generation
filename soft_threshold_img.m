function filtered_img = soft_threshold_img(img, threshold)
    % 软阈值滤波应用于图像
    % img: 输入图像或矩阵
    % threshold: 阈值
    
    filtered_img = sign(img) .* max(abs(img) - threshold, 0);
end
