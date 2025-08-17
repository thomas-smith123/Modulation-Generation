function filtered_img = hard_threshold_img(img, threshold)
    % 硬阈值滤波应用于图像
    % img: 输入图像或矩阵
    % threshold: 阈值
    
    filtered_img = img;
    filtered_img(abs(img) < threshold) = 0;
end
