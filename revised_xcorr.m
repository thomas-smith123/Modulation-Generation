function [offset] = revised_xcorr(spectrum)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    % N = length(spectrum);  % 频谱长度
    % spectrum_flipped = flip(spectrum);  % 生成镜像频谱
    % max_similarity = -inf;  % 初始化最大相似度
    % best_offset = 0;        % 初始化最佳偏移量
    % 
    % % 滑动频谱进行相似度比较
    % for shift = -(N-1):(N-1)
    %     % 将频谱向右/向左平移 shift 个位置
    %     if shift >= 0
    %         shifted_spectrum = [zeros(1, shift), spectrum_flipped(1:end-shift)];
    %     else
    %         shifted_spectrum = [spectrum_flipped(-shift+1:end), zeros(1, -shift)];
    %     end
    % 
    %     % 计算当前平移下的相似度，这里用点积作为相似度度量
    %     similarity = sum(spectrum .* shifted_spectrum);
    % 
    %     % 找到最大相似度和对应的偏移量
    %     if similarity > max_similarity
    %         max_similarity = similarity;
    %         best_offset = shift;
    %     end
    % end
    % 
    % offset = best_offset;  % 返回最优偏移量
    N = length(spectrum);
    spectrum_flipped = flip(spectrum);  % 生成镜像频谱

    
    % imf=emd(spectrum);


    rxx = zeros(1, 2*N-1);
    result = zeros(size(spectrum));
    s = zeros(N*3,1);
    s(1:N)=spectrum;
    s
    for j=-(N-1):(N-1)
        for i=1:N
            result(i)=spectrum(i)*s(i-N)+result(i);
        end
    end


    for lag = -(N-1):(N-1)
        for n = 1:N
            if (n+lag > 0) && (n+lag <= N)
                rxx(lag+N) = rxx(lag+N) + spectrum(n) * spectrum_flipped(n+lag);
            end
        end
    end
    plot(rxx((4095-1)/2+1:end));hold on;plot(spectrum);hold off;
    % axis([])
end