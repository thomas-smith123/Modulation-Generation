classdef ADSB
    properties
        map 
        center_frequency
        samples_per_symbol
        total_samples
        upper_frequency
        lower_frequency
        sample_rate
        num_iq_samples
        iq_data
        endPoint
        startPoint
        frame
    end
    methods
        %% 构造函数
        function self = ADSB(sample_rate,num_iq_samples)
            self.map = containers.Map;
            self.map('DF') = 5;
            self.map('CA') = 3;
            self.map('ICAO') = 24;
            self.map('TC') = 5;
            self.map('MSG') = 51;
            self.map('Interrogator') = 24;
            self.num_iq_samples = num_iq_samples;
            self.sample_rate = sample_rate;
            self.center_frequency = 1090e6;
          
            self.samples_per_symbol = 1e-6/(1/self.sample_rate);
            self.total_samples = self.samples_per_symbol*120;
            self.frame = ones(1,112);
            self.upper_frequency = self.center_frequency+0.5e6;
            self.lower_frequency = self.center_frequency-0.5e6;
        end
        %% 生成随机数
        function rand_u = rand_uniform(self,low,high)
            rand_u = low + (high-low)*rand;
        end
        %%
        function u = GenerateFrame(self)
            %#只产生数据位
            cnt = 1;
            for tmp_i=1:self.map.Count
                k = keys(self.map);
                i = k{tmp_i};
                if strcmp(i,'DF')
                    tmp_data = [17,18,18];
                    idx = randperm(length([17,18,18]),1);
                    data = tmp_data(idx);
                    tmp_u = split(dec2bin(data,self.map(k{tmp_i})), '');
                    self.frame(cnt:cnt+self.map(k{tmp_i})-1) = str2double(tmp_u(2:size(tmp_u,1)-1));
                elseif strcmp(i,'CA')
                    tmp_data = [0,1,2,3,4,5,6,7];
                    idx = randperm(length([17,18,18]),1);
                    data = tmp_data(idx);
                    tmp_u = split(dec2bin(data,self.map(k{tmp_i})), '');
                    self.frame(cnt:cnt+self.map(k{tmp_i})-1) = str2double(tmp_u(2:size(tmp_u,1)-1));
                else
                    self.frame(cnt:cnt+self.map(k{tmp_i})-1) = randi([0,1],1,self.map(k{tmp_i}));
                end
                cnt=self.map(k{tmp_i})+cnt;
            end
            u = self.frame;
        end
       %% 组成帧
       function c = reconstruct(self,a)
            c=[];
            w=length(a);
            for j =1:w
                if a(j)==1
                    if isempty(c)
                        c=[1,0];
                    else
                        c = [c,1,0];
                    end
                else
                    if isempty(c)
                        c=[0,1];
                    else
                        c = [c,0,1];
                    end
                end
            end
            header=[1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0];
            header=[header,c];
            c = header; 
        end
       %% 主函数
       function self = call(self)
            u = self.GenerateFrame();
            self.frame = u;
            frame_ = self.reconstruct(u);
            self.frame = repelem(frame_,1,ceil(self.samples_per_symbol/2)); %# ads-b包络情况       

            self.iq_data = zeros(1,self.num_iq_samples);
            self.frame = self.frame';

            position = randi([1,2]);
            tmp_time_domain = zeros(self.num_iq_samples,1);
            if size(self.frame,1) >= self.num_iq_samples % 如果信号长度大于一帧的长度
                if position == 0
                    self.startPoint = randi([0,ceil(self.num_iq_samples*0.5)]);
                    tmp_time_domain(self.startPoint:self.num_iq_samples,:) = self.frame(self.startPoint:self.num_iq_samples,:);
                    self.frame = tmp_time_domain;
                    self.endPoint = self.num_iq_samples;
                elseif position == 1
                    self.startPoint = randi([0,(size(self.frame,1)-self.num_iq_samples)]); % 信号截取的位置
                    self.frame = self.frame(self.startPoint:self.num_iq_samples+self.startPoint-1,:);
                    self.startPoint = 1;self.endPoint = self.num_iq_samples;
                else
                    self.endPoint = randi([0,ceil(self.num_iq_samples*0.5)]); % 信号截取的位置
                    tmp_time_domain(1:self.endPoint,1:size(tmp_time_domain,2)) = self.frame(size(self.frame,1)-self.endPoint+1:size(self.frame,1),1:size(self.frame,2));
                    self.frame = tmp_time_domain;
                    self.startPoint = 1;
                end

            else % 如果信号长度小于一帧的长度
                tmp = floor((self.num_iq_samples-size(self.frame,1))/2);
                self.startPoint = ceil(self.rand_uniform(0,tmp)); % 信号截取的位置
                tmp = zeros(self.num_iq_samples,1);
                if self.startPoint+size(self.frame,1) <= self.num_iq_samples
                    tmp(self.startPoint:self.startPoint+size(self.frame,1)-1,:) = self.frame;
                    self.endPoint = self.startPoint+size(self.frame,1);
                else
                    tmp(self.startPoint:self.num_iq_samples-1,1:size(tmp,2)) = iqdata(1:self.num_iq_samples-self.startPoint);
                    self.endPoint = self.num_iq_samples;
                end
                self.frame = tmp;
            end
            time_tmp = linspace(0,self.num_iq_samples/self.sample_rate,self.num_iq_samples);
            carrier = exp(1j*2*pi*time_tmp*self.center_frequency);
            self.iq_data = carrier.*self.frame';
            % % % plot(abs(self.iq_data))
       end
    end
end