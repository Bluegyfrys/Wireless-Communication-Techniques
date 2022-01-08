<<<<<<< Updated upstream

%% The code is first copied from "https://www.mathworks.com/matlabcentral/fileexchange/79706-music-multiple-signal-classification-algorithm" with author information
%%%In The Name of GOD%%%
%%%Department of Electrical and Computer Engineering
%%%Isfahan University of Technology-IRAN
%%%By MohammadReza Jabbari-Email: Mo.re.jabbari@gmail.com


%% The code is modified by me  for the described MUSCI algorhtm in the pdf file


clc 
clear
close all
format compact     

%% Parameters
AOA = [-4 4];      %Direction of arrival (Degree)
L   = length(AOA); %The number of paths in the chanel
T   = 500;         %Snapshots (or Samples)
Nr  = 12;          %Number of receiver's antennas 
lambda = 150;      %Wavelength 
d   = lambda/2;    %Receiver's antennas spacing
SNR = 1;           %Signal to Noise Ratio (dB)

%% Channel Matrix
A = zeros(Nr,L);   %Steering Matrix 
for k=1:L 
    A(:,k) = exp(-1j*2*pi*d*sind(AOA(k))*(0:Nr-1)'/lambda); %Assignment matrix 
end 

%% Noise power
noise = sqrt(1/2)*(randn(Nr,T)+1j*randn(Nr,T));

%% Received signal: Noise power =1, SNR= Tx Power / Noise Power
Vj = diag(sqrt((10.^(SNR/10))/2));       % Tx Power = SNR*Noise Power = SNR, Change the SNR from dB to linearvalue and distribut the power in both real and imag part
s = Vj* ( randn(L,T) + 1j*randn(L,T) );  % Received signal


%% Singal transmit model
X = A*s; 
X = X+noise;      %Insert Additive White Gaussain Noise (AWGN) 



%% MUSIC (MUltiple SIgnal Classification)
Rx = cov(X');                     %Data covarivance matrix 
[eigenVec,eigenVal] = eig(Rx);    %Find the eigenvalues and eigenvectors of Rx 
Vn = eigenVec(:,1:Nr-L);          %Estimate noise subspace (Note that eigenvalues sorted ascendig on columns of "eigenVal")
theta = -90:0.05:90;              %Grid points of Peak Search 
for i=1:length(theta) 
    SS = zeros(Nr,1); 
    SS = exp(-1j*2*pi*d*(0:Nr-1)'*sind(theta(i))/lambda);
    PP = SS'*(Vn*Vn')*SS;
    Pmusic(i) = 1/ PP; 
end
Pmusic = real(10*log10(Pmusic)); %Spatial Spectrum function
[pks,locs] = findpeaks(Pmusic,theta,'SortStr','descend','Annotate','extents');
MUSIC_Estim = sort(locs(1:L))

figure;
plot(theta,Pmusic,'-b',locs(1:L),pks(1:L),'r*'); hold on
text(locs(1:L)+2*sign(locs(1:L)),pks(1:L),num2str(locs(1:L)'))

xlabel('Angle \theta (degree)'); ylabel('Spatial Power Spectrum P(\theta) (dB)') 
title('AOA estimation based on MUSIC algorithm ') 
xlim([min(theta) max(theta)])
grid on
=======

%% The code is first copied from "https://www.mathworks.com/matlabcentral/fileexchange/79706-music-multiple-signal-classification-algorithm" with author information
%%%In The Name of GOD%%%
%%%Department of Electrical and Computer Engineering
%%%Isfahan University of Technology-IRAN
%%%By MohammadReza Jabbari-Email: Mo.re.jabbari@gmail.com


%% The code is modified by me  for the described MUSCI algorhtm in the pdf file


clc 
clear
close all
format compact     

%% Parameters
AOA = [-4 4];      %Direction of arrival (Degree)
L   = length(AOA); %The number of paths in the chanel
T   = 500;         %Snapshots (or Samples)
Nr  = 12;          %Number of receiver's antennas 
lambda = 150;      %Wavelength 
d   = lambda/2;    %Receiver's antennas spacing
SNR = 1;           %Signal to Noise Ratio (dB)

%% Channel Matrix
A = zeros(Nr,L);   %Steering Matrix 
for k=1:L 
    A(:,k) = exp(-1j*2*pi*d*sind(AOA(k))*(0:Nr-1)'/lambda); %Assignment matrix 
end 

%% Noise power
noise = sqrt(1/2)*(randn(Nr,T)+1j*randn(Nr,T));

%% Received signal: Noise power =1, SNR= Tx Power / Noise Power
Vj = diag(sqrt((10.^(SNR/10))/2));       % Tx Power = SNR*Noise Power = SNR, Change the SNR from dB to linearvalue and distribut the power in both real and imag part
s = Vj* ( randn(L,T) + 1j*randn(L,T) );  % Received signal


%% Singal transmit model
X = A*s; 
X = X+noise;      %Insert Additive White Gaussain Noise (AWGN) 



%% MUSIC (MUltiple SIgnal Classification)
Rx = cov(X');                     %Data covarivance matrix 
[eigenVec,eigenVal] = eig(Rx);    %Find the eigenvalues and eigenvectors of Rx 
Vn = eigenVec(:,1:Nr-L);          %Estimate noise subspace (Note that eigenvalues sorted ascendig on columns of "eigenVal")
theta = -90:0.05:90;              %Grid points of Peak Search 
for i=1:length(theta) 
    SS = zeros(Nr,1); 
    SS = exp(-1j*2*pi*d*(0:Nr-1)'*sind(theta(i))/lambda);
    PP = SS'*(Vn*Vn')*SS;
    Pmusic(i) = 1/ PP; 
end
Pmusic = real(10*log10(Pmusic)); %Spatial Spectrum function
[pks,locs] = findpeaks(Pmusic,theta,'SortStr','descend','Annotate','extents');
MUSIC_Estim = sort(locs(1:L))

figure;
plot(theta,Pmusic,'-b',locs(1:L),pks(1:L),'r*'); hold on
text(locs(1:L)+2*sign(locs(1:L)),pks(1:L),num2str(locs(1:L)'))

xlabel('Angle \theta (degree)'); ylabel('Spatial Power Spectrum P(\theta) (dB)') 
title('AOA estimation based on MUSIC algorithm ') 
xlim([min(theta) max(theta)])
grid on
>>>>>>> Stashed changes
