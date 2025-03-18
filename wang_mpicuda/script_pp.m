clear all
clc

graphics_toolkit("gnuplot");  % 그래픽 툴킷을 gnuplot로 설정

nprocs = 2;

t_start = 0; 
t_end   = 2000000; 
t_intvl = 100000;

h1 = figure;
for i = t_start:t_intvl:t_end
  clear phi;
  phi = [];
  for ip = 0:1:nprocs-1
    fname = sprintf("prof_%08d_%d.dat", i, ip);
    a1 = importdata(fname);
    a2 = a1(2:end-1, :);
    phi = [phi;a2];

    % 개별 prof 파일의 이미지를 저장
    figure;
    imagesc(a2); axis image; colorbar; colormap jet;
    title(sprintf("prof_%08d_%d", i, ip));
    saveas(gcf, sprintf("prof_img_%08d_%d.png", i, ip));
    close;
  endfor
  subplot(1,2,1); imagesc(phi); axis image; colorbar; colormap jet;
  title(sprintf("phi: t = %08d", i));
  save(sprintf("tot_phi_%08d.dat",i), "phi", "-ascii");
  
  clear com;
  com = [];
  for ip = 0:1:nprocs-1
    fname = sprintf("comp_%08d_%d.dat", i, ip);
    a1 = importdata(fname);
    a2 = a1(2:end-1, :);
    com = [com;a2];

    % 개별 comp 파일의 이미지를 저장
    figure;
    imagesc(a2); axis image; colorbar; colormap jet;
    title(sprintf("comp_%08d_%d", i, ip));
    saveas(gcf, sprintf("comp_img_%08d_%d.png", i, ip));
    close;
  endfor
  subplot(1,2,2); imagesc(com); axis image; colorbar; colormap jet;
  title(sprintf("com: t = %08d", i));
  save(sprintf("tot_com_%08d.dat",i), "com");
  saveas(h1, sprintf("tot_img_phi_com_%08d.png", i));
endfor




