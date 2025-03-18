graphics_toolkit('gnuplot');
clear all
clc

pathl = sprintf("./result");

nprocs = 4;

t_start = 0;
t_end   = 2000000;
t_intvl = 40000;

for i = t_start:t_intvl:t_end
  for ip = 0:1:nprocs-1
    fname = sprintf("%s/prof_sim_params.in_%08d_%d.dat", pathl, i, ip);
    if exist(fname, 'file')
      a1 = importdata(fname);
      a2 = a1(2:end-1, :);

      % 각 prof 파일을 이미지로 저장
      h2 = figure('Visible', 'off'); % 이미지를 표시하지 않도록 설정
      imagesc(a2); axis image; colorbar; colormap jet;
      title(sprintf("prof_sim_params.in_%08d_%d", i, ip));
      saveas(h2, sprintf("%s/prof_sim_params_img_%08d_%d.png", pathl, i, ip));
      close(h2);
    end
  endfor

  % 전체 phi 이미지 생성 및 저장
  clear phi;
  phi = [];
  for ip = 0:1:nprocs-1
    fname = sprintf("%s/prof_sim_params.in_%08d_%d.dat", pathl, i, ip);
    a1 = importdata(fname);
    a2 = a1(2:end-1, :);
    phi = [phi; a2];
  endfor
  
  % phi 데이터를 이미지로 저장
  h1 = figure('Visible', 'off'); % 이미지를 표시하지 않도록 설정
  subplot(1,1,1); imagesc(phi); axis image; colorbar; colormap jet;
  title(sprintf("phi: t = %d", i));
  save([pathl, sprintf("/tot_phi_%08d.dat", i)], "phi", "-ascii");
  saveas(h1, sprintf("%s/tot_img_phi%08d.png", pathl, i));
  close(h1);
endfor

